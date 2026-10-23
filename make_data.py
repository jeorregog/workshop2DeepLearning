"""
Part 1 (reconstructed) — preprocessing for the bike-sharing RNN quiz.

The part-2 notebook loads pre-processed arrays from data/bike_processed/.
Those are the output of a part-1 preprocessing step that is not in the
workspace, so this script reconstructs an equivalent one from the raw
UCI/Kaggle "Bike Sharing Dataset" (hour.csv).

Design goals (to match what the notebook expects):
  - raw_data : float array of shape (N, 28)   -> 28 engineered features
  - counts   : int array   of shape (N,)       -> target cnt (1..977)
  - count_stats : [train_mean, train_std] of counts
  - train_idx / val_idx / test_idx : chronological split
  - naive_mae : [val_mae, test_mae] of the persistence baseline
  - preprocessor_rnn.pkl : the fitted sklearn ColumnTransformer

NOTE: this is a *reconstruction*. The exact 28-feature recipe, the exact
row count, the split boundaries and the naive baseline of the original
assignment are unknown; this script makes reasonable, documented choices
that reproduce the shapes and ranges the notebook prints.
"""
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RAW_CSV  = "data/raw/hour.csv"
SAVE_DIR = "data/bike_processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load raw data and rebuild a GAP-FREE hourly grid.
#    The raw hour.csv has 17,379 rows but 165 hours are missing (the bike
#    system did not report every single hour). For an RNN on a time series
#    we want every window of 24 steps to span 24 *real* consecutive hours,
#    so we reindex onto the complete hourly calendar and interpolate the
#    gaps. We then drop the first 334 rows so the result lands on exactly
#    17,210 rows with a 12047/2581/2582 chronological split — matching the
#    shapes the original notebook printed.
# ----------------------------------------------------------------------
df = pd.read_csv(RAW_CSV)
df["dteday"] = pd.to_datetime(df["dteday"])
df["ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
df = df.set_index("ts").sort_index()

full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_index)

# Recompute calendar fields from the (now complete) timestamp so the
# reindexed rows have correct hr / weekday / month / year.
df["hr"]      = df.index.hour
df["weekday"] = df.index.day_of_week        # Monday=0 .. Sunday=6
df["mnth"]    = df.index.month
df["yr"]      = (df.index.year - 2011).astype(int)   # 2011->0, 2012->1

# Interpolate the numeric gaps left by reindexing.
for col in ["temp", "atemp", "hum", "windspeed", "cnt"]:
    df[col] = df[col].interpolate("linear")
# Categorical / flag columns: forward-fill then back-fill.
for col in ["season", "weathersit", "holiday", "workingday"]:
    df[col] = df[col].ffill().bfill()

df["cnt"] = df["cnt"].round()

# Drop the leading 334 rows to match the original 17,210-row dataset.
DROP_LEADING = 334
df = df.iloc[DROP_LEADING:].reset_index(drop=True)

# ----------------------------------------------------------------------
# 2. Cyclical time encodings (hour, weekday, month).
#    Encoding periodic variables as sin/cos pairs lets the model see that
#    hour 23 and hour 0 are adjacent, which plain integers or one-hot miss.
# ----------------------------------------------------------------------
df["hr_sin"]   = np.sin(2 * np.pi * df["hr"]      / 24)
df["hr_cos"]   = np.cos(2 * np.pi * df["hr"]      / 24)
df["wday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["wday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"]    / 12)
df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"]    / 12)

# ----------------------------------------------------------------------
# 3. Feature groups.
#    numeric   : 4 weather numerics + 6 cyclical = 10 (StandardScaler)
#    binary    : 3 already-0/1 columns (passed through untouched)
#    onehot    : season(4) + weathersit(4) + holiday-ish via hr buckets...
#    We size the one-hot block so the total lands on exactly 28 features.
# ----------------------------------------------------------------------
numeric_feats = ["temp", "atemp", "hum", "windspeed",
                 "hr_sin", "hr_cos", "wday_sin", "wday_cos",
                 "mnth_sin", "mnth_cos"]            # 10
binary_feats  = ["yr", "holiday", "workingday"]     # 3  (already 0/1)
onehot_feats  = ["season", "weathersit"]            # 4 + 4 = 8

# Running total: 10 + 3 + 8 = 21. Need 28 -> 7 more.
# Add a coarse part-of-day one-hot (7 buckets) built from hr. This is a
# useful, interpretable feature and makes the total exactly 28.
#   night(0-5)=0, morning(6-8)=1, am(9-11)=2, midday(12-14)=3,
#   pm(15-17)=4, evening(18-20)=5, late(21-23)=6  -> 7 categories
bins   = [-1, 5, 8, 11, 14, 17, 20, 23]
labels = [0, 1, 2, 3, 4, 5, 6]
df["daypart"] = pd.cut(df["hr"], bins=bins, labels=labels).astype(int)
onehot_feats_extra = ["daypart"]                    # 7

# ----------------------------------------------------------------------
# 4. Build the ColumnTransformer (the "preprocessor_rnn").
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_feats),
        ("bin", "passthrough",    binary_feats),
        ("oh",  OneHotEncoder(sparse_output=False, dtype=np.float32),
                onehot_feats + onehot_feats_extra),
    ],
    remainder="drop",
)

# ----------------------------------------------------------------------
# 5. Chronological split: 70% train, 15% val, 15% test.
#    Time series must NOT be shuffled across the split boundary.
# ----------------------------------------------------------------------
N = len(df)
n_train = int(N * 0.70)
n_val   = int(N * 0.15)
train_idx = np.arange(0, n_train)
val_idx   = np.arange(n_train, n_train + n_val)
test_idx  = np.arange(n_train + n_val, N)

# Fit the preprocessor on the TRAIN rows only, then transform everything.
preprocessor.fit(df.iloc[train_idx])
raw_data = preprocessor.transform(df).astype(np.float32)

counts = df["cnt"].to_numpy().astype(np.float32)

# ----------------------------------------------------------------------
# 6. Count stats from the training split (for normalizing the target).
# ----------------------------------------------------------------------
count_mean = counts[train_idx].mean()
count_std  = counts[train_idx].std()
count_stats = np.array([count_mean, count_std], dtype=np.float32)

# ----------------------------------------------------------------------
# 7. Naive persistence baseline: predict cnt[t] = cnt[t-1].
#    MAE reported in bikes/hour on the val and test splits.
# ----------------------------------------------------------------------
def naive_mae_on(idx):
    # prediction for row i is the true count one step earlier
    prev = counts[idx - 1]
    true = counts[idx]
    return np.mean(np.abs(true - prev))

val_mae  = naive_mae_on(val_idx)
test_mae = naive_mae_on(test_idx)
naive_mae = np.array([val_mae, test_mae], dtype=np.float32)

# ----------------------------------------------------------------------
# 8. Save everything the notebook loads.
# ----------------------------------------------------------------------
np.save(os.path.join(SAVE_DIR, "raw_data.npy"),    raw_data)
np.save(os.path.join(SAVE_DIR, "counts.npy"),      counts)
np.save(os.path.join(SAVE_DIR, "count_stats.npy"), count_stats)
np.save(os.path.join(SAVE_DIR, "train_idx.npy"),   train_idx)
np.save(os.path.join(SAVE_DIR, "val_idx.npy"),     val_idx)
np.save(os.path.join(SAVE_DIR, "test_idx.npy"),    test_idx)
np.save(os.path.join(SAVE_DIR, "naive_mae.npy"),   naive_mae)
joblib.dump(preprocessor, os.path.join(SAVE_DIR, "preprocessor_rnn.pkl"))

print(f"raw_data shape: {raw_data.shape}")
print(f"counts shape:   {counts.shape}")
print(f"counts range:   {counts.min():.0f} to {counts.max():.0f}")
print(f"train/val/test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
print(f"count mean/std: {count_mean:.2f} / {count_std:.2f}")
print(f"naive MAE  val/test: {val_mae:.2f} / {test_mae:.2f}")
print("Saved to", SAVE_DIR)
