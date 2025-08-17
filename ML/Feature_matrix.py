import pandas as pd

# Load your statistics file
df = pd.read_csv(r"F:\polito\term4\sdp project\all_statistics.csv")

# Build a feature name: sensor_channel_stat (e.g., IIS3DWB_Z_mean)
df['channel_clean'] = df['channel'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
df['feature'] = df['sensor'] + "_" + df['channel_clean'] + "_" + df.columns[4]

# Pivot to wide format: one row per (label, sample), one column per feature
feature_matrix = df.pivot_table(index=['label', 'sample'], columns='feature', values=df.columns[4])

# Flatten multi-index columns
feature_matrix.columns = feature_matrix.columns.to_flat_index()
feature_matrix.columns = [col if isinstance(col, str) else "_".join(col) for col in feature_matrix.columns]

# Reset index
feature_matrix = feature_matrix.reset_index()

# Replace bad values with NaN
feature_matrix.replace(['#NAME?', 'inf', '=-inf'], pd.NA, inplace=True)

# Option 1 (safe): fill missing values with 0
feature_matrix.fillna(0, inplace=True)

# Option 2 (if 0 distorts data): use column mean
# feature_matrix.fillna(feature_matrix.mean(), inplace=True)

# Save to CSV
feature_matrix.to_csv("feature_matrix.csv", index=False)
print("✅ Feature matrix saved to feature_matrix.csv")
