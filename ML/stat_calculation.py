import os
import pandas as pd
import numpy as np

# Root of processed data
processed_root = r"F:\polito\term4\sdp project\Sensor_STWIN\processed data\vel-fissa"

# Output list of all stats
all_stats = []

# Loop through label folders
for label in os.listdir(processed_root):
    label_path = os.path.join(processed_root, label)
    if not os.path.isdir(label_path):
        continue

    # Loop through each sample
    for sample in os.listdir(label_path):
        sample_path = os.path.join(label_path, sample)
        if not os.path.isdir(sample_path):
            continue

        # Loop through each sensor CSV
        for csv_file in os.listdir(sample_path):
            if not csv_file.endswith(".csv"):
                continue

            file_path = os.path.join(sample_path, csv_file)
            sensor = csv_file.replace(".csv", "")

            try:
                df = pd.read_csv(file_path)
                if 'Time[s]' in df.columns:
                    df = df.drop(columns=['Time[s]'])

                for column in df.columns:
                    values = df[column].values
                    stats = {
                        'label': label,
                        'sample': sample,
                        'sensor': sensor,
                        'channel': column,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'variance': np.var(values),
                    }
                    all_stats.append(stats)

            except Exception as e:
                print(f"⚠️ Failed to process {file_path}: {e}")

# Convert to DataFrame and save
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv("all_statistics.csv", index=False)
print("✅ All statistics saved to all_statistics.csv")
