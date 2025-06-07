import pandas as pd
import numpy as np


input_file = "dataset_preprocessed.xlsx"
output_file = "dataset_augmented.xlsx"
num_augmented = 5  # most
noise_std = 0.01   # std

# === read ===
df = pd.read_excel(input_file)

# === add noise ===
noisy_cols = ['SSA', 'catalyst dosage', 'PMS dosage', 'pollutant dosage']

# === keep ===
element_cols = [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W', 'Al', 'Sn',
    'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S', 'Cu', 'Ca', 'Co', 'Ce', 'Cl'
]
active_site_cols = [col for col in df.columns if col.startswith("active site")]
target_col = 'log2k'
frozen_cols = element_cols + active_site_cols + [target_col]
other_cols = [col for col in df.columns if col not in frozen_cols + noisy_cols]


def augment_row(row):
    augmented_rows = []
    for _ in range(num_augmented):
        noisy_part = row[noisy_cols] + np.random.normal(0, noise_std, size=len(noisy_cols))
        combined = pd.concat([
            row[frozen_cols],         # keep
            pd.Series(noisy_part, index=noisy_cols),
            row[other_cols]          # keep
        ])
        augmented_rows.append(combined)
    return augmented_rows
# === apply ===
augmented_data = []
for _, row in df.iterrows():
    augmented_data.extend(augment_row(row))

df_aug = pd.DataFrame(augmented_data)

# combine
df_combined = pd.concat([df, df_aug], ignore_index=True)

# 保存新文件
df_combined.to_excel(output_file, index=False)
print(f"done {len(df)} rows，after  {len(df_combined)} rows。")
print(f"saved as：{output_file}")
