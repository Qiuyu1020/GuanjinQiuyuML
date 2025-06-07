import pandas as pd


df = pd.read_excel("dataset_kkk.xlsx")


log2k_col = [col for col in df.columns if "log2k" in col.lower()]
if not log2k_col:
    raise ValueError("not found log2k column")
log2k_col = log2k_col[0]


df_clean = df.dropna(subset=[log2k_col])


df_clean.to_excel("dataset_kdone.xlsx", index=False)

print(f" dataset_kdone.xlsx， {len(df_clean)} 。")
