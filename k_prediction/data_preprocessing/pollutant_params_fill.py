import pandas as pd
import unicodedata


df_main = pd.read_excel("dataset_filled_elements_ions_0_k_log2k.xlsx")
df_props = pd.read_excel("ppp.xlsx")


def normalize_column_name(col):
    if isinstance(col, str):
        col = unicodedata.normalize("NFKC", col)  # 转为兼容形式
        col = col.replace("−", "-").replace("–", "-").replace("（", "(").replace("）", ")")
        col = col.strip()
    return col

df_main.columns = [normalize_column_name(col) for col in df_main.columns]
df_props.columns = [normalize_column_name(col) for col in df_props.columns]


df_props = df_props.rename(columns={"Unnamed: 0": "Pollutant"})


df_main["Pollutant"] = df_main["Pollutant"].str.strip().str.lower()
df_props["Pollutant"] = df_props["Pollutant"].str.strip().str.lower()


target_columns = df_main.loc[:, "Ehomo":"f(0)"].columns.tolist()


props_subset = df_props[["Pollutant"] + target_columns]
props_dict = props_subset.set_index("Pollutant").to_dict(orient="index")


for idx, row in df_main.iterrows():
    pollutant = row["Pollutant"]
    if pollutant in props_dict:
        for col in target_columns:
            df_main.at[idx, col] = props_dict[pollutant].get(col, None)

# 8. 保存结果
df_main.to_excel("corrected_merged_dataset.xlsx", index=False)
print("saved as corrected_merged_dataset.xlsx")
