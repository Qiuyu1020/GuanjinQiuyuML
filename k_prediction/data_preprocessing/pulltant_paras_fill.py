import pandas as pd

# 读取 Excel 文件
ppp = pd.read_excel("ppp.xlsx", header=None)
dataset = pd.read_excel("dataset_filled_elements_ions_0_k_log2k.xlsx")  # 你可以替换为实际文件名

import pandas as pd



# ---------- 2. 提取污染物参数名与数据 ----------
pollutant_params_ppp = ppp.iloc[0, 1:].astype(str).tolist()
pollutant_names_ppp = ppp.iloc[1:, 0].astype(str).str.lower().tolist()
ppp_data = ppp.iloc[1:, 1:]

# ---------- 3. 建立污染物 → 参数字典（按参数名） ----------
pollutant_param_map = {
    name: dict(zip(pollutant_params_ppp, list(row._asdict().values())))
    for name, row in zip(pollutant_names_ppp, ppp_data.itertuples(index=False))
}

# ---------- 4. 构造 dataset 参数列名标准化映射 ----------
# 原始 dataset 的参数列名（在 "Pollutant" 列之后）
pollutant_col_index = dataset.columns.get_loc("Pollutant")
dataset_param_cols = dataset.columns[pollutant_col_index + 1:]

# 构造标准化列名映射（忽略大小写和空格）
standardized_col_map = {
    col.lower().replace(" ", ""): col
    for col in dataset_param_cols
}

# ---------- 5. 填入参数值（按列名匹配） ----------
for i, row in dataset.iterrows():
    pollutant_name = str(row["Pollutant"]).lower()
    if pollutant_name in pollutant_param_map:
        param_dict = pollutant_param_map[pollutant_name]
        for p_name_raw, value in param_dict.items():
            p_name_std = p_name_raw.lower().replace(" ", "")
            if p_name_std in standardized_col_map:
                true_col = standardized_col_map[p_name_std]
                dataset.at[i, true_col] = value

# ---------- 6. 保存结果 ----------
dataset.to_excel("dataset_with_pollutant_parameters.xlsx", index=False)
print("✅ 已按参数名正确填入，保存为 dataset_with_pollutant_parameters.xlsx")


import pandas as pd

# === 1. 读取数据 ===
dataset = pd.read_excel("dataset_filled_elements_ions_0_k_log2k.xlsx")
ppp = pd.read_excel("ppp.xlsx")

# === 2. 清洗列名：全角负号 `−` 替换为半角 `-`，并统一大小写与空格 ===
def normalize_column_names(df):
    return df.rename(columns=lambda x: x.replace("−", "-").strip())

dataset = normalize_column_names(dataset)
ppp = normalize_column_names(ppp)

# === 3. 确保 "Pollutant" 和 "f(-)" 列存在 ===
assert "Pollutant" in dataset.columns, "dataset 缺少 Pollutant 列"
assert "Pollutant" in ppp.columns, "ppp 缺少 Pollutant 列"
assert "f(-)" in ppp.columns, "ppp 缺少 f(-) 列"

# === 4. 构建污染物到 f(-) 的映射字典 ===
pollutant_to_fneg = ppp[["Pollutant", "f(-)"]].dropna().drop_duplicates()
pollutant_to_fneg_dict = dict(zip(pollutant_to_fneg["Pollutant"], pollutant_to_fneg["f(-)"]))

# === 5. 将 f(-) 填入 dataset ===
dataset["f(-)"] = dataset["Pollutant"].map(pollutant_to_fneg_dict)

# === 6. 保存结果（可选）===
dataset.to_excel("dataset_with_f_neg_filled.xlsx", index=False)

print("✅ f(-) 已成功填入 dataset 中并保存为 dataset_with_f_neg_filled.xlsx")
