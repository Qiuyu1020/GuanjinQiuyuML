import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
from scipy.optimize import differential_evolution

# === 1. 加载模型与输入 ===
model = joblib.load("xgb_best_model.pkl")
df_input = pd.read_excel("datainput.xlsx")

if df_input.shape[0] != 1:
    raise ValueError("datainput.xlsx 应该只包含一行数据")

fixed_input = df_input.iloc[0]

# === 2. 手动指定要优化的元素字段 ===
element_fields = ["C", "O", "S", "Cu", "Co"]

# === 3. 获取模型输入顺序 ===
model_input_features = model.get_booster().feature_names

# === 4. 构造目标函数 ===
def predict_log2k(element_ratios):
    full_input = fixed_input.copy()

    # 设置指定元素比例
    for i, elem in enumerate(element_fields):
        full_input[elem] = element_ratios[i]

    # 补全模型需要的其他字段（未出现的置为 0）
    for feat in model_input_features:
        if feat not in full_input:
            full_input[feat] = 0.0

    X = full_input[model_input_features].values.reshape(1, -1)
    return -model.predict(X)[0]  # 负号：因为我们用的是最小化

# === 5. 处理约束（总和为 1）通过惩罚项实现 ===
def penalized_objective(x):
    penalty = 1000 * abs(np.sum(x) - 1)  # 强制元素加和为1
    return predict_log2k(x) + penalty

# === 6. 定义搜索边界（每个元素 ∈ [0, 1]）===
bounds = [(0, 1)] * len(element_fields)

# === 7. 全局优化 ===
result = differential_evolution(penalized_objective, bounds, strategy='best1bin', seed=42)

# === 8. 输出结果 ===
if result.success:
    ratios_raw = result.x
    optimal_ratios = ratios_raw / np.sum(ratios_raw)  # 归一化保证和为1
    optimal_log2k = -predict_log2k(optimal_ratios)

    print("全局优化成功！")
    print("最优元素比例（归一化后）：")
    for elem, ratio in zip(element_fields, optimal_ratios):
        print(f"  {elem}: {ratio:.4f}")
    print(f"\n预测最大 log2k：{optimal_log2k:.4f}")
else:
    print("优化失败：", result.message)
