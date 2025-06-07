import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import joblib
import os

# === 1. 数据加载与预处理 ===
df = pd.read_excel("dataset_k.xlsx")
element_cols = [col for col in df.columns if col in ['C', 'O', 'S', 'Cu', 'Co']]  # 根据实际列名调整
active_site_cols = [col for col in df.columns if col.startswith("active sites_")]
ion_cols = [col for col in ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM'] if col in df.columns]

raw_vars = [
    'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'u',
    'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO',
    'f(+)', 'f(-)', 'f(0)', 'catalyst dosage', 'PMS dosage',
    'pollutant dosage', 'pH'
]

X_raw = df[raw_vars]
X_elem = df[element_cols]
X_ion = df[ion_cols]
X_active = df[active_site_cols]

# 标准化
scaler = StandardScaler()
X_all = pd.concat([X_raw, X_elem, X_ion, X_active], axis=1)
X_all = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)
y = df["log2k"]

# === 2. 间隔划分 ===
interval = 5
test_indices = np.arange(0, len(X_all), interval)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
X_train = X_all.drop(index=test_indices).reset_index(drop=True)
y_train = y.drop(index=test_indices).reset_index(drop=True)

# === 3. Grid Search 训练 SVR ===
param_grid = {
    "C": [1, 10],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf"]
}
grid = GridSearchCV(SVR(), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# === 4. 模型评估 ===
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test     R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test     RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
print(f"Test     MAE : {np.mean(np.abs(y_test - y_pred)):.4f}")

# === 5. 学习曲线 ===
train_sizes, train_scores, test_scores = learning_curve(
    model, X_all, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring="r2"
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Train R²")
plt.plot(train_sizes, test_mean, label="Test R²")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("SVR Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/svr_learning_curve.png", dpi=300)
plt.close()

# === 6. 预测可视化 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("SVR: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/svr_actual_vs_predicted.png", dpi=300)
plt.close()

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========== 基本假设 ===========
# 已完成 SVR 模型训练：
# best_model 是训练好的 SVR 模型
# X_train, X_test 是 DataFrame
# y_train, y_test 是对应标签
# 你的模型性能已经通过 r2_score、RMSE、MAE 等评估过




# 可选：SHAP 运行较慢时，只用部分样本分析（建议最多几十个）
X_test_sample = X_test.sample(n=50, random_state=42)
X_train_sample = X_train.sample(n=100, random_state=42)
from sklearn.svm import SVR

# 假设 X_train, y_train 已经准备好
best_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
best_model.fit(X_train, y_train)

# =========== 1. SHAP KernelExplainer ===========
explainer = shap.Explainer(best_model.predict, X_train_sample, feature_names=X_train.columns)
shap_values = explainer(X_test_sample)

# =========== 2. 处理 SHAP 值为 DataFrame ===========
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

# =========== 3. 元素特征 & active site 特征合并 ===========
element_cols = [col for col in shap_df.columns if col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl']]
active_site_cols = [col for col in shap_df.columns if col.startswith("active sites_")]

shap_df["Element Composition"] = shap_df[element_cols].abs().sum(axis=1)
shap_df["Active Sites"] = shap_df[active_site_cols].abs().sum(axis=1)

X_display = X_test_sample.copy()
X_display["Element Composition"] = 0
X_display["Active Sites"] = 0

# 删除单个元素特征列
shap_df.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")
X_display.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")

# =========== 4. Top 24 特征图 ===========
mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(24).index.tolist()

os.makedirs("plots", exist_ok=True)

# 条形图
mean_shap[top_features].plot(kind="barh", figsize=(10, 8), title="Top 24 SHAP Features (SVR)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/shap_bar_top24_svr.png", dpi=300)
plt.close()

# 蜂群图
explanation = shap.Explanation(
    values=shap_df[top_features].values,
    data=X_display[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=24, show=False)
plt.tight_layout()
plt.savefig("plots/shap_beeswarm_top24_svr.png", dpi=300)
plt.close()
