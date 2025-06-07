import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import joblib
import os

# === 1. 加载数据 ===
df = pd.read_excel("dataset_k.xlsx")

# === 2. 特征准备 ===
element_cols = [col for col in df.columns if col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl'
]]
ion_cols = [col for col in df.columns if col in ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM']]
raw_vars = [
    'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'u',
    'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO',
    'f(+)', 'f(-)', 'f(0)', 'catalyst dosage', 'PMS dosage',
    'pollutant dosage', 'pH'
]
active_site_cols = [col for col in df.columns if col.startswith("active sites_")]

# 标准化 & 合并
scaler = StandardScaler()
X_elem = pd.DataFrame(scaler.fit_transform(df[element_cols]), columns=element_cols)
X_ion = pd.DataFrame(scaler.fit_transform(df[ion_cols]), columns=ion_cols)
X_raw = df[raw_vars].reset_index(drop=True)
X_active = df[active_site_cols].reset_index(drop=True)

X_all = pd.concat([X_raw, X_elem, X_ion, X_active], axis=1)
y = df["log2k"]

# === 3. 训练测试集划分 ===
interval = 5
test_indices = np.arange(0, len(X_all), interval)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
X_train = X_all.drop(index=test_indices).reset_index(drop=True)
y_train = y.drop(index=test_indices).reset_index(drop=True)

# === 4. MLP + GridSearch ===
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'alpha': [0.0001, 0.001,0.01,0.00005],
    'learning_rate_init': [0.001, 0.01,0.0005]
}
grid = GridSearchCV(
    estimator=MLPRegressor(max_iter=1000, random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# === 5. 模型评估 ===
y_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)
print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test     R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test     RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
print(f"Test     MAE : {np.mean(np.abs(y_test - y_pred)):.4f}")

# === 6. 保存模型 ===
joblib.dump(best_model, "mlp_best_model.pkl")
joblib.dump(X_all.columns.tolist(), "mlp_feature_names.pkl")

# === 7. 学习曲线 ===
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_all, y, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 5)
)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train R²")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test R²")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve (MLP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/mlp_learning_curve.png", dpi=300)
plt.close()


# === 7. 预测图与误差图 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("Actual vs Predicted (MLP)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.close()


errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=20, edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/error_distribution.png", dpi=300)
plt.close()

# === MLP SHAP 分析 ===
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 采样部分训练集做 SHAP baseline
X_train_sample = X_train.sample(n=100, random_state=42)

# 1. 构建 KernelExplainer
explainer = shap.Explainer(best_model.predict, X_train_sample)

# 2. 计算 SHAP 值
shap_values = explainer(X_test)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

# 3. 合并元素类和 Active Sites 特征
element_cols = [col for col in shap_df.columns if col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl']]
active_site_cols = [col for col in shap_df.columns if col.startswith("active sites_")]

shap_df["Element Composition"] = shap_df[element_cols].abs().sum(axis=1)
shap_df["Active Sites"] = shap_df[active_site_cols].abs().sum(axis=1)

X_display = X_test.copy()
X_display["Element Composition"] = 0
X_display["Active Sites"] = 0

shap_df.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")
X_display.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")

# 4. Top 24 特征
mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(24).index.tolist()

# 5. 条形图
os.makedirs("plots", exist_ok=True)
mean_shap[top_features].plot(kind="barh", figsize=(10, 8), title="Top 24 Mean SHAP Values (Grouped)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/mlp_shap_bar_top24.png", dpi=300)
plt.close()

# 6. 蜂群图
explanation = shap.Explanation(
    values=shap_df[top_features].values,
    data=X_display[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=24, show=False)
plt.tight_layout()
plt.savefig("plots/mlp_shap_beeswarm_top24.png", dpi=300)
plt.close()
