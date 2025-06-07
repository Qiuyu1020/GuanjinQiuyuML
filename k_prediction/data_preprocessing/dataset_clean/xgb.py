import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
import joblib
import os

# === 1. 读取数据 ===
df = pd.read_excel("dataset_augmented.xlsx")

# 元素与离子变量
element_cols = [col for col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl'] if col in df.columns]
ion_cols = [col for col in ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM'] if col in df.columns]

# 替换原始变量（去除 active sites）
raw_vars = [
    'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'u',
    'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO',
    'f(+)', 'f(-)', 'f(0)', 'catalyst dosage', 'PMS dosage',
    'pollutant dosage', 'pH'
]
raw_df = df[raw_vars].reset_index(drop=True)

# 元素、离子特征
X_elem = pd.DataFrame(StandardScaler().fit_transform(df[element_cols]), columns=element_cols)
X_ion = pd.DataFrame(StandardScaler().fit_transform(df[ion_cols]), columns=ion_cols)

# active sites 的 One-Hot 编码列
active_site_cols = [col for col in df.columns if col.startswith("active sites_")]
X_active = df[active_site_cols].reset_index(drop=True)

# 合并所有特征
X_all = pd.concat([raw_df, X_elem, X_ion, X_active], axis=1)
y = df["log2k"]

# === 2. 间隔划分训练测试集 ===
interval = 5
test_indices = np.arange(0, len(X_all), interval)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
X_train = X_all.drop(index=test_indices).reset_index(drop=True)
y_train = y.drop(index=test_indices).reset_index(drop=True)

# === 3. GridSearchCV 调参 ===
param_grid = {
    'n_estimators': [300],
    'max_depth': [5],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
scorer = make_scorer(r2_score)
grid = GridSearchCV(
    estimator=XGBRegressor(
        random_state=42,
        reg_alpha=0.1,     # L1 正则，防止过拟合
        reg_lambda=1.0     # L2 正则，增强泛化能力
    ),
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# === 4. 模型评估 ===
y_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)
print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test     R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test     RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
print(f"Test     MAE : {np.mean(np.abs(y_test - y_pred)):.4f}")


# === 4b. 绘制不同学习率下的测试 R² 表现 ===
learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1]
r2_scores = []

for lr in learning_rates:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred_lr = model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_scores.append(r2_lr)
    print(f"Learning rate: {lr:.2f} --> Test R²: {r2_lr:.4f}")

# 绘图
plt.figure(figsize=(7, 5))
plt.plot(learning_rates, r2_scores, marker='o', linestyle='-', color='blue')
plt.xlabel("Learning Rate")
plt.ylabel("Test R² Score")
plt.title("Learning Rate vs Test R²")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/learning_rate_vs_r2.png", dpi=300)
plt.close()

# === 5. 保存模型与特征名 ===
joblib.dump(best_model, "xgb_best_model.pkl")
joblib.dump(X_all.columns.tolist(), "xgb_feature_names.pkl")

# === 6. 学习曲线 ===
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_all, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring="r2"
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_mean, label="Train R²")
plt.plot(train_sizes, test_mean, label="Test R²")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/learning_curve.png", dpi=300)
plt.close()

# === 7. 预测图与误差图 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("Actual vs Predicted (XGBoost)")
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

# === 8. SHAP 分析 ===
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

# Top 10 特征条形图
mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(10).index.tolist()
mean_shap[top_features].plot(kind="barh", figsize=(8, 5), title="Top 10 Mean SHAP Values")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=300)
plt.close()

# 蜂群图（beeswarm）
explanation = shap.Explanation(
    values=shap_df[top_features].values,
    data=X_test[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=10, show=False)
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", dpi=300)
plt.close()
