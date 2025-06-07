import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import shap
import joblib
import os

# === 1. 读取数据 ===
df = pd.read_excel("dataset_k.xlsx")

# 特征列
element_cols = [col for col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl'] if col in df.columns]
ion_cols = [col for col in ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM'] if col in df.columns]
raw_vars = [
    'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'u',
    'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO',
    'f(+)', 'f(-)', 'f(0)', 'catalyst dosage', 'PMS dosage',
    'pollutant dosage', 'pH'
]
raw_df = df[raw_vars].reset_index(drop=True)
X_elem = pd.DataFrame(StandardScaler().fit_transform(df[element_cols]), columns=element_cols)
X_ion = pd.DataFrame(StandardScaler().fit_transform(df[ion_cols]), columns=ion_cols)
active_site_cols = [col for col in df.columns if col.startswith("active sites_")]
X_active = df[active_site_cols].reset_index(drop=True)

X_all = pd.concat([raw_df, X_elem, X_ion, X_active], axis=1)
y = df["log2k"]

# === 2. 划分数据集（规则间隔法） ===
interval = 5
test_indices = np.arange(0, len(X_all), interval)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
X_train = X_all.drop(index=test_indices).reset_index(drop=True)
y_train = y.drop(index=test_indices).reset_index(drop=True)

# === 3. GridSearchCV 调参 ===
param_grid = {
    'n_estimators': [1, 2, 3],
    'max_depth': [1, 2, None],
    'max_features': ['sqrt']
}
scorer = make_scorer(r2_score)
grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
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

# === 5. 保存模型 ===
joblib.dump(best_model, "rf_best_model.pkl")
joblib.dump(X_all.columns.tolist(), "rf_feature_names.pkl")

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
os.makedirs("plots_rf", exist_ok=True)
plt.savefig("plots_rf/learning_curve.png", dpi=300)
plt.close()

# === 7. 预测图和误差图 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("Actual vs Predicted (RF)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_rf/actual_vs_predicted.png", dpi=300)
plt.close()

errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=20, edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_rf/error_distribution.png", dpi=300)
plt.close()

# === 8. SHAP 分析（合并元素类与 Active Sites 特征）===
explainer = shap.Explainer(best_model, X_test)
shap_values = explainer(X_test, check_additivity=False)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

shap_df["Element Composition"] = shap_df[element_cols].abs().sum(axis=1)
shap_df["Active Sites"] = shap_df[active_site_cols].abs().sum(axis=1)

X_display = X_test.copy()
X_display["Element Composition"] = X_test[element_cols].sum(axis=1)
X_display["Active Sites"] = X_test[active_site_cols].sum(axis=1)

shap_df.drop(columns=element_cols + active_site_cols, inplace=True)
X_display.drop(columns=element_cols + active_site_cols, inplace=True)

mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(24).index.tolist()

mean_shap[top_features].plot(kind="barh", figsize=(10, 8), title="Top 24 Mean SHAP Values (RF)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots_rf/shap_bar_top24.png", dpi=300)
plt.close()

explanation = shap.Explanation(
    values=shap_df[top_features].values,
    data=X_display[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=24, show=False)
plt.tight_layout()
plt.savefig("plots_rf/shap_beeswarm_top24.png", dpi=300)
plt.close()
