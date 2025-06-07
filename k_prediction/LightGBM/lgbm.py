import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import joblib
import os

# === 1. 读取数据 ===
df = pd.read_excel("dataset_k.xlsx")

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
import re
from collections import Counter

# 清洗特征名：替换非法字符，并确保唯一性
def clean_and_unique_columns(columns):
    cleaned = [re.sub(r'[^\w]', '_', col) for col in columns]
    counter = Counter()
    unique_cols = []
    for col in cleaned:
        if counter[col]:
            unique_cols.append(f"{col}_{counter[col]}")
        else:
            unique_cols.append(col)
        counter[col] += 1
    return unique_cols

X_all.columns = clean_and_unique_columns(X_all.columns)
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
    'colsample_bytree': [0.8],
}
scorer = make_scorer(r2_score)
grid = GridSearchCV(
    estimator=LGBMRegressor(
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0
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

# === 5. 学习率 vs R² 图 ===
learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1]
r2_scores = []
for lr in learning_rates:
    model = LGBMRegressor(
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
    r2_scores.append(r2_score(y_test, y_pred_lr))
    print(f"Learning rate: {lr:.2f} --> Test R²: {r2_scores[-1]:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(learning_rates, r2_scores, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Test R² Score")
plt.title("Learning Rate vs Test R²")
plt.grid(True)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lgbm_learning_rate_vs_r2.png", dpi=300)
plt.close()

# === 6. 保存模型与特征名 ===
joblib.dump(best_model, "lgbm_best_model.pkl")
joblib.dump(X_all.columns.tolist(), "lgbm_feature_names.pkl")

# === 7. 学习曲线 ===
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_all, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring="r2"
)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure()
plt.plot(train_sizes, train_mean, label="Train R²")
plt.plot(train_sizes, test_mean, label="Test R²")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve (LightGBM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/lgbm_learning_curve.png", dpi=300)
plt.close()

# === 8. 预测图与误差图 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("Actual vs Predicted (LightGBM)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/lgbm_actual_vs_predicted.png", dpi=300)
plt.close()

errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=20, edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/lgbm_error_distribution.png", dpi=300)
plt.close()

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

# 合并元素类特征与 Active Sites 特征
element_cols = [col for col in shap_df.columns if col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl']]
active_site_cols = [col for col in shap_df.columns if col.startswith("active sites_")]

# 组合 SHAP 值（按绝对值）
shap_df["Element Composition"] = shap_df[element_cols].abs().sum(axis=1)
shap_df["Active Sites"] = shap_df[active_site_cols].abs().sum(axis=1)

# 为了 SHAP beeswarm 图保留 X 数据结构
X_display = X_test.copy()

# 使用 sum 或 mean 作为代表性 feature value 来给 beeswarm 图着色
X_display["Element Composition"] = 0
X_display["Active Sites"] = 0

# 删除原始元素类与 active sites 特征列
shap_df.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")
X_display.drop(columns=element_cols + active_site_cols, inplace=True, errors="ignore")

# 计算平均 SHAP 值并获取前 24 个重要特征
mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(60).index.tolist()

# === Top 24 SHAP 条形图 ===
mean_shap[top_features].plot(kind="barh", figsize=(10, 8), title="Top 24 Mean SHAP Values (Grouped)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/shap_bar_top24_grouped.png", dpi=300)
plt.close()

# === SHAP 蜂群图（Top 24） ===
explanation = shap.Explanation(
    values=shap_df[top_features].values,
    data=X_display[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=60, show=False)
plt.tight_layout()
plt.savefig("plots/shap_beeswarm_top24_grouped.png", dpi=300)
plt.close()
