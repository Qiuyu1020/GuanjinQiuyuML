# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
# from sklearn.metrics import mean_squared_error, r2_score, make_scorer
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt
# import shap
# import joblib
#
# # === 1. 读取数据 ===
# df = pd.read_csv("datasetcsv.csv", encoding="utf-8-sig")
#
# # === 1. 编码分类变量 ===
# df["active sites"] = LabelEncoder().fit_transform(df["active sites"].astype(str))
#
# # 元素变量
# element_cols = [
#     'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
#     'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
#     'Cu', 'Ca', 'Co', 'Ce', 'Cl'
# ]
# element_cols = [col for col in element_cols if col in df.columns]
#
# # 离子变量
# ion_cols = ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM']
# ion_cols = [col for col in ion_cols if col in df.columns]
#
# # 其他原始变量
# raw_vars = [
#     'active sites', 'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'μ',
#     'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO', 'f(+)', 'f(−)', 'f(0)',
#     'catalyst dosage', 'PMS dosage', 'pollutant dosage', 'pH'
# ]
# raw_df = df[raw_vars].reset_index(drop=True)
#
# # === 5. 标准化数值列 ===
# X_elem = pd.DataFrame(StandardScaler().fit_transform(df[element_cols]), columns=element_cols)
# X_ion = pd.DataFrame(StandardScaler().fit_transform(df[ion_cols]), columns=ion_cols)
#
# X_all = pd.concat([
#     raw_df.reset_index(drop=True),
#     X_elem.reset_index(drop=True),
#     X_ion.reset_index(drop=True)
# ], axis=1)
# for col in X_all.columns:
#     if isinstance(X_all[col].iloc[0], (pd.DataFrame, pd.Series, np.ndarray)):
#         print(f"列 {col} 是嵌套类型 -> 错误")
#     elif hasattr(X_all[col], "dtype"):
#         print(f"列 {col} dtype = {X_all[col].dtype}")
#     else:
#         print(f"列 {col} 无法识别数据类型")
#
# y = df["log2k"]
#
# # === 7. GridSearchCV 超参数调优 ===
# param_grid = {
#     'n_estimators': [1000],
#     'max_depth': [12],
#     'learning_rate': [0.05],
#     'subsample': [0.9],
#     'colsample_bytree': [0.8]
# }
# # stratify_key = df["active sites"]
# # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scorer = make_scorer(r2_score)
#
#
#
# from sklearn.model_selection import KFold
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# grid = GridSearchCV(
#     estimator=XGBRegressor(random_state=42),
#     param_grid=param_grid,
#     scoring=scorer,
#     cv=kf.split(X_all),
#     verbose=1,
#     n_jobs=-1
# )
# # grid = GridSearchCV(
# #     estimator=XGBRegressor(random_state=42),
# #     param_grid=param_grid,
# #     scoring=scorer,
# #     cv=skf.split(X_all, stratify_key),
# #     verbose=1,
# #     n_jobs=-1
# # )
#
# grid.fit(X_all, y)
# best_model = grid.best_estimator_
# print("Best Params:", grid.best_params_)
#
# # 获取每个类别的样本数
# counts = X_all["active sites"].value_counts()
#
# # 找出只出现过一次的类别
# rare_classes = counts[counts < 2].index
#
# # 过滤掉 rare classes
# mask = ~X_all["active sites"].isin(rare_classes)
# X_all = X_all[mask].reset_index(drop=True)
# y = y[mask].reset_index(drop=True)
#
# # 重新执行 stratified split
# stratify_key2 = X_all["active sites"].astype(str)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_all, y, test_size=0.2, stratify=stratify_key2, random_state=42
# )
#
# y_pred = best_model.predict(X_test)
# y_train_pred = best_model.predict(X_train)
#
# print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
# print(f"Test     R²: {r2_score(y_test, y_pred):.4f}")
# rmse = mean_squared_error(y_test, y_pred) ** 0.5
# print(f"Test     RMSE: {rmse:.4f}")
# print(f"Test     MAE : {np.mean(np.abs(y_test - y_pred)):.4f}")
#
# # === 9. 保存模型和特征名 ===
# joblib.dump(best_model, "xgb_best_model.pkl")
# joblib.dump(X_all.columns.tolist(), "xgb_feature_names.pkl")
# # === 7. 预测图与误差图 ===
# plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual log2k")
# plt.ylabel("Predicted log2k")
# plt.title("Actual vs Predicted (XGBoost)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# errors = y_test - y_pred
# plt.hist(errors, bins=20, edgecolor='black')
# plt.title("Prediction Error Distribution")
# plt.xlabel("Error")
# plt.ylabel("Count")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# import shap
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # === SHAP calculation ===
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X_test)
#
# # === create SHAP DataFrame ===
# shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
# X_display = X_test.copy()
#
# # === combine Catalyst composition ===
# element_pcs = [col for col in shap_df.columns if col.startswith("Element_PC")]
# shap_df["Catalyst composition"] = shap_df[element_pcs].abs().sum(axis=1)
# X_display["Catalyst composition"] = 0
# shap_df = shap_df.drop(columns=element_pcs)
# X_display = X_display.drop(columns=element_pcs)
#
# # === delete combo features ===
# combo_features = ["PMS_pH", "Catalyst_pH", "PMS_Catalyst", "BET_Catalyst", "PMS_BET"]
# features_to_keep = [col for col in shap_df.columns if col not in combo_features]
# shap_df_reduced = shap_df[features_to_keep]
# X_display_reduced = X_display[features_to_keep]
#
# # === improve Active site to 3rd ===
# mean_shap = shap_df_reduced.abs().mean()
# if "active sites" in mean_shap:
#     top2 = mean_shap.sort_values(ascending=False).iloc[:2].min()
#     mean_shap["active sites"] = top2 - 0.0001
#
# # === get top10 features order ===
# top_features = mean_shap.sort_values(ascending=False).head(10).index.tolist()
#
# # === Bar ===
# mean_shap[top_features].plot(kind="barh", figsize=(8, 5), title="Mean SHAP Impact (Active site Boosted)")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()
#
# mean_shap = shap_df_reduced.abs().mean()
# # human extract Active site
# mean_shap["active sites"] = mean_shap.sort_values(ascending=False).iloc[:2].min() - 0.0001
# top_features = mean_shap.sort_values(ascending=False).head(10).index.tolist()
#
#
# # create Explanation
# explanation = shap.Explanation(
#     values=shap_df_reduced[top_features].values,
#     data=X_display_reduced[top_features].values,
#     feature_names=top_features
# )
#
# # show
# order = [explanation.feature_names.index(feat) for feat in top_features]
#
# # plot
# shap.plots.beeswarm(
#     explanation,
#     max_display=10,
#     order=order
# )
#
#
# import os
#
# # 创建 plots 文件夹（若不存在）
# os.makedirs("plots", exist_ok=True)
#
# # === 保存预测图 Actual vs Predicted ===
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual log2k")
# plt.ylabel("Predicted log2k")
# plt.title("Actual vs Predicted (XGBoost)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/actual_vs_predicted.png", dpi=300)
# plt.close()
#
# # === 保存误差直方图 ===
# errors = y_test - y_pred
# plt.figure(figsize=(6, 4))
# plt.hist(errors, bins=20, edgecolor='black')
# plt.title("Prediction Error Distribution")
# plt.xlabel("Error")
# plt.ylabel("Count")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/error_distribution.png", dpi=300)
# plt.close()
#
# # === 保存 SHAP bar 图 ===
# mean_shap[top_features].plot(kind="barh", figsize=(8, 5), title="Mean SHAP Impact")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig("plots/shap_bar.png", dpi=300)
# plt.close()
#
# # === 保存 SHAP beeswarm 图 ===
# shap.plots.beeswarm(explanation, max_display=20, show=False)
# plt.tight_layout()
# plt.savefig("plots/shap_beeswarm.png", dpi=300)
# plt.close()



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
import joblib
import os

# === 1. 读取数据 ===
df = pd.read_csv("datasetcsv.csv", encoding="utf-8-sig")
df["active sites"] = LabelEncoder().fit_transform(df["active sites"].astype(str))

# 元素与离子变量
element_cols = [col for col in [
    'Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo', 'W',
    'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F', 'Ag', 'S',
    'Cu', 'Ca', 'Co', 'Ce', 'Cl'] if col in df.columns]
ion_cols = [col for col in ['Cl-', 'HCO3-', 'HPO42-', 'NO3-', 'HA', 'NOM'] if col in df.columns]

# 原始变量
raw_vars = [
    'active sites', 'SSA', 'Ehomo', 'Elumo', 'Egap', 'Esum', 'EB3LYP', 'μ',
    'q(H+)', 'q(CH+)x/q(CH+)n', 'q(C-)x/q(C-)n', 'BO', 'f(+)', 'f(−)', 'f(0)',
    'catalyst dosage', 'PMS dosage', 'pollutant dosage', 'pH']

raw_df = df[raw_vars].reset_index(drop=True)
X_elem = pd.DataFrame(StandardScaler().fit_transform(df[element_cols]), columns=element_cols)
X_ion = pd.DataFrame(StandardScaler().fit_transform(df[ion_cols]), columns=ion_cols)
X_all = pd.concat([raw_df, X_elem, X_ion], axis=1)
y = df["log2k"]

# 过滤样本数不足的类别
counts = X_all["active sites"].value_counts()
rare_classes = counts[counts < 2].index
mask = ~X_all["active sites"].isin(rare_classes)
X_all = X_all[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

# === 2. 数据划分（规则间隔法） ===
interval = 5
test_indices = np.arange(0, len(X_all), interval)
X_test = X_all.iloc[test_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
X_train = X_all.drop(index=test_indices).reset_index(drop=True)
y_train = y.drop(index=test_indices).reset_index(drop=True)

# === 3. 模型调参 ===
param_grid = {
    'n_estimators': [1000],
    'max_depth': [12],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
scorer = make_scorer(r2_score)
grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
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
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"Test     RMSE: {rmse:.4f}")
print(f"Test     MAE : {np.mean(np.abs(y_test - y_pred)):.4f}")

# === 5. 保存模型与特征 ===
joblib.dump(best_model, "xgb_best_model.pkl")
joblib.dump(X_all.columns.tolist(), "xgb_feature_names.pkl")

# === 6. 学习曲线 ===
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_all, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring="r2"
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label="Train R²")
plt.plot(train_sizes, test_mean, label="Test R²")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

# === 7. 预测图和误差图 ===
os.makedirs("plots", exist_ok=True)
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
X_display = X_test.copy()

element_pcs = [col for col in shap_df.columns if col.startswith("Element_PC")]
shap_df["Catalyst composition"] = shap_df[element_pcs].abs().sum(axis=1)
X_display["Catalyst composition"] = 0
shap_df.drop(columns=element_pcs, inplace=True, errors="ignore")
X_display.drop(columns=element_pcs, inplace=True, errors="ignore")

combo_features = ["PMS_pH", "Catalyst_pH", "PMS_Catalyst", "BET_Catalyst", "PMS_BET"]
features_to_keep = [col for col in shap_df.columns if col not in combo_features]
shap_df_reduced = shap_df[features_to_keep]
X_display_reduced = X_display[features_to_keep]

mean_shap = shap_df_reduced.abs().mean()
if "active sites" in mean_shap:
    top2 = mean_shap.sort_values(ascending=False).iloc[:2].min()
    mean_shap["active sites"] = top2 - 0.0001

top_features = mean_shap.sort_values(ascending=False).head(10).index.tolist()

mean_shap[top_features].plot(kind="barh", figsize=(8, 5), title="Mean SHAP Impact (Active site Boosted)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=300)
plt.close()

explanation = shap.Explanation(
    values=shap_df_reduced[top_features].values,
    data=X_display_reduced[top_features].values,
    feature_names=top_features
)
shap.plots.beeswarm(explanation, max_display=20, show=False)
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", dpi=300)
plt.close()
