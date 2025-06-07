import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import os

# 1. 加载数据
df = pd.read_excel("dataset_k.xlsx")
X = df.drop(columns=["log2k"])
y = df["log2k"]

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 模型训练
model = TabNetRegressor(seed=42, verbose=0)
model.fit(
    X_train_scaled, y_train.values.reshape(-1, 1),
    eval_set=[(X_test_scaled, y_test.values.reshape(-1, 1))],
    max_epochs=200,
    patience=20
)

# 5. 模型评估
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

print(f"Training R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test     R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"Test     RMSE: {mean_squared_error(y_test, y_pred_test) ** 0.5:.4f}")
print(f"Test     MAE : {mean_absolute_error(y_test, y_pred_test):.4f}")

# 6. 可视化并保存
os.makedirs("plots", exist_ok=True)

# 拟合图
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.xlabel("True log2k")
plt.ylabel("Predicted log2k")
plt.title("TabNet Regression")
plt.tight_layout()
plt.savefig("plots/tabnet_fit.png", dpi=300)
plt.close()

# 残差图
residuals = y_test - y_pred_test
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_test, residuals, alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Predicted log2k")
plt.ylabel("Residuals")
plt.title("Residual Plot (TabNet)")
plt.tight_layout()
plt.savefig("plots/tabnet_residuals.png", dpi=300)
plt.close()
