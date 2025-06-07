import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. 数据读取与预处理 ===
df = pd.read_excel("dataset_k.xlsx")
y = df["log2k"]
X = df.drop(columns=["log2k"])

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 2. DNN 模型构建 ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# === 3. 模型训练 ===
early_stop = EarlyStopping(patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=200, batch_size=32, verbose=1,
                    callbacks=[early_stop])

# === 4. 评估模型 ===
y_pred = model.predict(X_test).flatten()
y_train_pred = model.predict(X_train).flatten()

print(f"Training R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test     R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test     RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
print(f"Test     MAE : {mean_absolute_error(y_test, y_pred):.4f}")

# === 5. 可视化部分 ===
os.makedirs("plots", exist_ok=True)

# 学习曲线
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Curve")
plt.legend()
plt.savefig("plots/dnn_learning_curve.png", dpi=300)
plt.close()

# 预测 vs 真实
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log2k")
plt.ylabel("Predicted log2k")
plt.title("DNN: Actual vs Predicted")
plt.tight_layout()
plt.savefig("plots/dnn_actual_vs_predicted.png", dpi=300)
plt.close()

# 残差分布图
plt.hist(y_test - y_pred, bins=20, edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.tight_layout()
plt.savefig("plots/dnn_error_distribution.png", dpi=300)
plt.close()

# === 6. SHAP 分析 ===
background = X_train.sample(100, random_state=0)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test, nsamples=100)

shap_df = pd.DataFrame(shap_values[0], columns=X.columns)
mean_shap = shap_df.abs().mean()
top_features = mean_shap.sort_values(ascending=False).head(24).index.tolist()

# 条形图
mean_shap[top_features].plot(kind="barh", figsize=(10, 8), title="Top 24 SHAP Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/dnn_shap_bar.png", dpi=300)
plt.close()

# 蜂群图
shap.summary_plot(shap_values[0], X_test, feature_names=X.columns, max_display=24, show=False)
plt.tight_layout()
plt.savefig("plots/dnn_shap_beeswarm.png", dpi=300)
plt.close()
