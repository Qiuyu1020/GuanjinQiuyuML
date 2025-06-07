import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("dataset_nan_filled.xlsx")


numeric_df = df.select_dtypes(include=["float64", "int64"])


corr_matrix = numeric_df.corr(method="spearman")


plt.figure(figsize=(22, 18))


sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.6})

# 6. 添加标题
plt.title("Spearman Correlation Matrix", fontsize=16)
plt.tight_layout()

# 7. 保存为高分辨率图片
plt.savefig("spearman_correlation_heatmap.png", dpi=300)

# 8. 显示图像
plt.show()
