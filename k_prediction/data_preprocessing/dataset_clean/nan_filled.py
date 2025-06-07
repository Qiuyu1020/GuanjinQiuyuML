import pandas as pd
from sklearn.impute import KNNImputer


df = pd.read_excel("dataset_kdone.xlsx")


df["catalyst dosage"] = df["catalyst dosage"].fillna(df["catalyst dosage"].median())


numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()


assert "pH" in numeric_cols, "'pH' not found"
assert "SSA" in numeric_cols, "'SSA' not found"


imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


df.to_excel("dataset_nan_filled.xlsx", index=False)
print("done saved as dataset_knn_filled.xlsx")
