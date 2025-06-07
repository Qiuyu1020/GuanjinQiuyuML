import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("dataset_nan_filled.xlsx")


if 'active sites' in df.columns:
    df = pd.get_dummies(df, columns=['active sites'], drop_first=True)


element_cols = ['Fe', 'Mn', 'Zn', 'Mg', 'Bi', 'V', 'Zr', 'Na', 'Ni', 'Ru', 'La', 'Mo',
                'W', 'Al', 'Sn', 'Si', 'Ti', 'B', 'C', 'N', 'H', 'O', 'P', 'K', 'F',
                'Ag', 'S', 'Cu', 'Ca', 'Co', 'Ce', 'Cl']


X = df.drop(columns=["log2k"])
y = df["log2k"]

element_free_cols = [col for col in X.columns if col not in element_cols]
scaler = StandardScaler()
X_scaled_part = pd.DataFrame(scaler.fit_transform(X[element_free_cols]),
                             columns=element_free_cols, index=X.index)


X_element_part = X[element_cols]


X_final = pd.concat([X_element_part, X_scaled_part], axis=1)
df_final = X_final.copy()
df_final["log2k"] = y


df_final.to_excel("dataset_preprocessed.xlsx", index=False)
print("dataset_preprocessed.xlsx")
