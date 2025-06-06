import pandas as pd


df = pd.read_excel("dataset_filled_elements_0.xlsx")


target_columns = ["Cl-", "HCO3-", "HPO42-", "NO3-", "HA", "NOM"]


for col in target_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)


df.to_excel("dataset_filled_elements_ions_0.xlsx", index=False)

print("saved as dataset_filled_elements_ions_0.xlsx")
