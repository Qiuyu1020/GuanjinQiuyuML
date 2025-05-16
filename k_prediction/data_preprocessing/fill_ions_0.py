import pandas as pd


df = pd.read_excel("PollutantEncode.xlsx")


target_columns = ["Cl-", "HCO3-", "HPO42-", "NO3-", "HA"]


for col in target_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)


df.to_excel("kdatafill0.xlsx", index=False)

print("save as kdatafill0.xlsx")
