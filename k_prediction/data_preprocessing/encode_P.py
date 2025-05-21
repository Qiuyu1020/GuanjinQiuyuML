import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_excel("dataset.xlsx")


if "Pollutant" in df.columns:
    le = LabelEncoder()
    df["Pollutant"] = le.fit_transform(df["Pollutant"].astype(str).str.lower())
    print("'Pollutant' encode done")
    print("class set as：")
    for label, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {label} → {code}")
else:
    raise KeyError("fail")


df.to_excel("PollutantEncode.xlsx", index=False)
print("save as：PollutantEncode.xlsx")

