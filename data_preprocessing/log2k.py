import pandas as pd
import numpy as np


df = pd.read_excel("k.xlsx")


def compute_log2k(x):
    if pd.notna(x) and isinstance(x, (int, float)) and x > 0:
        return np.log2(x)
    else:
        return ""


df["log2k"] = df["k"].apply(compute_log2k)


df.to_excel("log2k.xlsx", index=False)

print("saved as：log2k.xlsx")