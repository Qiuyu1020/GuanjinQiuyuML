import pandas as pd
import numpy as np


file_path = r'D:\DESKTOP\Guanjin_ML\Guanjin-Qiuyu-Machine-Learning\数据集(1).xlsx'


df = pd.read_excel(file_path)
df.to_excel('数据集(1)_backup.xlsx', index=False)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


required_cols = {'C1', 'C0', 't'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"列缺失！必须包含: {required_cols}")

def compute_k(row):
    C1, C0, t = row['C1'], row['C0'], row['t']
    if pd.isnull(C1) or pd.isnull(C0) or pd.isnull(t):
        return np.nan
    if C1 /C0 <= 0 or t == 0:
        return np.nan
    return -np.log(C1 / C0) / t


df['k'] = df.apply(lambda row: compute_k(row), axis=1)


df.to_excel(file_path, index=False)

print(f"处理完成k列已计算并保存到：{file_path}")
