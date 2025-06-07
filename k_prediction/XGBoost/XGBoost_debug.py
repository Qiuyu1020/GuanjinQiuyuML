import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
import joblib


df = pd.read_csv("datasetcsv.csv", encoding="utf-8-sig")
correlations = df.corr(numeric_only=True)["log2k"].sort_values(ascending=False)
print(correlations.head(20))