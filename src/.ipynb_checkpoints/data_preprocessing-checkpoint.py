import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/train_transaction.csv")

df = df.dropna(thresh=len(df) * 0.5, axis=1)

df = pd.get_dummies(df, drop_first=True)

df.fillna(df.mean(), inplace=True)

X = df.drop(columns=["isFraud"]) 
y = df["isFraud"]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled Training Data: {X_train_resampled.shape}, {y_train_resampled.shape}")

processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data["isFraud"] = y

print("Processed DataFrame Info:")
print(processed_data.info())

processed_data.to_csv("data/processed_data.csv", index=False)

print("✅ Data Preprocessing Complete. File saved at 'data/processed_data.csv'")
