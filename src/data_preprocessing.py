import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load dataset (Ensure 'train_transaction.csv' is in your working directory)
df = pd.read_csv("data/train_transaction.csv")

# Drop columns with more than 50% missing values
df = df.dropna(thresh=len(df) * 0.5, axis=1)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Fill remaining NaNs with mean
df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.drop(columns=["isFraud"])  # Features
y = df["isFraud"]  # Target

# Normalize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"✅ Resampled Training Data: {X_resampled.shape}, {y_resampled.shape}")

# ✅ Save resampled data
processed_data_smote = pd.DataFrame(X_resampled, columns=X.columns)
processed_data_smote["isFraud"] = y_resampled

# ✅ Debug: Print DataFrame Info Before Saving
print("✅ Processed SMOTE DataFrame Info:")
print(processed_data_smote.info())

# ✅ Save the new balanced dataset
processed_data_smote.to_csv("data/processed_data_smote.csv", index=False, chunksize=100000)
print("✅ SMOTE Applied! Balanced file saved at 'data/processed_data_smote.csv'")

# ✅ Save original processed data (without SMOTE)
processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data["isFraud"] = y
processed_data.to_csv("data/processed_data.csv", index=False)
print("✅ Original Preprocessing Complete. File saved at 'data/processed_data.csv'")
