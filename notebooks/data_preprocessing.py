import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load dataset (Ensure 'train_transaction.csv' is in your working directory)
df = pd.read_csv("train_transaction.csv")

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled Training Data: {X_train_resampled.shape}, {y_train_resampled.shape}")
