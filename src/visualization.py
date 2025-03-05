import pandas as pd
import matplotlib.pyplot as plt

df_smote = pd.read_csv("data/processed_data_smote.csv", usecols=["isFraud"])

fraud_counts = df_smote["isFraud"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(["Not Fraud (0)", "Fraud (1)"], fraud_counts, color=['blue', 'red'])
plt.title("Fraud vs Non-Fraud Transactions After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("data/fraud_distribution_smote.png")
print("Fraud Distribution Plot Saved: data/fraud_distribution_smote.png")
