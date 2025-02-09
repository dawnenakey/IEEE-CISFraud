# IEEE-CISFraud

📌 IEEE Fraud Detection using Genetic Algorithm (GA) & Stochastic Gradient Descent (SGD)

🚀 Project Overview

This project applies Genetic Algorithms (GA) for feature selection and Stochastic Gradient Descent (SGD) for fraud classification on the IEEE-CIS Fraud Detection dataset.

📂 Repository Structure

📦 ieee-fraud-detection-ga-sgd
 ┣ 📂 data
 ┃ ┣ 📜 train_transaction.csv   (Not included, download from Kaggle)
 ┃ ┣ 📜 train_identity.csv      (Optional, for additional features)
 ┃ ┗ 📜 processed_data.csv      (Processed dataset for feature selection & training)
 ┣ 📂 notebooks
 ┃ ┣ 📜 01_data_preprocessing.ipynb
 ┃ ┣ 📜 02_feature_selection_ga.ipynb
 ┃ ┣ 📜 03_fraud_classification_sgd.ipynb
 ┃ ┗ 📜 04_model_tuning.ipynb
 ┣ 📂 src
 ┃ ┣ 📜 data_preprocessing.py
 ┃ ┣ 📜 genetic_algorithm_feature_selection.py
 ┃ ┣ 📜 train_sgd_classifier.py
 ┃ ┣ 📜 evaluate_model.py
 ┃ ┗ 📜 hyperparameter_tuning.py
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┗ 📜 LICENSE

 📌 Dataset

The dataset comes from the IEEE-CIS Fraud Detection Challenge on Kaggle.

Download it here: Kaggle IEEE-CIS Fraud Detection

Place the train_transaction.csv file inside the data/ folder.

🔧 Setup Instructions

1. Clone the repository
git clone https://github.com/yourusername/ieee-fraud-detection-ga-sgd.git
cd ieee-fraud-detection-ga-sgd

2. Install dependencies
pip install -r requirements.txt

3. Run Data Preprocessing
python src/data_preprocessing.py

4. Run Genetic Algorithm for Feature Selection
python src/genetic_algorithm_feature_selection.py

5. Train SGD Classifier
python src/train_sgd_classifier.py

6. Run Hyperparameter Tuning
python src/hyperparameter_tuning.py

Results & Evaluation

Feature Selection: GA selects the best fraud-detection features.

SGD Model Performance: Evaluated with AUC-ROC, confusion matrix.

Hyperparameter Tuning: Optimized loss function, regularization, and learning rate.

📌 Next Steps

✅ Experiment with different selection methods in GA (Roulette, Rank).
✅ Compare SGD with other models (Random Forest, XGBoost).
✅ Improve fraud detection in real-time.

📜 License

This project is licensed under the MIT License.

🚀 Contributions welcome! If you have improvements, feel free to submit a pull request.
