# IEEE Fraud Detection using Genetic Algorithm (GA)

## Project Overview
This project applies **Genetic Algorithms (GA)** for feature selection and optimization on the **IEEE-CIS Fraud Detection dataset**. We use GA to identify the most relevant features for fraud detection, improving efficiency and interpretability.
Document - https://docs.google.com/document/d/1K3E69GaKV6Heqa25tK27rFNepTufOJQl5uROI_lT2OA/edit?tab=t.g19al5vmlj5k

## Repository Structure
```
IEEE-CISFraud
 ┣ 📂 data
 ┃ ┣ 📜 train_transaction.csv   (Not included, download from Kaggle)
 ┃ ┣ 📜 train_identity.csv      (Optional, for additional features)
 ┃ ┣ 📜 processed_data.csv      (Processed dataset for feature selection)
 ┃ ┗ 📜 selected_features.txt   (Features selected using GA)
 ┣ 📂 notebooks
 ┃ ┣ 📜 01_data_preprocessing.ipynb
 ┃ ┗ 📜 02_feature_selection_ga.ipynb
 ┣ 📂 src
 ┃ ┣ 📜 data_preprocessing.py
 ┃ ┣ 📜 genetic_algorithm_feature_selection.py
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┗ 📜 LICENSE
```

##  Dataset
The dataset comes from the **IEEE-CIS Fraud Detection Challenge** on Kaggle.
- **Download it here:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- Place the **`train_transaction.csv`** file inside the `data/` folder.

## Setup Instructions
### 1️ Clone the repository
```bash
git clone https://github.com/yourusername/IEEE-CISFraud.git
cd IEEE-CISFraud
```

### 2️ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️ Run Data Preprocessing
```bash
python src/data_preprocessing.py
```

### 4️ Run Genetic Algorithm for Feature Selection
```bash
python src/genetic_algorithm_feature_selection.py
```

##  Results & Evaluation
- **Feature Selection:** GA identifies the most important fraud detection features.
- **Processed Data:** Stored in `data/processed_data.csv`.
- **Selected Features:** Saved in `data/selected_features.txt`.

##  Next Steps
 Experiment with different selection methods in GA (Roulette, Rank, Elitism).
 Compare GA-selected features with full dataset performance.
 Optimize GA parameters (mutation rate, crossover probability).

##  License
This project is licensed under the **MIT License**.

---
🚀 **Contributions welcome!** If you have improvements, feel free to submit a pull request.
