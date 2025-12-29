# Binary Classification with SVM, Logistic Regression, and KNN

While the .ipynb file is available, it is recommended to run it on Google Colab. Link : [https://colab.research.google.com/drive/1DWJE2fATjiYWMgSAARvZMyTDzx9K2Pdn?usp=sharing]

This repository presents a comparative machine learning analysis using **Support Vector Machines (SVM)**, **Logistic Regression**, and **k-Nearest Neighbors (KNN)** to predict a binary outcome variable from a set of numeric features. The project emphasizes **proper preprocessing**, **reproducible modeling pipelines**, and **transparent evaluation** using multiple performance metrics.

The implementation and results are documented in detail in the accompanying technical report.

---

## Project Overview

The goal of this project is to evaluate and compare several standard supervised learning models on the same dataset under consistent experimental conditions. All available input variables are used to predict a single binary target variable.

Key methodological principles:
- Consistent **80/20 train–test split**
- **Z-score standardization** applied correctly within pipelines
- Hyperparameter tuning via **GridSearchCV**
- Evaluation using **accuracy**, **ROC AUC**, and **confusion matrices**

---

## Models Implemented

### Support Vector Machine (SVM)
- Implemented using an `sklearn` Pipeline with `StandardScaler`
- Hyperparameters tuned via `GridSearchCV`:
  - `kernel`: `linear`, `rbf`
  - `C`: `[0.001, 0.01, 1, 5, 25, 50]`
  - `gamma`: `[0.001, 0.01, 0.1, 0.5, 1, 2, 5]`
- Reports:
  - Train and test accuracy
  - ROC AUC on test data
  - Confusion matrix on test data

### Logistic Regression
- Implemented using an `sklearn` Pipeline with `StandardScaler`
- Uses the same train/test split as SVM for direct comparability
- Reports:
  - Train and test accuracy
  - ROC AUC on test data
  - Confusion matrix on test data

### k-Nearest Neighbors (KNN)
- Implemented using an `sklearn` Pipeline with `StandardScaler`
- Hyperparameter tuning via `GridSearchCV` over `n_neighbors`
- Uses the same train/test split as the other models
- Reports:
  - Train and test accuracy
  - ROC AUC on test data
  - Confusion matrix on test data

---

## Evaluation & Findings

All three models are evaluated using the same metrics to ensure a fair comparison. Results show that while tuned models such as SVM and KNN can capture more complex decision boundaries, **Logistic Regression provides competitive performance with significantly greater interpretability**.

A detailed comparison of model performance, tradeoffs, and final recommendations is provided in the technical report included in this repository.

---

## Repository Contents

- **`MLFA25_PredictXY.ipynb`**  
  End-to-end Google Colab notebook containing:
  - Data loading and preprocessing
  - Pipeline construction
  - Hyperparameter tuning
  - Model evaluation and visualization

- **`MLFA25 - PredictXY Report.pdf`**  
  Formal report describing:
  - Methodology
  - Model results
  - Comparative analysis
  - Final recommendation

---

## Running the Project (Google Colab)

This project is designed to be run in **Google Colab**.
Link: https://colab.research.google.com/drive/1DWJE2fATjiYWMgSAARvZMyTDzx9K2Pdn?usp=sharing


### Dataset note
The notebook currently expects the dataset to be loaded from Google Drive. You may either:
- Mount your Google Drive in Colab and place the dataset at the specified path, **or**
- Upload the CSV file directly to Colab and update the `pd.read_csv()` path accordingly

---

## Tools & Libraries

- Python (Google Colab)
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `plotnine`

---

## References

- Scikit-learn documentation
- Dataset source (linked in notebook and report)

