# Applied ML: Classifier Optimization for Churn and Rain Prediction

This repository contains a machine learning project for the OKN1 Task 1, which involves building, evaluating, and optimizing two distinct classification models:

1.  A **Random Forest** model to predict telco customer churn.
2.  A **Bayesian Network** model to predict rain in Australia.

The primary focus of this project was not just initial model creation, but the **optimization process** to improve model performance on imbalanced datasets, specifically by targeting the **F1-score**.

---

## Models and Datasets

### 1. 🌲 Random Forest (Customer Churn)

* **Goal:** Predict whether a customer will churn (leave the company).
* **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
* **Data Prep:** Included encoding categorical data (e.g., 'Yes'/'No' to 1/0), one-hot encoding for multi-value columns, and converting 'TotalCharges' to a numeric type.
* **Validation:** Assessed using accuracy and a confusion matrix.

### 2. 🌧️ Bayesian Network (Rain Prediction)

* **Goal:** Predict the probability of rain on the subsequent day.
* **Dataset:** [Rain Prediction in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) from Kaggle.
* **Data Prep:** Involved feature selection, median/mode imputation for missing values, and **discretization** (grouping continuous values like humidity and pressure into categories like 'Low', 'Medium', 'High').
* **Validation:** Assessed using accuracy and a confusion matrix based on a probability threshold.

---

## Key Focus: Model Optimization

The initial models performed poorly at identifying the target minority class (customers who churn or rainy days). The core of this project was to optimize each model to improve this weakness, using the **F1-score** as the primary success metric.

### Random Forest Optimization (D804_PA_Optimization_RF_Churn_F1)

* **Problem:** The base model had low recall for churners (missed customers who left).
* **Solution:** Used **Random Search** to find better hyperparameters and applied `class_weight='balanced'` to force the model to pay more attention to the minority churn class.
* **Result:** A **15% increase in the F1-score** (from 0.552 to 0.632), successfully improving the model's ability to find churners.

### Bayesian Network Optimization (D804_PA_Optimization_BN_Rain_F1)

* **Problem:** The base model (using a 50% probability cutoff) was bad at predicting rain (low recall).
* **Solution:** Instead of retraining, the **prediction threshold** was optimized. By testing different cutoffs, a threshold of **25% (0.25)** was found to provide the best F1-score.
* **Result:** A **significant increase in the F1-score** (from 0.447 to 0.554) by correctly identifying more rainy days.

---
## Technologies Used

* Python
* scikit-learn
* pandas
* NLTK
* Jupyter Notebook
* Matplotlib / Seaborn

### Install Dependencies
pip install -r requirements.txt

### Activate Virtual Environment
source myenv/bin/activate
python3 -m venv myenv

### Start Jupyter Server
jupyter notebook

### Run Bayesian Network Rain Model
python3 Bayesian_Network_Rain_Model.py

### Run Random Forest Churn Model
python3 Random_Forest_Churn_Model.py