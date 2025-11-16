# NLP Model Comparison for Sentiment Analysis

## Project Summary

This project is a complete, end-to-end NLP workflow that builds, evaluates, and compares two machine learning models for sentiment analysis. The goal is to determine the most effective algorithm for classifying product review sentiment. The project begins with a raw dataset of 1,000 reviews, applies a full text preprocessing pipeline, and concludes with a detailed performance analysis of both models.

## Core Process

1.  **Data Preprocessing:** The text data was cleaned and standardized using the NLTK library. The process included lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization.

2.  **Feature Extraction:** Cleaned text was converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization method.

3.  **Model Training (Baseline):** A baseline model was trained using Logistic Regression.

4.  **Model Training (Comparison):** A second model, LinearSVC (Linear Support Vector Classifier), was trained on the same data for a direct performance comparison.

5.  **Evaluation:** Both models were evaluated using a full suite of metrics including accuracy, precision, recall, F1-score, and a confusion matrix.

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