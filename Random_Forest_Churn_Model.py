### Import Libraries
# pandas - works with data tables 
import pandas as pd
# numpy - math and working with lists 
import numpy as np
# train_test_split helps us divide our data into a learning set and a testing set
from sklearn.model_selection import train_test_split
# RandomForestClassifier is the actual model we want to use
from sklearn.ensemble import RandomForestClassifier
# These tools help us check how good our model is
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
# These help us make charts and graphs
import matplotlib.pyplot as plt
import seaborn as sns
# Helps measure time
import time
# This tool helps us search for the best model settings randomly
from sklearn.model_selection import RandomizedSearchCV

### Load the Dataset
try:
    # Read dataset from local directory.
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    # Print error when dataset file cannot be found.
    print("Error: Could not find the dataset file.")

### Exploring the Data
# Show the first 5 rows of the table
print("\nFirst 5 rows of the data:")
print(df.head())

# Get some basic info about each column (like data type and if there are missing values)
# to determine how to approach data engineering.
print("\nInformation about the columns:")
df.info()

# Get some summary statistics for number columns (like average, min, max)
print("\nSummary statistics for numerical columns:")
print(df.describe())

# See how many customers churned vs. stayed
print("\nChurn count:")
print(df['Churn'].value_counts())

### Prepare the Data for the Model
# Models can only consume numbers. Clean up and change some columns.

# Step 4a: Fix 'TotalCharges'
# This column should be numbers (money), but sometimes it has spaces for new customers.
# First, try to turn the column into numbers. If it causes an error (like hitting a space), put 'NaN' (Not a Number).
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now, check if we created any 'NaN' values
print(f"\nNumber of missing TotalCharges: {df['TotalCharges'].isnull().sum()}")

# If there are missing values (NaN), fill them with 0. We assume new customers have 0 total charges.
df['TotalCharges'].fillna(0, inplace=True)
print("Filled missing TotalCharges with 0.")

# Step 4b: Drop CustomerID 
# The customer ID is just a label, it doesn't help predict churn. Let's remove it.
df.drop('customerID', axis=1, inplace=True)
print("Dropped customerID column.")

# Step 4c: Convert 'Yes'/'No' columns to 1/0
# Many columns just have 'Yes' or 'No'. Let's change 'Yes' to 1 and 'No' to 0.
# List of columns to change
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# Also include MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
# but handle 'No phone service' or 'No internet service' separately if needed (pandas get_dummies handles this later)

# Loop through each column in our list
for col in binary_cols:
    # If the column exists in our table
    if col in df.columns:
        # Change 'Yes' to 1 and 'No' to 0
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"Converted column '{col}' to 1s and 0s.")

# Step 4d: Convert other text columns using One-Hot Encoding ---
# Some columns have more than two word options (like payment methods).
# We use 'get_dummies' which creates new columns for each option.
# For example, 'PaymentMethod' might become 'PaymentMethod_Bank transfer', 'PaymentMethod_Credit card', etc.
# These new columns will have 1 or 0.

# Select only the columns that are still text (object type)
categorical_cols = df.select_dtypes(include=['object']).columns

# Use get_dummies to convert them. drop_first=True helps avoid having redundant columns.
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nConverted remaining text columns using one-hot encoding.")

# Let's check the data types again to make sure everything is numbers
print("\nInformation after converting text to numbers:")
df.info()

# Show the first few rows again to see the changes
print("\nFirst 5 rows after data preparation:")
print(df.head())

### Split Data into Features (X) and Target (y)

# We need to separate the information we use for prediction (features) from the answer we want to predict (target).

# The 'Churn' column is our target (y).
y = df['Churn']

# Everything else in the table becomes our features (X).
# axis=1 means we are dropping a column
X = df.drop('Churn', axis=1)

# Show the shapes (how many rows and columns) of our features and target
print(f"\nShape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

### Split Data into Training and Testing Sets
# Now we divide our data into two groups:
# Training set: The model learns from this (usually the bigger part).
# Test set: We use this to check how well the model learned (like a final exam).

# We'll use 80% for training and 20% for testing.
# random_state makes sure the split is the same every time we run the code.
# stratify=y tries to keep the same percentage of churners in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the number of rows in each set
print(f"\nNumber of rows in training set: {X_train.shape[0]}")
print(f"Number of rows in test set: {X_test.shape[0]}")

### Create and Train the Random Forest Model
# It's time to build our Random Forest model and teach it using the training data.

# Create the Random Forest model object.
# n_estimators is the number of decision trees to build (like asking 100 opinions).
# random_state makes the model building repeatable.
# class_weight='balanced' helps if we have way more non churners than churners.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("\nCreated Random Forest model.")

# Record the start time for training
start_time_initial = time.time()

# Train the model using the training features (X_train) and training answers (y_train).
# The model learns the patterns here.
rf_model.fit(X_train, y_train)

# Record the end time for training
end_time_initial = time.time()
training_time_initial = end_time_initial - start_time_initial
print(f"Initial model training complete. Time taken: {training_time_initial:.2f} seconds.")

print("Model training complete.")

### Make Predictions on the Test Set

# Let's use our trained model to predict churn for the customers in the test set.

# Record the start time for inference
start_time_pred_initial = time.time()

# Use the model to predict answers for the test features (X_test)
y_pred_initial = rf_model.predict(X_test)
print("\nMade predictions on the test set.")

# Record the end time for inference
end_time_pred_initial = time.time()
inference_time_initial = end_time_pred_initial - start_time_pred_initial

# Calculate average time per prediction
avg_inference_time_initial = (inference_time_initial / len(X_test)) * 1000 # Convert to milliseconds

print(f"\nMade predictions on the test set with the initial model.")
print(f"Total inference time: {inference_time_initial:.4f} seconds.")
print(f"Average inference time per sample: {avg_inference_time_initial:.4f} ms.")

### Evaluate the Model's Performance

# Checking Predictions Below:

# Calculate the accuracy: (Number of correct predictions) / (Total number of predictions)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
print(f"\nModel Accuracy on Test Set: {accuracy_initial:.4f}") # Show accuracy with 4 decimal places

# Print a more detailed report showing precision, recall, f1 score for each class (Churn=0, Churn=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_initial))

# Show the confusion matrix: correct vs. incorrect predictions
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_initial)
print(cm)

# Let's make the confusion matrix easier to read with a plot
plt.figure(figsize=(6, 4)) # Set the size of the plot

# Use seaborn to create a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # Show numbers, use integer format, blue colors
            xticklabels=['Did Not Churn', 'Churned'], # Labels for columns
            yticklabels=['Did Not Churn', 'Churned']) # Labels for rows
plt.xlabel('Predicted Label') # Label for x-axis
plt.ylabel('True Label') # Label for y-axis
plt.title('Confusion Matrix') # Title for the plot
plt.show() # Display the plot

# Print BENCHMARKS for Initial Model
f1_churn_initial = f1_score(y_test, y_pred_initial, pos_label=1) # F1 for Churn=1
print(f"\nInitial Model Benchmarks:")
print(f"- Accuracy: {accuracy_initial:.4f}")
print(f"- F1 Score (Churn): {f1_churn_initial:.4f}")
print(f"- Training Time: {training_time_initial:.2f} seconds")
print(f"- Avg Inference Time: {avg_inference_time_initial:.4f} ms")

### Optimization using Random Search (Task J)

# Step 10a: Define the settings we want to try 

# Let's try to find better settings (hyperparameters) for our Random Forest model.
# Our goal is to improve the F1-score for predicting churn (class 1).

# We create a 'grid' of possible values for different settings.
param_grid = {
    'n_estimators': [100, 200, 300, 400], # Number of trees in the forest
    'max_depth': [10, 20, 30, None], # How deep each tree can grow (None means no limit)
    'min_samples_split': [2, 5, 10], # Minimum number of samples needed to split a node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples allowed in a leaf node
    'max_features': ['sqrt', 'log2', None] # Number of features to consider for the best split
}
print("Defined parameter grid for Random Search.")

# Step 10b: Set up the Random Search 
# Create a scorer that focuses on the F1 score for the Churn class (label 1)
f1_scorer_churn = make_scorer(f1_score, pos_label=1)

# Create the Random Search object.
# It will try 'n_iter' random combinations from our grid.
# cv=3 means it will use 3-fold cross-validation to check each combination.
# verbose=1 shows some progress messages.
# n_jobs=-1 uses all available computer power (CPU cores).
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'), # Base model
    param_distributions=param_grid, # The grid of settings to try
    n_iter=50, # Number of random combinations to test (increase for better search, but slower)
    cv=3, # Number of cross-validation folds
    scoring=f1_scorer_churn, # What score to optimize for (F1 for churn)
    verbose=1, # Show progress
    random_state=42, # Make the random search repeatable
    n_jobs=-1 # Use all CPU cores
)
print("Set up RandomizedSearchCV.")

# Step 10c: Run the Random Search 
print("Running Random Search... this may take some time.")
start_time_search = time.time()
rf_random_search.fit(X_train, y_train)
end_time_search = time.time()
search_time = end_time_search - start_time_search
print(f"Random Search complete. Time taken: {search_time:.2f} seconds.")

# Step 10d: Get the best settings
# Find out which combination of settings worked best according to the search.
best_params = rf_random_search.best_params_
print(f"\nBest hyperparameters found by Random Search:\n{best_params}")

### Create and Train the Optimized Random Forest Model

# Now we build a new model using the best settings found by the Random Search.

# Create a new model with the best parameters
rf_model_optimized = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
print("\nCreated Optimized Random Forest model.")

# Record the start time for training the optimized model
start_time_optimized = time.time()

# Train this new model
rf_model_optimized.fit(X_train, y_train)

# Record the end time for training
end_time_optimized = time.time()
training_time_optimized = end_time_optimized - start_time_optimized
print(f"Optimized model training complete. Time taken: {training_time_optimized:.2f} seconds.")

### Make Predictions with the Optimized Model

# Use the *new* model to predict churn on the test set.

# Record the start time for inference
start_time_pred_optimized = time.time()

# Make predictions
y_pred_optimized = rf_model_optimized.predict(X_test)

# Record the end time for inference
end_time_pred_optimized = time.time()
inference_time_optimized = end_time_pred_optimized - start_time_pred_optimized

# Calculate average time per prediction
avg_inference_time_optimized = (inference_time_optimized / len(X_test)) * 1000 # Convert to milliseconds

print(f"\nMade predictions on the test set with the optimized model.")
print(f"Total inference time: {inference_time_optimized:.4f} seconds.")
print(f"Average inference time per sample: {avg_inference_time_optimized:.4f} ms.")

### Re-evaluate Optimized Model's Performance (Task J)
# Let's check the benchmarks for our optimized model.

print("\n--- Optimized Model Evaluation ---")
# Calculate accuracy for the new model
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"\nOptimized Model Accuracy on Test Set: {accuracy_optimized:.4f}")

# Print the detailed classification report for the new model
print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred_optimized))

# Show the confusion matrix for the new model
print("\nOptimized Confusion Matrix:")
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
print(cm_optimized)

# Plot the new confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Churn', 'Churned'],
            yticklabels=['Did Not Churn', 'Churned'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Optimized Model Confusion Matrix')
plt.show()

# BENCHMARKS for Optimized Model
f1_churn_optimized = f1_score(y_test, y_pred_optimized, pos_label=1) # F1 for Churn=1
print(f"\nOptimized Model Benchmarks:")
print(f"- Accuracy: {accuracy_optimized:.4f}")
print(f"- F1 Score (Churn): {f1_churn_optimized:.4f}")
print(f"- Training Time: {training_time_optimized:.2f} seconds")
print(f"- Avg Inference Time: {avg_inference_time_optimized:.4f} ms")

# ## 14. Compare Initial vs Optimized Model
# Let's see if the optimization helped based on our benchmarks.
# We display the difference between our initial model and optimized model metrics 
# to make the difference easy to observe.
print(f"Metric             | Initial Model to Optimized Model Changes")
print(f"-------------------|--------------------------------")
print(f"Accuracy           | {accuracy_optimized-accuracy_initial:+.4f}")
print(f"F1 Score (Churn=1) | {f1_churn_optimized-f1_churn_initial:+.4f}")
print(f"Training Time (s)  | {training_time_optimized-training_time_initial:+.2f}")
print(f"Avg Inference (ms) | {avg_inference_time_optimized-avg_inference_time_initial:+.4f}")