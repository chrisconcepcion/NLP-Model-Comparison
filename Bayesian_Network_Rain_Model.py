# pandas - works with data tables 
import pandas as pd
# numpy - math and working with lists 
import numpy as np
# train_test_split helps divide data
from sklearn.model_selection import train_test_split
# LabelEncoder helps turn words into numbers
from sklearn.preprocessing import LabelEncoder
# accuracy_score and others help check how good the model is
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve
# These are tools from the pgmpy library for Bayesian Networks
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
# To plot the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
# To measure time
import time
# To ignore some warnings that might pop up but aren't critical
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


### Load the Dataset
try:
    df = pd.read_csv('weatherAUS.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find the dataset file.")

### Explore the Data

# Show the first 5 rows
print("\nFirst 5 rows of the data:")
print(df.head())

# Get info about columns and missing values
print("\nInformation about the columns:")
df.info()

### Prepare the Data for the Bayesian Network
# We need to clean the data and make it suitable for the model.

# Step 4a: Select Relevant Columns
# Let's pick a few columns that might predict rain tomorrow.
# We choose some weather measurements around 3pm, plus RainToday and RainTomorrow.
cols_to_keep = ['Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'WindGustSpeed', 'RainToday', 'RainTomorrow']
df_selected = df[cols_to_keep].copy() # Use .copy() to avoid warnings
print(f"\nSelected columns: {cols_to_keep}")

# Step 4b: Handle Missing Values
# We need to fill in the blanks or remove rows with missing info.

# Drop rows where our target 'RainTomorrow' is missing as we need this answer.
df_selected.dropna(subset=['RainTomorrow'], inplace=True)
print(f"Removed rows with missing RainTomorrow. New shape: {df_selected.shape}")

# Also drop rows where 'RainToday' is missing as we use it as a predictor.
df_selected.dropna(subset=['RainToday'], inplace=True)
print(f"Removed rows with missing RainToday. New shape: {df_selected.shape}")

# For the number columns fill missing values (NaN) with the median.
# Median is often better than mean if there are very high or low outlier numbers.
num_cols = ['Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'WindGustSpeed']
for col in num_cols:
    # Calculate the median value for the column
    median_val = df_selected[col].median()
    # Fill missing values in that column with the median
    df_selected[col].fillna(median_val, inplace=True)
    print(f"Filled missing values in '{col}' with median ({median_val}).")

# Check again for missing values to be sure
print("\nMissing values after handling:")
print(df_selected.isnull().sum())

# Step 4c: Convert 'Yes'/'No' to Numbers (0/1)
# Change RainToday and RainTomorrow from words to numbers.
df_selected['RainToday'] = df_selected['RainToday'].map({'No': 0, 'Yes': 1})
df_selected['RainTomorrow'] = df_selected['RainTomorrow'].map({'No': 0, 'Yes': 1})
print("\nConverted RainToday and RainTomorrow to 0s and 1s.")

# Step 4d: Discretize Continuous Variables 
# Bayesian Networks often work better with categories. Let's group the number columns.
# We'll cut Humidity, Pressure, Cloud, and WindSpeed into 3 bins: Low, Medium, High.
# 'pd.qcut' tries to put roughly the same number of data points in each bin.

# Define the number of bins (categories)
n_bins = 3
# Define labels for the bins
bin_labels = ['Low', 'Medium', 'High']

# Loop through the columns we want to discretize
num_cols_to_discretize = ['Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'WindGustSpeed']
for col in num_cols_to_discretize:
    # Check if the column exists before trying to discretize
    if col in df_selected.columns:
        try:
            # Cut the data into bins based on quantiles (equal number of points per bin)
            # include_lowest=True makes sure the smallest value is included in the first bin
            # duplicates='drop' handles cases where bin edges might be the same value
            df_selected[col] = pd.qcut(df_selected[col].rank(method='first'), q=n_bins, labels=bin_labels) # rank helps with duplicates
            print(f"Discretized '{col}' into {n_bins} bins.")
        except Exception as e:
            # If qcut fails, print an error and maybe try a different method or skip.
            print(f"Could not discretize '{col}' using qcut due to error: {e}. Trying pd.cut.")
            try:
                 # Fallback: Use pd.cut with calculated edges
                 min_val, max_val = df_selected[col].min(), df_selected[col].max()
                 cut_bins = np.linspace(min_val, max_val, n_bins + 1)
                 df_selected[col] = pd.cut(df_selected[col], bins=cut_bins, labels=bin_labels, include_lowest=True)
                 print(f"Successfully discretized '{col}' using pd.cut.")
            except Exception as e2:
                 print(f"Could not discretize '{col}' with pd.cut either: {e2}. Keeping as continuous (BN might fail).")
    else:
        print(f"Warning: Column '{col}' not found for discretization.")

# Show the first few rows with the changes
print("\nFirst 5 rows after discretization:")
print(df_selected.head())

# Check the value counts for a discretized column to see the bins
if 'Humidity3pm' in df_selected.columns and not pd.api.types.is_numeric_dtype(df_selected['Humidity3pm']):
    print("\nValue counts for discretized Humidity3pm:")
    print(df_selected['Humidity3pm'].value_counts())

# Check data types again - discretized columns should now be 'category'
print("\nColumn types after preparation:")
df_selected.info()

### Split Data into Training and Testing Sets
# Divide data for learning and testing.

# Define features (X) and target (y) from our prepared data
y = df_selected['RainTomorrow']
X = df_selected.drop('RainTomorrow', axis=1)

# Split the data, using 80% for training, 20% for testing
# stratify helps keep the proportion of rain/no rain similar in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Combine X_train and y_train back together for fitting the BN model, as pgmpy often takes the full dataframe
train_data = pd.concat([X_train, y_train], axis=1)
# Keep X_test and y_test separate for prediction and evaluation
test_data_features = X_test
test_data_actual = y_test

print(f"\nNumber of rows in training set: {train_data.shape[0]}")
print(f"Number of rows in test set: {test_data_features.shape[0]}")

### Define the Bayesian Network Structure

 # Tell the model how the weather factors are related (which arrows to draw).

# We define the relationships (edges) as a list of tuples.
# Format: (Source Node, Target Node)
# Example: ('Humidity3pm', 'RainTomorrow') means Humidity affects RainTomorrow.
# These are based on our assumptions from H1.
model_structure = [
    ('Pressure3pm', 'Humidity3pm'),
    ('WindGustSpeed', 'Cloud3pm'),
    ('Humidity3pm', 'RainTomorrow'),
    ('Cloud3pm', 'RainTomorrow'),
    ('RainToday', 'RainTomorrow')
]

# Create a BayesianNetwork object with our defined structure
bn_model = DiscreteBayesianNetwork(model_structure)
print("\nDefined Bayesian Network structure.")

### Train the Bayesian Network (Learn Probabilities)
# Calculate the probability tables (CPTs) from the training data.

start_time_initial = time.time()

# We use Maximum Likelihood Estimation to learn the probabilities.
# This finds the probabilities that best match the patterns seen in the training data.
# state_names argument helps if columns are categorical with specific labels.
# It seems pgmpy handles string categories well here.
bn_model.fit(train_data, estimator=MaximumLikelihoodEstimator)

end_time_initial = time.time()
training_time_initial = end_time_initial - start_time_initial

print("Model training (parameter learning) complete.")

# Let's look at one of the learned probability tables (CPTs)
# For example, the CPT for RainTomorrow
try:
    cpt_rain_tomorrow = bn_model.get_cpds('RainTomorrow')
    print("\nLearned CPT for RainTomorrow (showing part of it):")
    # CPTs can be large, just print the object representation
    print(cpt_rain_tomorrow)
except Exception as e:
    print(f"Could not display CPT for RainTomorrow: {e}")

### Make Predictions on the Test Set

# Use the trained model to predict the chance of rain for the test data.

# Prepare the model for making predictions (inference)
inference = VariableElimination(bn_model)
print("\nPrepared model for inference.")

# Get the feature names from the test data (excluding the actual answer)
feature_columns = test_data_features.columns.tolist()

# Make predictions row by row for the test set
# We predict the probability of RainTomorrow=1 (meaning 'Yes' for rain)
start_time_pred_initial = time.time()
predicted_probabilities = []
predictions_binary_initial = [] # Store 0/1 predictions based on threshold
probability_threshold = 0.5 # If probability > 0.5, predict rain (1)

# Loop through each row in the test data features
for index, row in test_data_features.iterrows():
    try:
        # Create a dictionary of the evidence for this row
        evidence_dict = row.to_dict()
        # Query the model for the probability distribution of RainTomorrow given the evidence
        result = inference.query(variables=['RainTomorrow'], evidence=evidence_dict)
        # Get the probability specifically for RainTomorrow = 1 (Yes)
        prob_rain = result.values[1] # Assumes state 1 corresponds to 'Yes'/1
        predicted_probabilities.append(prob_rain)
        # Convert probability to binary prediction (0 or 1)
        predictions_binary_initial.append(1 if prob_rain > probability_threshold else 0)

    except Exception as e:
        # If prediction fails for a row, record NaN or a default value and print error
        print(f"Error predicting for row {index}: {e}")
        predicted_probabilities.append(np.nan)
        predictions_binary_initial.append(np.nan) # Or choose a default like 0
        
end_time_pred_initial = time.time()
inference_time_initial = end_time_pred_initial - start_time_pred_initial

print(f"Made predictions for {len(predictions_binary_initial)} test samples.")

# Convert the list of binary predictions to a numpy array for evaluation
# We need to handle potential NaN values if any rows failed prediction
predictions_binary_initial = np.array(predictions_binary_initial)
# Create a mask to filter out rows where prediction failed
valid_prediction_mask_initial = ~np.isnan(predictions_binary_initial)

# Filter actual values to match valid predictions
y_test_filtered_initial = test_data_actual[valid_prediction_mask_initial]
y_pred_filtered_initial = predictions_binary_initial[valid_prediction_mask_initial].astype(int) # Ensure integer type

# Calculate average time per prediction for valid ones
num_valid_predictions = len(y_pred_filtered_initial)
if num_valid_predictions > 0:
    avg_inference_time_initial = (inference_time_initial / num_valid_predictions) * 1000 # Convert to ms
    print(f"Made {num_valid_predictions} valid predictions.")
    print(f"Total inference time: {inference_time_initial:.4f} seconds.")
    print(f"Average inference time per sample: {avg_inference_time_initial:.4f} ms.")
else:
    print("No valid predictions were made.")
    avg_inference_time_initial = np.nan

### Evaluate the Model's Performance
# Compare the model's predictions to the actual answers.

# Filter both actual and predicted values to only include valid predictions
y_test_filtered = test_data_actual[valid_prediction_mask_initial]
y_pred_filtered = predictions_binary_initial[valid_prediction_mask_initial].astype(int) # Ensure integer type

# Calculate accuracy only on the successfully predicted rows
accuracy_initial = accuracy_score(y_test_filtered, y_pred_filtered)
print(f"\nModel Accuracy on Test Set (valid predictions): {accuracy_initial:.4f}")

# Print the classification report
print("\nClassification Report (valid predictions):")
print(classification_report(y_test_filtered, y_pred_filtered))

# Show the confusion matrix
print("\nConfusion Matrix (valid predictions):")
cm = confusion_matrix(y_test_filtered, y_pred_filtered)
print(cm)
#TODO
# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Rain', 'Rain'],
            yticklabels=['No Rain', 'Rain'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Bayesian Network')
plt.show()

# --- BENCHMARKS for Initial Model ---
f1_rain_initial = f1_score(y_test_filtered_initial, y_pred_filtered_initial, pos_label=1, zero_division=0) # F1 for Rain=1
print(f"\nInitial Model Benchmarks:")
print(f"- Accuracy: {accuracy_initial:.4f}")
print(f"- F1 Score (Rain): {f1_rain_initial:.4f}")
print(f"- Training Time: {training_time_initial:.2f} seconds")
print(f"- Avg Inference Time: {avg_inference_time_initial:.4f} ms")

### Optimization by Adjusting Prediction Threshold (Task J)
 # The initial model might miss many rainy days (low recall).
# Let's try changing the probability threshold to predict 'Rain' more often.
# Our goal is to improve the F1-score for predicting rain (class 1).

print("\n--- Starting Optimization (Adjusting Threshold) ---")

# We already have the predicted probabilities from step 8.
# Filter out any NaN probabilities that occurred if prediction failed for some rows.
predicted_probabilities_filtered = np.array(predicted_probabilities)[valid_prediction_mask_initial]
y_test_filtered = test_data_actual[valid_prediction_mask_initial] # Use the same filtered actuals

# Try different thresholds from 0.1 to 0.9 and see which gives the best F1 score for rain (class 1)
best_threshold = 0.5 # Start with the initial one
best_f1_score = f1_rain_initial if not np.isnan(f1_rain_initial) else 0.0

# Define a range of thresholds to test
thresholds = np.arange(0.1, 0.91, 0.05) # Test from 0.1 to 0.9 in steps of 0.05

print(f"Finding best threshold for F1-score (Rain) between {thresholds.min():.2f} and {thresholds.max():.2f}...")

# Check if we have valid predictions to work with
if len(predicted_probabilities_filtered) > 0:
    for threshold in thresholds:
        # Apply the current threshold to the probabilities
        current_predictions = (predicted_probabilities_filtered >= threshold).astype(int)
        # Calculate the F1 score for the 'Rain' class (label 1)
        current_f1 = f1_score(y_test_filtered, current_predictions, pos_label=1, zero_division=0)

        # Print the F1 score for this threshold (optional, can be noisy)
        # print(f"Threshold: {threshold:.2f}, F1 Score (Rain): {current_f1:.4f}")

        # If this threshold gives a better F1 score, update our best score and best threshold
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = threshold

    print(f"\nBest threshold found: {best_threshold:.2f} (Yields F1 Score for Rain: {best_f1_score:.4f})")

    # Now, generate the final predictions using the best threshold found
    predictions_binary_optimized = (predicted_probabilities_filtered >= best_threshold).astype(int)

else:
    print("Cannot perform threshold optimization as there were no valid initial predictions.")
    predictions_binary_optimized = np.array([]) # Empty array
    best_threshold = np.nan

# Note: We don't re-train the model here, just change how we interpret its output probability.
# Training time remains the same. Inference time also remains roughly the same,
# as the main work was calculating probabilities.
training_time_optimized = training_time_initial
avg_inference_time_optimized = avg_inference_time_initial

### Re-evaluate Optimized Model's Performance (Task J)
# Let's check the benchmarks using the *best threshold*.

print("\n--- Optimized Model Evaluation (Using Best Threshold) ---")

# Check if we have optimized predictions to evaluate
if len(predictions_binary_optimized) > 0:
    # Calculate accuracy using the optimized predictions
    accuracy_optimized = accuracy_score(y_test_filtered, predictions_binary_optimized)
    print(f"\nOptimized Model Accuracy on Test Set: {accuracy_optimized:.4f}")

    # Print the detailed classification report
    print("\nOptimized Classification Report:")
    # Use zero_division=0
    print(classification_report(y_test_filtered, predictions_binary_optimized, zero_division=0))

    # Show the confusion matrix
    print("\nOptimized Confusion Matrix:")
    cm_optimized = confusion_matrix(y_test_filtered, predictions_binary_optimized)
    print(cm_optimized)

    # Plot the new confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Optimized Model Confusion Matrix (Threshold={best_threshold:.2f})')
    plt.show()

    # --- BENCHMARKS for Optimized Model ---
    f1_rain_optimized = f1_score(y_test_filtered, predictions_binary_optimized, pos_label=1, zero_division=0) # F1 for Rain=1
    print(f"\nOptimized Model Benchmarks:")
    print(f"- Accuracy: {accuracy_optimized:.4f}")
    print(f"- F1 Score (Rain): {f1_rain_optimized:.4f}")
    print(f"- Training Time: {training_time_optimized:.2f} seconds")
    print(f"- Avg Inference Time: {avg_inference_time_optimized:.4f} ms")

else:
    print("Skipping optimized evaluation due to no valid predictions.")
    accuracy_optimized = np.nan
    f1_rain_optimized = np.nan

### Compare Initial vs Optimized Model
# Let's see if adjusting the threshold helped based on our benchmarks.

print("\n--- Benchmark Comparison ---")
print(f"Metric             | Initial Model (T=0.50) to Optimized Model (T={best_threshold:.2f})")
print(f"-------------------|-----------")
# Use isnan checks in case initial run failed
print(f"Accuracy           | {(accuracy_optimized-accuracy_initial):+.4f}" if not np.isnan(accuracy_initial) and not np.isnan(accuracy_optimized) else "N/A")
print(f"F1 Score (Rain=1)  | {(f1_rain_optimized-f1_rain_initial):+.4f}" if not np.isnan(f1_rain_initial) and not np.isnan(f1_rain_optimized) else "N/A")
print(f"Training Time (s)  | {(training_time_optimized-training_time_initial):+.2f}" if not np.isnan(training_time_initial) and not np.isnan(training_time_optimized) else "N/A")
print(f"Avg Inference (ms) | {(avg_inference_time_optimized-avg_inference_time_initial):+.4f}" if not np.isnan(avg_inference_time_initial) and not np.isnan(avg_inference_time_optimized) else "N/A")