import pandas as pd
import numpy as np
import os

# Ensure the directory exists
data_dir = 'simulated_data'
os.makedirs(data_dir, exist_ok=True)

# Read the original data
original_data = pd.read_csv('Fraud.csv', delimiter=";")

# Ensure 'amount' column is numeric, coercing errors to NaN
original_data['amount'] = pd.to_numeric(original_data['amount'], errors='coerce')

# Handle NaN values if there are any
if original_data['amount'].isnull().any():
    median_amount = original_data['amount'].median()
    original_data['amount'].fillna(median_amount, inplace=True)

# Remove 'isFraud' and 'isFlaggedFraud' columns for simulated future data
original_data.drop(['isFraud', 'isFlaggedFraud'], axis=1, inplace=True)


def introduce_variation(data, week):
    np.random.seed(week)
    data['amount'] *= (1 + np.random.normal(0, 0.1, size=data.shape[0]))
    if week % 6 == 0:
        data['oldbalanceOrg'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
        data['newbalanceOrig'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
    return data

# Generate data for each week
all_data = pd.DataFrame()
for week in range(1, 53):
    sampled_data = original_data.sample(n=1000, random_state=week).copy()
    simulated_data = introduce_variation(sampled_data, week)
    simulated_data["week"] = week  # Add week column
    all_data = pd.concat([all_data, simulated_data])

# Save all data to a single CSV file
file_path = os.path.join(data_dir, 'simulated_data_year.csv')
all_data.to_csv(file_path, index=False, sep=";")
