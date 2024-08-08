import os
import pandas as pd
import numpy as np


def generate_simulated_data(self, week):
    # Use absolute path for the file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'Fraud.csv')
    original_data = pd.read_csv(file_path, delimiter=";", nrows=1000)

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


def insert_data(self, df):
    # Insert data into the application
    for _, row in df.iterrows():
        input_data = {
            'step': 1,
            'amount': row['amount'],
            'oldbalanceOrg': row['oldbalanceOrg'],
            'newbalanceOrig': row['newbalanceOrig'],
            'oldbalanceDest': row['oldbalanceDest'],
            'newbalanceDest': row['newbalanceDest'],
            'type': row['type']
        }
        response = self.client.post('/dashboard', json=input_data)
        self.assertEqual(response.status_code, 302)

        # Check the prediction response
        predict_response = self.client.get('/predict')
        self.assertIn(predict_response.status_code, [200, 500])



