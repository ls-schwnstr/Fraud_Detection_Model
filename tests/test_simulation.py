import unittest
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from app import app
import time
import os
import numpy as np

# Mocking MLflow functions for testing
mlflow.set_tracking_uri("http://localhost:5004")

DATABASE_URL = 'sqlite:///fraud_detection.db'
engine = create_engine(DATABASE_URL, echo=True)


def generate_fake_data():
    # Use absolute path for the file
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, '..', 'app', 'Fraud.csv')

    # Load original data
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

    # Generate data for each week
    all_data = pd.DataFrame()
    for week in range(1, 53):
        sampled_data = original_data.sample(n=2, random_state=week).copy()
        simulated_data = introduce_variation(sampled_data, week)
        simulated_data["week"] = week  # Add week column
        all_data = pd.concat([all_data, simulated_data])

    return all_data


class TestInsertData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

        # Load credentials from environment variables
        cls.username = os.getenv('USERNAME')
        cls.password = os.getenv('PASSWORD')

        # Authenticate and get the session
        login_response = cls.client.post('/', data={
            'username': cls.username,
            'password': cls.password
        }, follow_redirects=True)

        print(f"Login Response: {login_response.data}")
        assert login_response.status_code == 200

        # Generate fake data
        cls.all_data = generate_fake_data()

    def test_insert_data(self):
        self.insert_data(self.all_data)

    def insert_data(self, df):
        for week in range(1, 53):
            weekly_data = df[df['week'] == week]
            for _, row in weekly_data.iterrows():
                input_data = {
                    'step': row['step'],
                    'type': row['type'],
                    'amount': row['amount'],
                    'nameOrig': row['nameOrig'],
                    'oldbalanceOrg': row['oldbalanceOrg'],
                    'newbalanceOrig': row['newbalanceOrig'],
                    'nameDest': row['nameDest'],
                    'oldbalanceDest': row['oldbalanceDest'],
                    'newbalanceDest': row['newbalanceDest']
                }

                response = self.client.post('/dashboard', json=input_data)
                self.assertEqual(response.status_code, 200)

                # Call the prediction function
                response = self.client.post('/predict', json=input_data)
                self.assertEqual(response.status_code, 200)

            print(f'Inserted data for week {week}')
            time.sleep(2)  # Simulate time passing (adjust as needed)


if __name__ == '__main__':
    unittest.main()
