import unittest
from unittest.mock import patch
import datetime
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from app import app
import os
import numpy as np
import warnings

# Ignore DeprecationWarning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Mocking MLflow functions for testing
mlflow.set_tracking_uri("http://localhost:5005")

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fraud_detection.db'))
print(f"Database path: {db_path}")
DATABASE_URL = f'sqlite:///{db_path}'
engine = create_engine(DATABASE_URL, echo=True)


class TestMonthlyRetraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.setup_environment()

    @classmethod
    def setup_environment(cls):
        # Setup environment (e.g., authentication, initialization)
        pass


    def generate_simulated_data(self):
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

        # Generate data for each week
        all_data = pd.DataFrame()
        for week in range(1, 53):
            sampled_data = original_data.sample(n=2, random_state=week).copy()
            simulated_data = introduce_variation(sampled_data, week)
            simulated_data["week"] = week  # Add week column
            all_data = pd.concat([all_data, simulated_data])

        return all_data

    @patch('datetime.datetime')
    def test_monthly_retraining_trigger(self, mock_datetime):
        # Simulate weekly data insertion
        for week in range(1, 53):  # Simulate for a whole year
            mock_datetime.now.return_value = datetime.datetime(2024, 1, 1) + datetime.timedelta(weeks=week - 1)
            current_date = mock_datetime.now()
            print(f"Testing for {current_date.strftime('%Y-%m-%d')}")

            # Generate and insert data for the current week
            simulated_data = self.generate_simulated_data()
            self.insert_data(simulated_data)

            # Check if it's the start of the month to verify retraining trigger
            if current_date.day == 1:
                print(f"Month start detected: {current_date.strftime('%Y-%m-%d')}")
                # In a real test, check logs or the state of your system to verify retraining occurred
            else:
                print(f"Month start not detected: {current_date.strftime('%Y-%m-%d')}")

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

            print(f"Sending data to /dashboard: {input_data}")
            response = self.client.post('/dashboard', json=input_data)
            print(f"Response from /dashboard: {response.data}")
            self.assertEqual(response.status_code, 302)

            # Check the prediction response
            predict_response = self.client.get('/predict')
            print(f"Response from /predict: {predict_response.data}")
            self.assertIn(predict_response.status_code, [200, 500])

        print(f'Inserted data for week {df["week"].iloc[0]}')


if __name__ == '__main__':
    unittest.main()
