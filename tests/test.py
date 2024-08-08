import unittest
from unittest.mock import patch
import time
from datetime import datetime, timedelta
import os
from app.db import RetrainingLog, get_session  # Ensure these imports are correct
import numpy as np
import pandas as pd


def introduce_variation(data, week):
    np.random.seed(week)
    # Introduce variation in the 'amount' column
    data['amount'] *= (1 + np.random.normal(0, 0.1, size=data.shape[0]))
    if week % 6 == 0:
        data['oldbalanceOrg'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
        data['newbalanceOrig'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
    return data


def generate_simulated_data(week):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'Fraud.csv')
    original_data = pd.read_csv(file_path, delimiter=";", nrows=1000)
    original_data['amount'] = pd.to_numeric(original_data['amount'], errors='coerce')
    original_data['amount'].fillna(original_data['amount'].median(), inplace=True)
    original_data.drop(['isFraud', 'isFlaggedFraud'], axis=1, inplace=True)
    return introduce_variation(original_data.copy(), week)


class TestRetraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = get_session()
        cls.client = ...  # Initialize your test client here

    def insert_data(self, df):
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
            predict_response = self.client.get('/predict')
            self.assertIn(predict_response.status_code, [200, 500])

    def check_retraining_triggered(self, expected_retraining_timestamp):
        retraining_logs = self.session.query(RetrainingLog).order_by(RetrainingLog.retraining_timestamp.desc()).all()
        if retraining_logs:
            latest_retraining = retraining_logs[0].retraining_timestamp
            self.assertGreaterEqual(latest_retraining, expected_retraining_timestamp)
        else:
            self.fail("No retraining logs found.")

    @patch('datetime.datetime')
    def test_retraining_triggers(self, mock_datetime):
        start_date = datetime(2024, 1, 1)
        mock_datetime.now.return_value = start_date

        for month in range(1, 13):  # Simulate for a whole year
            for week in range(1, 5):  # Assume 4 weeks per month
                current_date = start_date + timedelta(weeks=week + (month - 1) * 4)
                simulated_data = generate_simulated_data(week)
                self.insert_data(simulated_data)

                mock_datetime.now.return_value = current_date
                if current_date.day == 1:  # Check for the first day of the month
                    print(f"Simulating start of month: {current_date.strftime('%Y-%m-%d')}")
                    self.check_retraining_triggered(current_date)

                time.sleep(10)  # Short delay for processing

        # Final check after one year of data
        self.check_retraining_triggered(start_date + timedelta(days=365))


def check_if_new_month():
    # Load the last retraining timestamp from a persistent location or database
    last_retraining_date = ...  # Retrieve this from your logging mechanism

    current_date = datetime.now()

    if last_retraining_date is None:
        return True

    # Check if a new month has begun
    return current_date.year > last_retraining_date.year or \
        current_date.month > last_retraining_date.month


def perform_retraining():
    # Your retraining logic here
    print("Monthly retraining performed")


def main():
    if check_if_new_month():
        perform_retraining()
    else:
        print("No monthly retraining needed")


if __name__ == "__main__":
    main()
