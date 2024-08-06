import unittest
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from app import app
import time
import os

# Mocking MLflow functions for testing
mlflow.set_tracking_uri("http://localhost:5004")

DATABASE_URL = 'sqlite:///fraud_detection.db'
engine = create_engine(DATABASE_URL, echo=True)


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

    def test_insert_data(self):
        file_path = 'app/simulated_data/simulated_data_year.csv'
        all_data = pd.read_csv(file_path, delimiter=';')
        self.insert_data(all_data)

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
