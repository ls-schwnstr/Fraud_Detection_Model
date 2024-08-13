import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app import app as flask_app
from app.db import RetrainingLog, get_session, get_db_connection_url
import mlflow
import os
from app.models.model import train_model


mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5004')  # Updated port
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment('test')

db_path = get_db_connection_url()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'models', 'model.pkl'))


def introduce_variation(data, week):
    np.random.seed(week)
    data['amount'] *= (1 + np.random.normal(0, 0.1, size=data.shape[0]))
    if week % 6 == 0:
        data['oldbalanceOrg'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
        data['newbalanceOrig'] *= (1 + np.random.normal(0.2, 0.1, size=data.shape[0]))
    return data


def generate_simulated_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'Fraud.csv')
    original_data = pd.read_csv(file_path, delimiter=";", nrows=1000)
    original_data['amount'] = pd.to_numeric(original_data['amount'], errors='coerce')
    original_data['amount'].fillna(original_data['amount'].median(), inplace=True)
    original_data.drop(['isFraud', 'isFlaggedFraud'], axis=1, inplace=True)

    # Add a 'week' column with values from 1 to 53
    weeks = np.arange(1, 54)
    n_rows = len(original_data)
    original_data['week'] = np.tile(weeks, n_rows // len(weeks) + 1)[:n_rows]

    return original_data


class TestRetraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = flask_app
        cls.client = cls.app.test_client()
        cls.session = get_session()

        cls.username = os.getenv('USERNAME')
        cls.password = os.getenv('PASSWORD')

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @classmethod
    def authenticate(cls, timestamp=None):
        data = {
            'username': cls.username,
            'password': cls.password
        }
        # Include timestamp in the query string if provided
        query_string = {'timestamp': timestamp.isoformat()} if timestamp else {}

        response = cls.client.post('/', data=data, query_string=query_string, follow_redirects=True)

        print(f"Login response status code: {response.status_code}")
        print(f"Login response data: {response.data.decode('utf-8')}")

    @classmethod
    def dashboard(cls, input_data, timestamp=None):
        payload = {'timestamp': timestamp.isoformat()} if timestamp else {}
        response = cls.client.post('/dashboard', json=input_data, query_string=payload, follow_redirects=True)
        return response


    def train_model(self, timestamp=None):
        payload = {}
        if timestamp:
            payload['timestamp'] = timestamp.isoformat()
        print("timestamp", timestamp)
        print("payload", payload)
        response = self.client.post('/train', query_string=payload)
        self.assertEqual(response.status_code, 200)

    def wait_for_training_to_complete(self):
        import time
        print("Waiting for model training to complete...")
        while True:
            response = self.client.get('/check-training-status')
            status = response.json.get('status', '')
            print(f"Training status: {status}")  # Detailed status logging
            if status == 'complete':
                print("Model training completed.")
                break
            elif status == 'failed':
                print("Model training failed.")
                raise Exception("Model training failed.")
            time.sleep(5)  # Wait before checking again

    def insert_data(self, df, week, timestamp):
        session = self.session
        try:
            filtered_df = df[df['week'] == week]

            if filtered_df.empty:
                print(f"No data found for week {week}")
                return

            # Get the first row of the filtered DataFrame
            row = filtered_df.iloc[0]
            print(f"Inserting data for Week {week}")
            input_data = {
                'step': 1,
                'amount': row['amount'],
                'oldbalanceOrg': row['oldbalanceOrg'],
                'newbalanceOrig': row['newbalanceOrig'],
                'oldbalanceDest': row['oldbalanceDest'],
                'newbalanceDest': row['newbalanceDest'],
                'type': row['type']
            }

            self.dashboard(input_data, timestamp=timestamp)

        finally:
            session.close()

    def check_retraining_logs(self, expected_dates, retraining_type):
        logs = self.session.query(RetrainingLog).filter_by(retraining_type=retraining_type).all()
        logged_dates = {log.retraining_timestamp.date() for log in logs}
        missing_dates = expected_dates - logged_dates
        self.assertTrue(not missing_dates, f"Missing retraining dates for {retraining_type}: {missing_dates}")

    def test_retraining(self):
        df = generate_simulated_data()
        start_date = datetime(2026, 1, 1)  # Start on January 1st, 2026
        end_date = datetime(2026, 12, 31)  # End on December 31st, 2026

        retraining_dates = set()  # To keep track of dates when retraining is triggered
        drift_dates = set()
        last_week = None
        last_week_year = None

        # Authenticate the user and pass the timestamp
        self.authenticate(timestamp=start_date)

        # Wait for training to complete before inserting data
        if not os.path.isfile(model_path):
            print("model path", model_path)
            self.wait_for_training_to_complete()
        else:
            print("Model already exists. Skipping training.")

        current_date = start_date

        while current_date <= end_date:
            # Get ISO week number and year
            week_num, week_year = current_date.isocalendar()[1], current_date.isocalendar()[0]

            # Handle 53-week years
            if week_num == 53:
                week_year += 1
                week_num = 1

            # Insert data if it's the first day of the week and it's a new week
            if current_date.weekday() == 0:  # Monday is the first day of the week
                if last_week != week_num or last_week_year != week_year:  # Check if we haven't already inserted data for this week
                    print(f"Inserting data for Week {week_num} of Year {week_year}")
                    self.insert_data(df, week_num, timestamp=current_date)
                    drift_dates.add(current_date.date())  # Log the date for drift checks
                    last_week = week_num
                    last_week_year = week_year

            # Check if it's the first of the month to trigger retraining
            if current_date.day == 1:
                print(f"Triggering retraining for {current_date.strftime('%Y-%m-%d')}")
                train_model(timestamp=current_date, retraining_type='monthly')  # Pass the simulated timestamp
                retraining_dates.add(current_date.date())

            # Move to the next day
            current_date += timedelta(days=1)

        # Ensure the last week is included if it spans into the next year
        # Handle 53-week years
            if week_num == 53 and week_year == 2026:
                print(f"Inserting data for the 53rd week of {week_year}")
                self.insert_data(df, week_num, timestamp=current_date)
                drift_dates.add(current_date.date())  # Log the date for drift checks

        # Check if retraining logs are correctly recorded
        all_dates = {datetime(2026, m, 1).date() for m in range(1, 13)}
        self.check_retraining_logs(all_dates, 'monthly')

        # Check if data drift logs are correctly recorded
        self.check_retraining_logs(drift_dates, 'data_drift')


if __name__ == "__main__":
    unittest.main()
