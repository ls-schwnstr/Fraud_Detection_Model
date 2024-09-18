import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app import app as flask_app
from app.db import RetrainingLog, get_session
import mlflow
import os
from app.models.model import train_model
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

azure_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
mlflow.set_tracking_uri("http://localhost:5004")
mlflow.set_registry_uri(f"wasbs://containerfraud@fraud.blob.core.windows.net")

experiment_name = "unittest_experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # Create experiment if it doesn't exist
    mlflow.create_experiment(experiment_name, artifact_location="wasbs://containerfraud@fraud.blob.core.windows.net")
else:
    # Set the experiment if it exists
    mlflow.set_experiment(experiment_name)

db_path = ('mssql+pyodbc://<your_username>:<your_password>@<server_name>.database.windows.net:1433/<db_name>?driver'
           '=ODBC+Driver+17+for+SQL+Server')
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

    def check_data_drift(self, timestamp=None):
        payload = {'timestamp': timestamp.isoformat()} if timestamp else {}
        response = self.client.get('/check_data_drift', query_string=payload)
        self.assertEqual(response.status_code, 200)

        # If data drift detected, log the retraining date
        session = self.session
        try:
            # Start a new MLflow run for logging the data drift retraining
            with mlflow.start_run() as run:
                run_id = run.info.run_id  # Get the run_id of the current MLflow run

                print(f"Logging data drift for {timestamp} with run_id {run_id}")
                retraining_log = RetrainingLog(
                    retraining_timestamp=timestamp,
                    retraining_type='data_drift',
                    run_id=run_id  # Use the valid run_id here
                )
                session.add(retraining_log)
                session.commit()

        except Exception as e:
            print(f"Failed to log data drift: {e}")
            session.rollback()
        finally:
            session.close()

    def check_retraining_logs(self, expected_dates, retraining_type):
        logs = self.session.query(RetrainingLog).filter_by(retraining_type=retraining_type).all()
        logged_dates = {log.retraining_timestamp.date() for log in logs}
        missing_dates = expected_dates - logged_dates
        self.assertTrue(not missing_dates, f"Missing retraining dates for {retraining_type}: {missing_dates}")

    def test_retraining(self):
        df = generate_simulated_data()
        start_date = datetime(2021, 1, 1)  # Start on January 1st, 2021
        end_date = datetime(2021, 12, 31)  # End on December 31st, 2021

        retraining_dates = set()  # To keep track of dates when retraining is triggered
        drift_dates = set()
        last_week = None
        last_week_year = None
        week_counter = 0

        # Authenticate the user and pass the timestamp
        self.authenticate(timestamp=start_date)

        # Wait for training to complete before inserting data
        if not os.path.isfile(model_path):
            self.wait_for_training_to_complete()
        else:
            print("Model already exists. Skipping training.")

        current_date = start_date

        while current_date <= end_date:
            # Get ISO week number and year
            week_num, week_year = current_date.isocalendar()[1], current_date.isocalendar()[0]

            # Handle 53-week
            if week_num == 53:
                week_year += 1
                week_num = 1

            # Insert data if it's the first day of the week and it's a new week
            if current_date.weekday() == 0:  # Monday is the first day of the week
                week_counter += 1
                if week_counter % 4 == 0:  # Check if it's the fourth week
                    if last_week != week_num or last_week_year != week_year:  # Check if we haven't already inserted data for this week
                        print(f"Inserting data for Week {week_num} of Year {week_year}")
                        self.insert_data(df, week_num, timestamp=current_date)
                        drift_dates.add(current_date.date())  # Log the date for drift checks
                        last_week = week_num
                        last_week_year = week_year

                        # Trigger data drift check immediately after data insertion
                        print(f"Triggering data drift check for {current_date.strftime('%Y-%m-%d')}")
                        self.check_data_drift(timestamp=current_date)
                        print("Data drift dates are being added")


            # Check if it's the first of the month to trigger retraining
            if current_date.day == 1:
                print(f"Triggering retraining for {current_date.strftime('%Y-%m-%d')}")
                train_model(timestamp=current_date, retraining_type='monthly')  # Pass the simulated timestamp
                retraining_dates.add(current_date.date())

            # Move to the next day
            current_date += timedelta(days=1)

        # Ensure the last week is included if it spans into the next year
        last_day_of_year = datetime(2021, 12, 31)
        last_week_num = last_day_of_year.isocalendar()[1]
        if last_week_num == 53:
            last_week_year += 1
            last_week_num = 1
        if last_week_num != last_week:
            self.insert_data(df, last_week_num, timestamp=last_day_of_year)
            drift_dates.add(current_date.date())  # Log the date for drift checks

        # Check if retraining logs are correctly recorded
        all_dates = {datetime(2021, m, 1).date() for m in range(1, 13)}
        self.check_retraining_logs(all_dates, 'monthly')

        # Check if data drift logs are correctly recorded
        self.check_retraining_logs(drift_dates, 'data_drift')


if __name__ == "__main__":
    unittest.main()