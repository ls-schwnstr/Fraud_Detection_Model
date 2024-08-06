import unittest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.model import preprocess_input_data, make_predictions_and_check_drift
from db import add_processed_data, get_latest_raw_data, add_raw_data, get_latest_processed_data
import mlflow
from unittest.mock import patch
import time

# Mocking MLflow functions for testing
mlflow.set_tracking_uri("http://mlflow:5004")

DATABASE_URL = 'sqlite:///fraud_detection.db'
engine = create_engine(DATABASE_URL, echo=True)


class TestInsertData(unittest.TestCase):
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_artifact')
    @patch('models.model.make_predictions_and_check_drift')
    @patch('db.get_latest_processed_data')
    def test_insert_data(self,  mock_log_params, mock_log_metric, mock_start_run):
        file_path = 'simulated_data/simulated_data_year.csv'
        all_data = pd.read_csv(file_path, delimiter=';')
        insert_data(all_data)
        mock_start_run.assert_called()
        mock_log_metric.assert_called()
        mock_log_params.assert_called()


# Function to insert data into the database
def insert_data(df):
    session = sessionmaker(bind=engine)()

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

            print(f"Received input data: {input_data}")
            add_raw_data(session, input_data)

            raw_data_entry = get_latest_raw_data(session)
            raw_data_dict = {column.name: getattr(raw_data_entry, column.name)
                             for column in raw_data_entry.__table__.columns}
            print(f"Retrieved raw data: {raw_data_dict}")

            raw_data_df = pd.DataFrame([raw_data_dict])
            print(f"Retrieved raw data DataFrame: {raw_data_df}")

            print(raw_data_df.head())

            processed_data_df = preprocess_input_data(raw_data_df)
            processed_data_dict = processed_data_df.to_dict(orient='records')[0]
            add_processed_data(session, processed_data_dict)

            # Call the prediction function
            processed_data_df = get_latest_processed_data(session)
            prediction, drift_detected = make_predictions_and_check_drift(processed_data_df)
            print(f'Prediction: {prediction}, Data Drift Detected: {drift_detected}')

        print(f'Inserted data for week {week}')
        time.sleep(2)  # Simulate time passing (adjust as needed)


if __name__ == '__main__':
    # Run the unittests
    unittest.main(exit=False)

    # Insert data week by week
    file_path = 'simulated_data/simulated_data_year.csv'
    all_data = pd.read_csv(file_path, delimiter=';')

    for week in range(1, 53):
        weekly_data = all_data[all_data['week'] == week]
        insert_data(weekly_data)
        print(f'Inserted data for week {week}')
        time.sleep(2)  # Simulate passing of time
