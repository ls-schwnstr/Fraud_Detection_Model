import unittest
from unittest.mock import patch
import time
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from app.db import RetrainingLog

# Database setup
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fraud_detection.db'))
DATABASE_URL = f'sqlite:///{db_path}'
engine = create_engine(DATABASE_URL, echo=True)

Session = sessionmaker(bind=engine)


def get_latest_retraining_log(end_of_day):
    try:
        # Convert end_of_day to datetime object
        end_date = datetime.strptime(end_of_day, '%Y-%m-%d %H:%M:%S')
        print("Checking logs at:", end_date)

        with Session() as session:
            # Query for retraining logs up to the end of the specified day
            result = session.query(RetrainingLog).order_by(RetrainingLog.retraining_timestamp.desc()).first()

            if result:
                retraining_timestamp = result.retraining_timestamp
                return retraining_timestamp
            else:
                print("No retraining logs found.")
                return None
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")
        return None


class TestMonthlyRetrainingTrigger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup environment if needed
        pass

    @patch('datetime.datetime')
    def test_monthly_retraining_trigger(self, mock_datetime):
        # Set a fixed start date
        start_date = datetime(2024, 1, 1)
        mock_datetime.now.return_value = start_date

        # Simulate time passing by manually
        for month in range(1, 13):  # Simulate for a whole year
            # Calculate end of the first day of the next month
            current_date = start_date + timedelta(days=(month * 30))  # Simulate month passing
            first_of_month = current_date.replace(day=1)
            end_of_day = (current_date.replace(day=1) + timedelta(days=1) - timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            mock_datetime.now.return_value = first_of_month
            print(f"Simulating time for {mock_datetime.now().strftime('%Y-%m-%d')}")

            # Add delay to allow workflow to trigger retraining
            time.sleep(3)  # 5 minutes delay, adjust as needed

            # Check for the latest retraining log up to the end of the first day of the month
            latest_log = get_latest_retraining_log(end_of_day)
            if latest_log:
                print(f"Latest retraining log up to end of the day: {latest_log.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"No retraining logs found up to the end of the day: {end_of_day}")


if __name__ == '__main__':
    unittest.main()
