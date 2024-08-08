import unittest
from unittest.mock import patch
import datetime
import time
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from app.db import RetrainingLog

# Database setup
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fraud_detection.db'))
DATABASE_URL = f'sqlite:///{db_path}'
engine = create_engine(DATABASE_URL, echo=True)

Session = sessionmaker(bind=engine)
session = Session()


def check_logs_for_retraining(date):
    # Convert date to datetime object
    target_date = datetime.strptime(date, '%Y-%m-%d')

    # Query the Retraining_log table
    query = select([RetrainingLog]).where(
        RetrainingLog.retraining_timestamp >= target_date,
        RetrainingLog.retraining_timestamp < target_date + timedelta(days=1)
    )

    result = session.execute(query).fetchall()

    # Close the session
    session.close()

    # Check if any rows were returned
    return len(result) > 0


class TestMonthlyRetrainingTrigger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup environment if needed
        pass

    @patch('datetime.datetime')
    def test_monthly_retraining_trigger(self, mock_datetime):
        # Set a fixed start date
        start_date = datetime.datetime(2024, 1, 1)
        mock_datetime.now.return_value = start_date

        # Simulate time passing by manually
        for month in range(1, 13):  # Simulate for a whole year
            # Set mock datetime to the start of the next month
            mock_datetime.now.return_value = start_date + datetime.timedelta(days=(month * 30))
            print(f"Simulating time for {mock_datetime.now().strftime('%Y-%m-%d')}")

            time.sleep(1)

            if check_logs_for_retraining(mock_datetime.now().strftime('%Y-%m-%d')):
                print(f"Retraining triggered for {mock_datetime.now().strftime('%Y-%m-%d')}")
            else:
                print(f"Retraining not triggered for {mock_datetime.now().strftime('%Y-%m-%d')}")


if __name__ == '__main__':
    unittest.main()
