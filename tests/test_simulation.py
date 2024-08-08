import unittest
from unittest.mock import patch
import datetime
import time
from sqlalchemy import MetaData, Table
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select

# Database setup
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fraud_detection.db'))
DATABASE_URL = f'sqlite:///{db_path}'
engine = create_engine(DATABASE_URL, echo=True)


# Session setup
def get_session():
    Session = sessionmaker(bind=engine)
    return Session()


def check_logs_for_retraining(engine, current_time):
    # Load the table metadata
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # Assuming 'ReTrainingLog' is the name of your table
    retraining_log_table = Table('ReTrainingLog', metadata, autoload_with=engine)

    # Query to get the most recent retraining timestamp
    query = select([retraining_log_table.c.retraining_timestamp]).order_by(
        retraining_log_table.c.retraining_timestamp.desc()).limit(1)

    with engine.connect() as connection:
        result = connection.execute(query).fetchone()

    # Check if the result matches the current time
    if result:
        retraining_time = result[0]
        print(f"Most recent retraining timestamp: {retraining_time}")
        return retraining_time == current_time
    else:
        print("No retraining logs found.")
        return False


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
