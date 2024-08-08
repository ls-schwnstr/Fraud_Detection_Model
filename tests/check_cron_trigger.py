import argparse
import os
import mlflow
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import RetrainingLog

# Database setup
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fraud_detection.db'))
DATABASE_URL = f'sqlite:///{db_path}'
engine = create_engine(DATABASE_URL, echo=True)
mlflow.set_tracking_uri("http://localhost:5005")

session = sessionmaker(bind=engine)
session = session()


def check_cron_trigger(start_date, end_date):
    # Replace with code to check if cron job was triggered
    print(f"Checking if cron job was triggered between {start_date} and {end_date}")
    retraining_logs = session.query(RetrainingLog).order_by(RetrainingLog.retraining_timestamp.desc()).all()
    print(retraining_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    args = parser.parse_args()
    check_cron_trigger(args.start_date, args.end_date)
