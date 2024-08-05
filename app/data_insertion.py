import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app
from app.models.model import preprocess_input_data, make_predictions_and_check_drift
from app.db import add_processed_data, get_latest_raw_data, add_raw_data

# Database configuration
DATABASE_URL = 'sqlite:///fraud_detection.db'
engine = create_engine(DATABASE_URL, echo=True)


# Function to insert data into the database
def insert_data(file_path):
    all_data = pd.read_csv(file_path, delimiter=';')
    session = sessionmaker(bind=engine)()

    for week in range(1, 53):
        weekly_data = all_data[all_data['week'] == week]
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

            processed_data_df = preprocess_input_data(raw_data_df)
            processed_data_dict = processed_data_df.to_dict(orient='records')[0]
            add_processed_data(session, processed_data_dict)

            # Call the prediction function directly
            with app.app_context():
                prediction, drift_detected = make_predictions_and_check_drift(processed_data_df)
                print(f'Prediction: {prediction}, Data Drift Detected: {drift_detected}')


        print(f'Inserted data for week {week}')
        time.sleep(2)  # Simulate time passing (adjust as needed)


if __name__ == '__main__':
    file_path = 'simulated_data/simulated_data_year.csv'
    all_data = pd.read_csv(file_path, delimiter=';')

    # Insert data week by week
    for week in range(1, 53):
        weekly_data = all_data[all_data['week'] == week]
        insert_data(weekly_data)
        print(f'Inserted data for week {week}')
        # Simulate passing of time
        # Sleep for 2 seconds instead of weeks to simulate
        import time

        time.sleep(2)
