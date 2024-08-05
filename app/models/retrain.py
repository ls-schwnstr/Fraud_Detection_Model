import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import get_predicted_data


def get_session():
    # Setup your database connection here
    engine = create_engine('sqlite:///fraud_detection.db')
    Session = sessionmaker(bind=engine)
    return Session()


def retrain_model():
    print("Retraining the model...")
    # Connect to the database
    db_session = get_session()

    # Fetch the data from the processed_data table
    data = get_predicted_data(db_session)

    # Drop non-numeric columns
    non_numeric_cols = ['timestamp', 'id']  # Adjust based on your schema
    data = data.drop(columns=[col for col in non_numeric_cols if col in data.columns], errors='ignore')

    # Ensure the correct feature order
    feature_order = [
        'oldbalanceOrg',
        'oldbalanceDest',
        'type_CASH_OUT',
        'type_PAYMENT',
        'type_TRANSFER',
        'amount',
        'newbalanceOrig',
        'step',
        'newbalanceDest',
        'type_DEBIT'
    ]
    data = data[feature_order]

    # Separate features and target
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model to a file
    model_path = 'model.pkl'
    joblib.dump(model, model_path)

    # Log the model to MLflow
    with mlflow.start_run() as run:
        example_input = pd.DataFrame({
            "step": [1],
            "amount": [100.0],
            "oldbalanceOrg": [200.0],
            "newbalanceOrig": [300.0],
            "oldbalanceDest": [400.0],
            "newbalanceDest": [500.0],
            "type_CASH_OUT": [0],
            "type_DEBIT": [0],
            "type_PAYMENT": [1],
            "type_TRANSFER": [0],
        })
        example_output = model.predict(example_input)
        signature = mlflow.models.signature.infer_signature(example_input, example_output)
        mlflow.sklearn.log_model(model, 'model', signature=signature)

        print(f"Model retrained and saved to {model_path}")


if __name__ == "__main__":
    retrain_model()
