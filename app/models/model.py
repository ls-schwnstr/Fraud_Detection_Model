import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

from app.models.data_quality import check_for_data_drift
from app.routes import load_feature_names
from db import get_session, get_predicted_data, RetrainingLog, add_predicted_data


def preprocess_data_for_training(data):
    data = data[
        ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type', 'isFraud']]
    data = pd.get_dummies(data, columns=['type'], drop_first=True)

    # Convert integer columns to float64 if they may have missing values
    integer_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    for col in integer_columns:
        if data[col].isnull().any():  # Check if there are missing values
            data[col] = data[col].astype('float64')  # Convert to float64

    return data


def train_model():
    # Connect to the SQLite database
    db_session = get_session()

    # Fetch the data from the predictions table
    data = get_predicted_data(db_session)

    # Drop non-numeric columns
    non_numeric_cols = ['timestamp', 'id']
    data = data.drop(columns=[col for col in non_numeric_cols if col in data.columns], errors='ignore')

    print(data.head())
    # Separate features and target
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']

    # Save feature names
    feature_names = X.columns.tolist()

    # Save feature names to a file
    with open('feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(X, y)

    with mlflow.start_run() as run:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives:", tp)

        # Log parameters, metrics, and the model
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_metric('True Negatives', tn)
        mlflow.log_metric('False Positives', fp)
        mlflow.log_metric('False Negatives', fn)
        mlflow.log_metric('True Positives', tp)
        mlflow.log_metric('precision', classification_report(y_test, y_pred, output_dict=True)['1']['precision'])
        mlflow.log_metric('recall', classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
        mlflow.log_metric('f1-score', classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'])

        # Log the model without sanitizing signature
        mlflow.sklearn.log_model(model, 'model')

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Save the model to a file
        model_path = 'model.pkl'
        joblib.dump(model, model_path)

        print(f"Model saved to {model_path}")

        # Log the training timestamp
        db_session.add(RetrainingLog())
        db_session.commit()

        return run_id


def preprocess_input_data(data):
    type_dummies = pd.get_dummies(data['type'], prefix='', prefix_sep='')
    data = data.join(type_dummies)
    data = data.drop(columns=['type'])  # Drop original 'type'

    data = data[
        ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'] +
        list(type_dummies.columns)  # Ensure dummy columns are included
        ]

    # Convert integer columns to float64 if they may have missing values
    integer_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in integer_columns:
        if data[col].isnull().any():  # Check if there are missing values
            data[col] = data[col].astype('float64')  # Convert to float64

    return data


def make_predictions_and_check_drift(processed_data_df):
    db_session = get_session()
    feature_names = load_feature_names()

    try:
        print("Making predictions and checking for data drift...")

        # Reorder columns if necessary
        processed_data_df = processed_data_df[feature_names]

        print(f"Features after processing: {processed_data_df.head()}")  # Debug

        # Load the trained model
        with open('model.pkl', 'rb') as f:
            model = joblib.load(f)

        # Make predictions
        predictions = model.predict(processed_data_df)
        print(f"Predictions: {predictions}")  # Debug
        predictions = [int(pred) for pred in predictions]  # Ensure JSON serializable

        # Log prediction to MLflow
        with mlflow.start_run() as run:
            # Log parameters as needed
            mlflow.log_params({
                "step": processed_data_df['step'].iloc[0],
                "amount": processed_data_df['amount'].iloc[0],
                "oldbalanceOrg": processed_data_df['oldbalanceOrg'].iloc[0],
                "newbalanceOrig": processed_data_df['newbalanceOrig'].iloc[0],
                "oldbalanceDest": processed_data_df['oldbalanceDest'].iloc[0],
                "newbalanceDest": processed_data_df['newbalanceDest'].iloc[0]
            })
            mlflow.log_metric("isFraud", predictions[0])

        # Prepare the dictionary for saving
        processed_data_dict = processed_data_df.iloc[0].to_dict()
        processed_data_dict['isFraud'] = predictions[0]

        # Use the refactored function to save the predicted data
        add_predicted_data(db_session, processed_data_dict)

        db_session.commit()

        # Check for data drift
        drift_detected = check_for_data_drift(db_session)

        return predictions[0], drift_detected

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e






