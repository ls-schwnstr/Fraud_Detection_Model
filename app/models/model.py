from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
from app.db import get_session, get_predicted_data, RetrainingLog, add_predicted_data, add_retraining_log
import os


def get_feature_names_path():
    # Relative path to feature_names.txt from the directory of this file
    return os.path.join(os.path.dirname(__file__), 'feature_names.txt')


def get_model_path():
    # Relative path to model.pkl from the directory of this file
    return os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_feature_names():
    feature_names_path = get_feature_names_path()
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    return feature_names


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


def train_model(timestamp=None, retraining_type=None):
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

    # Path to feature names file
    feature_names_path = get_feature_names_path()

    # Save feature names to a file
    with open(feature_names_path, 'w') as f:
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

        # Print MLflow tracking URI
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        # Log parameters, metrics, and the model
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_metric('True Negatives', tn)
        mlflow.log_metric('False Positives', fp)
        mlflow.log_metric('False Negatives', fn)
        mlflow.log_metric('True Positives', tp)
        mlflow.log_metric('precision', classification_report(y_test, y_pred, output_dict=True)['1']['precision'])
        mlflow.log_metric('recall', classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
        mlflow.log_metric('f1-score', classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'])

        print("parameters logged")
        # Log the model without sanitizing signature
        mlflow.sklearn.log_model(model, 'model')

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Save the model to a file
        model_path = get_model_path()
        joblib.dump(model, model_path)

        print(f"Model saved to {model_path}")

        # Log the training timestamp
        timestamp = datetime.utcnow() if timestamp is None else timestamp
        add_retraining_log(db_session, timestamp, retraining_type, run_id)
        print(f"Retraining log added to the database. {RetrainingLog}")
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


def make_predictions(processed_data_df, timestamp):
    db_session = get_session()
    feature_names = load_feature_names()

    try:
        print("Making predictions...")

        # Reorder columns if necessary
        processed_data_df = processed_data_df[feature_names]

        print(f"Features after processing: {processed_data_df.head()}")  # Debug

        # Load the trained model
        model_path = get_model_path()
        with open(model_path, 'rb') as f:
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
        add_predicted_data(db_session, processed_data_dict, timestamp)

        db_session.commit()

        return predictions[0]

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e






