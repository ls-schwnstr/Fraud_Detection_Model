import joblib
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import get_predicted_data, RetrainingLog


def get_session():
    # Setup your database connection here
    engine = create_engine('sqlite:///fraud_detection.db')
    Session = sessionmaker(bind=engine)
    return Session()


def retrain_model():
    print("Retraining the model...")
    # Connect to the database
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

        print(f"Model retrained and saved to {model_path}")

        return run_id


if __name__ == "__main__":
    retrain_model()
