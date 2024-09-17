import os
import mlflow
from model import train_model

if __name__ == "__main__":
    # Specify the retraining type
    retraining_type = 'monthly'

    #mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5004')  # Updated port
    #mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(
        "azure_blob://fraud.blob.core.windows.net/fraud_1726588205992?RI0ldUrrmVb8RLqezjS2ZMjQChQC6LMaeQszfXmt/e/rmQxdBtasnC6ChK5QcdPUkY4MFfSLEv3T+AStR3iZZA==")
    mlflow.set_experiment('monthly_retraining')

    # Call the train_model function with retraining_type
    train_model(retraining_type=retraining_type)
