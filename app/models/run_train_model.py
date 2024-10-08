import os
import mlflow
from model import train_model

if __name__ == "__main__":
    # Specify the retraining type
    retraining_type = 'monthly'

    mlflow.set_tracking_uri("http://localhost:5004")
    mlflow.set_experiment('monthly_retraining')

    # Call the train_model function with retraining_type
    train_model(retraining_type=retraining_type)
