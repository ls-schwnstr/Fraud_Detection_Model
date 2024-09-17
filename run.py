from app import app
import mlflow
import os

# Set the tracking URI using your storage account and container
mlflow.set_tracking_uri("azure_blob://fraud.blob.core.windows.net/fraud_1726588205992?RI0ldUrrmVb8RLqezjS2ZMjQChQC6LMaeQszfXmt/e/rmQxdBtasnC6ChK5QcdPUkY4MFfSLEv3T+AStR3iZZA==")

#mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5004')  # Updated port
#mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment('experiment')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)

