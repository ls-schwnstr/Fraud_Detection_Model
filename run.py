from app import app
import mlflow
import os

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://mlflow:5004')  # Updated port

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)

