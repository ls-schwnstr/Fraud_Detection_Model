from app import app
import mlflow
import os

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5004')  # Updated port
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment('experiment')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)

