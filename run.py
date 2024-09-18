from app import app
import mlflow

mlflow.set_tracking_uri("http://localhost:5004")
mlflow.set_experiment('experiment')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5003)

