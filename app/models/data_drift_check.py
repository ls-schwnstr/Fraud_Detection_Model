import sys
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import ks_2samp, chi2_contingency, skew, kurtosis, entropy
import numpy as np
import mlflow
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import get_reference_data, get_new_data, get_session
from app.models.model import train_model


# Database & MLFlow setup
mlflow.set_tracking_uri("azure_blob://fraud.blob.core.windows.net/containerfraud?sp=racw&st=2024-09-17T16:39:30Z&se=2024-09-18T00:39:30Z&sv=2022-11-02&sr=c&sig=r6clHemk6yyU%2BiNM2f6nJs6T%2FzxFMUx4uZBkKIaHWq8%3D")
db_path = ('mssql+pyodbc://adminuser:FraudDetection1!@fraud-detection-server.database.windows.net:1433'
           '/fraud_detection_db?driver=ODBC+Driver+17+for+SQL+Server')
session = get_session()



def calculate_descriptive_statistics(data):
    statistics = {}
    for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        statistics[column] = {
            'mean': data[column].mean(),
            'median': data[column].median(),
            'std_dev': data[column].std(),
            'skewness': skew(data[column]),
            'kurtosis': kurtosis(data[column])
        }
    return statistics


def ks_test(incoming_data, reference_data):
    ks_results = {}
    for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        ks_stat, p_value = ks_2samp(reference_data[column], incoming_data[column])
        ks_results[column] = {'ks_stat': ks_stat, 'p_value': p_value}
    return ks_results


def chi_square_test(incoming_data, reference_data, column):
    if column not in reference_data.columns or column not in incoming_data.columns:
        raise KeyError(f"Column {column} not found in data")
    contingency_table = pd.crosstab(reference_data[column], incoming_data[column])
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    return {'chi2_stat': chi2_stat, 'p_value': p_value}


def calculate_psi(expected, actual, buckets=10):
    print("Reference data types:\n", expected.dtypes)
    print("New data types:\n", actual.dtypes)

    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    epsilon = 1e-8
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.histogram(scale_range(expected, 0, 100), breakpoints)[0] / len(expected)
    actual_percents = np.histogram(scale_range(actual, 0, 100), breakpoints)[0] / len(actual)
    expected_percents = np.clip(expected_percents, epsilon, None)
    actual_percents = np.clip(actual_percents, epsilon, None)
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value


def calculate_kl_divergence(reference_data, incoming_data, column, bins=10):
    reference_hist, bin_edges = np.histogram(reference_data[column], bins=bins, density=True)
    incoming_hist, _ = np.histogram(incoming_data[column], bins=bin_edges, density=True)

    kl_divergence = entropy(reference_hist, incoming_hist)
    return kl_divergence


def check_for_data_drift(timestamp, db_connection_url):
    engine = create_engine(db_connection_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    new_data = get_new_data(session)
    reference_data = get_reference_data(session)

    if new_data.empty:
        print("No new data available for drift detection.")
        return  # Exit early if there is no new data

    # Perform statistical tests
    ks_results = ks_test(new_data, reference_data)
    psi_results = calculate_psi(reference_data, new_data)
    kl_divergence_results = {
        column: calculate_kl_divergence(reference_data, new_data, column)
        for column in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    }

    # Log the results to MLflow
    with mlflow.start_run(run_name="Data Drift Check") as run:
        for column, result in ks_results.items():
            mlflow.log_metric(f"ks_test_p_value_{column}", result['p_value'])
        mlflow.log_metric("psi", psi_results)
        for column, value in kl_divergence_results.items():
            mlflow.log_metric(f"kl_divergence_{column}", value)

    # Check thresholds and determine if retraining is needed
    drift_detected = (any(result['p_value'] > 0.05 for result in ks_results.values()) or
                      psi_results > 0.1 or
                      any(v > 0.5 for v in kl_divergence_results.values()))

    if drift_detected:
        print("Data drift detected. Triggering retraining.")
        train_model(timestamp=timestamp, retraining_type='data_drift')
    else:
        print("No data drift detected.")

    return drift_detected


def get_latest_drift_metrics():
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name("Data Drift Check").experiment_id

    runs = client.search_runs(experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return None

    latest_run = runs[0]
    ks_test_p_value = latest_run.data.metrics.get("ks_test_p_value")
    psi = latest_run.data.metrics.get("psi")
    kl_divergence = {k: v for k, v in latest_run.data.metrics.items() if k.startswith("kl_divergence_")}

    return {
        "ks_test_p_value": ks_test_p_value,
        "psi": psi,
        "kl_divergence": kl_divergence
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_drift_check.py <timestamp> <db_connection_url>")
        sys.exit(1)

    timestamp = sys.argv[1]
    db_connection_url = sys.argv[2]

    check_for_data_drift(timestamp, db_connection_url)
