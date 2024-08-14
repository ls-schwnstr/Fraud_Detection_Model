# Fraud Detection System

This repository contains a complete implementation of a fraud detection system using Flask for the REST API, MLflow for experiment tracking, and an Azure SQL Database for data storage. This project includes automatic data drift detection and retraining of the model when necessary.

# Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Data Drift Detection and Retraining](#data-drift-detection-and-retraining)


## Introduction

This project aims to build a scalable fraud detection system that monitors transaction data, checks for data drift, and retrains the machine learning model as necessary. The system is deployed using GitHub Actions.

## Features

- **Flask API**: Serves predictions via a RESTful API.
- **MLflow**: Tracks model experiments, metrics, and parameters.
- **Azure SQL Database**: Stores training data, logs, and other relevant information.
- **Data Drift Detection**: Monitors incoming data for changes and triggers model retraining when drift is detected.
- **GitHub Actions**: Automates workflows including data drift checks and model retraining.

## Setup Instructions

### Prerequisites

- **Python 3.8+** installed on your machine.
- **SQL Database**: Set up an Azure SQL Database or a database by your choice and obtain the connection URL.

### Clone the Repository

```bash
git clone https://github.com/your_username/Fraud_Detection_Model.git
cd Fraud_Detection_Model
```

### Set Up the Environment Variables

```env
USERNAME=your_username
PASSWORD=your_password
PAT_TOKEN=your_token
REPO_OWNER=your_ownername
REPO_NAME=your_reponame
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up MLflow

Start the MLflow server by running:

```bash
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root=mlruns/artifacts --host localhost --port 5004
```

### Setting up an Azure SQL Database
1. **Create an Azure SQL Database**
- Go to the [Azure portal](https://portal.azure.com/).
2. **Configure Firewall Rules**
3. **Obtain the Connection URL**
4. **Update the db_path**
 - Use the connection URL obtained in the previous step to set the `db_path` variable in the data-drift-check workflow and in the db.py. For example:

 - ```env
     db_path=mssql+pyodbc://adminuser:FraudDetection1!@fraud-detection-server.database.windows.net:1433/fraud_detection_db?driver=ODBC+Driver+17+for+SQL+Server
     ```

## Project Structure

- **app/** - Contains the Flask application code.
- **mlruns/** - Directory where MLflow logs runs.
- **requirements.txt** - Python dependencies required for the project.
- **.env.example** - Example of environment variables configuration.
- **README.md** - Detailed project documentation.

## Usage 
### Running Data Drift Check
The data drift check and the monthly retraining are automated via GitHub Actions. To manually trigger the check, go on the Actions Tab and select the workflow of your choice.

### Inserting manually Data
To manually insert data over the flask api, run run.py and open 

### API Endpoints

- **/login**
  - **Method:** POST
  - **Description:** Authenticates a user and redirects to the training page.

- **/train**
  - **Method:** POST
  - **Description:** Starts model training if not already in progress.

- **/dashboard**
  - **Method:** GET, POST
  - **Description:** Allows users to upload new data and get predictions.

### Data Drift Detection and Retraining

The project includes a mechanism for detecting data drift. The following methods are used:

- **Kolmogorov-Smirnov Test (KS Test):** Statistical test to check if the distributions of two samples are different.
- **Population Stability Index (PSI):** Measures the stability of the model over time.
- **Kullback-Leibler Divergence (KL Divergence):** Measures how one probability distribution diverges from a second expected probability distribution.

If any of these metrics exceed their thresholds, the model retrains automatically.


