import mlflow

mlflow.set_tracking_uri("http://mlflow:5004")

from flask import Flask
from config import Config


app = Flask(__name__)
app.config.from_object(Config)


from app import routes
