import mlflow
import os

mlflow.set_tracking_uri("http://mlflow:5004")

from flask import Flask
from app.config import Config


app = Flask(__name__)
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(Config)


from app import routes