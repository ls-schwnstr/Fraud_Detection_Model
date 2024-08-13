import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'my_precious')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
