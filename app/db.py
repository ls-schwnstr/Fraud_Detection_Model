from datetime import timezone, datetime
import sqlalchemy
from sqlalchemy import DateTime, func, create_engine, Column, Integer, Float, String, desc
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import os

# Load credentials from environment variables
DB_PASSWORD = os.getenv('DB_PASSWORD')


# Database setup
def get_db_connection_url():
    """
    Returns the database connection URL using environment variables.
    """
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    print(f"DB_PASSWORD: {DB_PASSWORD}")
    print(f"DB_USER: {db_user}")

    if not db_user or not db_password:
        raise ValueError("Database environment variables are not set properly.")

    return f'mssql+pyodbc://{db_user}:{db_password}@fraud-detection-server.database.windows.net:1433/fraud_detection_db?driver=ODBC+Driver+17+for+SQL+Server'


DATABASE_URL = get_db_connection_url()
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()


# Session setup
def get_session():
    Session = scoped_session(sessionmaker(bind=engine))
    return Session()


# Define the raw_data model
class RawData(Base):
    __tablename__ = 'raw_data'
    id = Column(Integer, primary_key=True)
    step = Column(Integer)
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    type = Column(String)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)  # Add timestamp


# Define the raw_data model
class ProcessedData(Base):
    __tablename__ = 'processed_data'
    id = Column(Integer, primary_key=True)
    step = Column(Integer)
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    type_CASH_OUT = Column(Integer)
    type_DEBIT = Column(Integer)
    type_PAYMENT = Column(Integer)
    type_TRANSFER = Column(Integer)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)  # Add timestamp


# Define the prediction model
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    step = Column(Integer)
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)
    type_CASH_OUT = Column(Integer)
    type_DEBIT = Column(Integer)
    type_PAYMENT = Column(Integer)
    type_TRANSFER = Column(Integer)
    isFraud = Column(Integer)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)


# Define the Retraining model
class RetrainingLog(Base):
    __tablename__ = 'retraining_log'
    id = Column(Integer, primary_key=True)
    retraining_timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    retraining_type = Column(String, nullable=False)
    run_id = Column(String, nullable=False)

    #def __init__(self, retraining_timestamp=None):
    #   # Call the superclass constructor
    #  super(RetrainingLog, self).__init__()
    # if retraining_timestamp is None:
    #    retraining_timestamp = datetime.utcnow()
    #self.retraining_timestamp = retraining_timestamp


# Create the table
Base.metadata.create_all(engine)


def add_raw_data(session, data, timestamp=None):
    if timestamp is None:
        timestamp = datetime.utcnow()
    try:
        # Create a new RawData instance
        raw_data_entry = RawData(
            step=data['step'],
            amount=data['amount'],
            oldbalanceOrg=data['oldbalanceOrg'],
            newbalanceOrig=data['newbalanceOrig'],
            oldbalanceDest=data['oldbalanceDest'],
            newbalanceDest=data['newbalanceDest'],
            type=data['type'],
            timestamp=timestamp
        )
        # Add and commit the entry to the database
        session.add(raw_data_entry)
        session.commit()
        print("Data inserted successfully")
    except sqlalchemy.exc.IntegrityError as e:
        print(f"IntegrityError: {e}")
        session.rollback()


def get_latest_raw_data(session):
    """Retrieve the most recent entry from the raw_data table."""
    return session.query(RawData).order_by(desc(RawData.timestamp)).first()


def get_latest_processed_data(session):
    """Retrieve the most recent entry from the raw_data table."""
    try:
        result = session.query(ProcessedData).order_by(desc(ProcessedData.timestamp)).first()

        # Convert SQLAlchemy results to a DataFrame
        data_df = pd.DataFrame([result.__dict__])

        # Drop SQLAlchemy metadata column if it exists
        if '_sa_instance_state' in data_df.columns:
            data_df = data_df.drop(columns=['_sa_instance_state'])

        return data_df

    except Exception as e:
        print(f"Error retrieving predicted data: {e}")
        return []


def add_processed_data(session, data, timestamp=None):
    try:
        # Extract transaction type from data
        type_CASH_OUT = int(data.get('CASH_OUT', 0))
        type_DEBIT = int(data.get('DEBIT', 0))
        type_PAYMENT = int(data.get('PAYMENT', 0))
        type_TRANSFER = int(data.get('TRANSFER', 0))

        # Create a new ProcessedData instance with the appropriate type column
        processed_data_entry = ProcessedData(
            step=data['step'],
            amount=data['amount'],
            oldbalanceOrg=data['oldbalanceOrg'],
            newbalanceOrig=data['newbalanceOrig'],
            oldbalanceDest=data['oldbalanceDest'],
            newbalanceDest=data['newbalanceDest'],
            type_CASH_OUT=type_CASH_OUT,
            type_DEBIT=type_DEBIT,
            type_PAYMENT=type_PAYMENT,
            type_TRANSFER=type_TRANSFER,
            timestamp=timestamp if timestamp else func.now()
        )

        # Add the new entry to the session and commit
        session.add(processed_data_entry)
        session.commit()

        print("Processed data added successfully.")
    except Exception as e:
        print(f"Error adding processed data: {e}")
        session.rollback()


def add_predicted_data(session, data, timestamp=None):
    try:
        # Convert data to a DataFrame if it's a dictionary
        if isinstance(data, dict):
            print("it is a dictionary")
            data = pd.DataFrame([data])

        # Check if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            print(data.head())
            print("it is a dataframe")
            for index, row in data.iterrows():
                # Prepare the timestamp
                entry_timestamp = timestamp if timestamp is not None else datetime.now()

                # Create a new Prediction instance
                predicted_data_entry = Prediction(
                    step=row.get('step', 0),
                    amount=row.get('amount', 0.0),
                    oldbalanceOrg=row.get('oldbalanceOrg', 0.0),
                    newbalanceOrig=row.get('newbalanceOrig', 0.0),
                    oldbalanceDest=row.get('oldbalanceDest', 0.0),
                    newbalanceDest=row.get('newbalanceDest', 0.0),
                    type_CASH_OUT=int(row.get('type_CASH_OUT', False)),
                    type_DEBIT=int(row.get('type_DEBIT', False)),
                    type_PAYMENT=int(row.get('type_PAYMENT', False)),
                    type_TRANSFER=int(row.get('type_TRANSFER', False)),
                    isFraud=int(row.get('isFraud', 0)),
                    timestamp=entry_timestamp
                )
                session.add(predicted_data_entry)
        else:
            raise ValueError("Unsupported data format")

        session.commit()
        print("Predicted data added successfully.")
    except Exception as e:
        print(f"Error adding predicted data: {e}")
        session.rollback()


def add_retraining_log(session, timestamp, retraining_type, run_id):
    # Convert timestamp to datetime object if it's a string
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    elif timestamp is None:
        timestamp = datetime.utcnow()

    # Create a new RetrainingLog entry
    new_retraining_event = RetrainingLog(
        retraining_timestamp=timestamp,
        retraining_type=retraining_type,
        run_id=run_id
    )

    # Add the entry to the session and commit
    session.add(new_retraining_event)
    session.commit()


def get_predicted_data(session):
    """Retrieve the data from the predictions table."""
    try:
        print("trying to retrieve data from prediction table")
        results = session.query(Prediction).all()

        print("data retrieved")

        # Convert SQLAlchemy results to a DataFrame
        data_df = pd.DataFrame([r.__dict__ for r in results])
        print("Data retrieved and converted successfully.")

        # Drop SQLAlchemy metadata column if it exists
        if '_sa_instance_state' in data_df.columns:
            data_df = data_df.drop(columns=['_sa_instance_state'])

        return data_df

    except Exception as e:
        print(f"Error retrieving predicted data: {e}")
        return []


def get_reference_data(session):
    print("get reference data")
    last_retraining = session.query(RetrainingLog).order_by(desc(RetrainingLog.retraining_timestamp)).first()
    if not last_retraining:
        print("no retraining so far")
        return pd.DataFrame()  # No retraining has been done, return an empty DataFrame

    last_retraining_time = last_retraining.retraining_timestamp

    print(f"Last retraining time: {last_retraining_time}")

    if last_retraining_time.tzinfo is None:
        last_retraining_time = last_retraining_time.replace(tzinfo=timezone.utc)

    reference_data = session.query(Prediction).filter(Prediction.timestamp <= last_retraining_time).all()
    print(f"Number of reference data records retrieved: {len(reference_data)}")
    reference_df = pd.DataFrame([r.__dict__ for r in reference_data])

    if '_sa_instance_state' in reference_df.columns:
        reference_df = reference_df.drop(columns=['_sa_instance_state'])

    reference_df = reference_df.drop(columns=['timestamp'], errors='ignore')

    return reference_df


def get_new_data(session):
    last_retraining = session.query(RetrainingLog).order_by(desc(RetrainingLog.retraining_timestamp)).first()
    if not last_retraining:
        print("no retraining so far")
        return pd.DataFrame()  # No retraining has been done, return an empty DataFrame

    last_retraining_time = last_retraining.retraining_timestamp

    if last_retraining_time.tzinfo is None:
        last_retraining_time = last_retraining_time.replace(tzinfo=timezone.utc)

    print(f"Last retraining time: {last_retraining_time}")

    new_data = session.query(Prediction).filter(Prediction.timestamp > last_retraining_time).all()
    print(f"Number of new data records retrieved: {len(new_data)}")
    new_data_df = pd.DataFrame([r.__dict__ for r in new_data])

    if '_sa_instance_state' in new_data_df.columns:
        new_data_df = new_data_df.drop(columns=['_sa_instance_state'])

    new_data_df = new_data_df.drop(columns=['timestamp'], errors='ignore')

    return new_data_df
