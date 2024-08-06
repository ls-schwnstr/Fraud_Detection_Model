from datetime import timezone
import sqlalchemy
from sqlalchemy import DateTime, func, create_engine, Column, Integer, Float, String, Boolean, desc, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database setup
DATABASE_URL = 'sqlite:///fraud_detection.db'
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()


# Session setup
def get_session():
    Session = sessionmaker(bind=engine)
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
    timestamp = Column(DateTime(timezone=True), default=func.now())  # Add timestamp


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
    timestamp = Column(DateTime(timezone=True), default=func.now())  # Add timestamp


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
    timestamp = Column(DateTime(timezone=True), default=func.now())


# Define the Retraining model
class RetrainingLog(Base):
    __tablename__ = 'retraining_log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    retraining_timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)


# Create the table
Base.metadata.create_all(engine)


def add_raw_data(session, data):
    try:
        # Create a new RawData instance
        raw_data_entry = RawData(
            step=data['step'],
            amount=data['amount'],
            oldbalanceOrg=data['oldbalanceOrg'],
            newbalanceOrig=data['newbalanceOrig'],
            oldbalanceDest=data['oldbalanceDest'],
            newbalanceDest=data['newbalanceDest'],
            type=data['type']
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


def add_processed_data(session, data):
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
            type_TRANSFER=type_TRANSFER
        )

        # Add the new entry to the session and commit
        session.add(processed_data_entry)
        session.commit()

        print("Processed data added successfully.")
    except Exception as e:
        print(f"Error adding processed data: {e}")
        session.rollback()


def add_predicted_data(session, data):
    try:
        # Convert data to a DataFrame if it's a dictionary
        if isinstance(data, dict):
            print("it is a dictionary")
            data = pd.DataFrame([data])

        # Check if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            print("it is a dataframe")
            for index, row in data.iterrows():
                predicted_data_entry = Prediction(
                    step=row['step'],
                    amount=row['amount'],
                    oldbalanceOrg=row['oldbalanceOrg'],
                    newbalanceOrig=row['newbalanceOrig'],
                    oldbalanceDest=row['oldbalanceDest'],
                    newbalanceDest=row['newbalanceDest'],
                    type_CASH_OUT=row.get('type_CASH_OUT', 0),
                    type_DEBIT=row.get('type_DEBIT', 0),
                    type_PAYMENT=row.get('type_PAYMENT', 0),
                    type_TRANSFER=row.get('type_TRANSFER', 0),
                    isFraud=row['isFraud']
                )
                session.add(predicted_data_entry)
        else:
            raise ValueError("Unsupported data format")

        session.commit()
        print("Predicted data added successfully.")
    except Exception as e:
        print(f"Error adding predicted data: {e}")
        session.rollback()


def add_retraining_log(session):
    new_retraining_event = RetrainingLog()
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
        return pd.DataFrame()  # No retraining has been done, return an empty DataFrame

    last_retraining_time = last_retraining.retraining_timestamp

    if last_retraining_time.tzinfo is None:
        last_retraining_time = last_retraining_time.replace(tzinfo=timezone.utc)

    reference_data = session.query(Prediction).filter(Prediction.timestamp <= last_retraining_time).all()
    reference_df = pd.DataFrame([r.__dict__ for r in reference_data])

    if '_sa_instance_state' in reference_df.columns:
        reference_df = reference_df.drop(columns=['_sa_instance_state'])

    reference_df = reference_df.drop(columns=['timestamp'], errors='ignore')

    return reference_df


def get_new_data(session):
    print("get new data")

    last_retraining = session.query(RetrainingLog).order_by(desc(RetrainingLog.retraining_timestamp)).first()
    print("last retraining retrieved")
    if not last_retraining:
        print("no retraining so far")
        return pd.DataFrame()  # No retraining has been done, return an empty DataFrame

    last_retraining_time = last_retraining.retraining_timestamp

    if last_retraining_time.tzinfo is None:
        last_retraining_time = last_retraining_time.replace(tzinfo=timezone.utc)

    new_data = session.query(Prediction).filter(Prediction.timestamp > last_retraining_time).all()
    print("new data retrieved")
    new_data_df = pd.DataFrame([r.__dict__ for r in new_data])
    print("new data converted to DataFrame")

    if '_sa_instance_state' in new_data_df.columns:
        new_data_df = new_data_df.drop(columns=['_sa_instance_state'])

    new_data_df = new_data_df.drop(columns=['timestamp'], errors='ignore')

    return new_data_df




