import traceback
from flask import render_template, request, redirect, url_for, session as flask_session, flash, jsonify
from app import app
import os
import pandas as pd
import threading
from app.db import get_session


# Load credentials from environment variables
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

file_path = os.path.join(os.path.dirname(__file__), 'Fraud.csv')
model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')

# Global variable for training status
training_status = {'status': 'not_started'}


def background_train():
    from app.models.model import preprocess_data_for_training, train_model, add_predicted_data
    global training_status
    training_status['status'] = 'in_progress'
    try:
        db_session = get_session()
        print("Starting model training...")
        new_data = pd.read_csv(file_path, delimiter=';', nrows=500)
        processed_data = preprocess_data_for_training(new_data)
        print("Preprocessed data succesfully")

        # Save Fraud.csv to predictions table
        add_predicted_data(db_session, processed_data)
        print("Added preprocessed data sucessfully")

        run_id = train_model()
        print(f'Training complete with run_id: {run_id}')
        training_status['status'] = 'complete'
    except Exception as e:
        print(f'Error during training: {e}')
        print(traceback.format_exc())
        training_status['status'] = 'error'


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            flask_session['username'] = username
            return redirect(url_for('train'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'username' in flask_session:
        if os.path.isfile(model_path):
            # If model already exists
            return redirect(url_for('dashboard'))
        elif training_status['status'] == 'complete':
            # If training is complete, redirect to dashboard
            return redirect(url_for('dashboard'))
        elif training_status['status'] != 'in_progress':
            # Start the model training in the background if not already in progress
            training_thread = threading.Thread(target=background_train)
            training_thread.start()
            return render_template('training.html')
        else:
            # If training is in progress, show the training page
            flash('Model training is still in progress. Please wait.')
            return render_template('training.html')
    else:
        print('User not in session, redirecting to login')
        return redirect(url_for('login'))


@app.route('/check-training-status')
def check_training_status():
    print(f'Checking training status: {training_status["status"]}')
    return jsonify({"status": training_status['status']})


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    from app.models.model import preprocess_input_data
    from app.db import add_raw_data, add_processed_data, get_latest_raw_data
    if 'username' in flask_session:
        db_session = get_session()

        if os.path.isfile(model_path) or training_status['status'] == 'complete':
            if request.method == 'POST':
                try:
                    if request.is_json:
                        # Handle JSON input
                        input_data = request.get_json()
                    else:
                        # Handle form input
                        input_data = {
                            'step': 1,
                            'amount': float(request.form['amount']),
                            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
                            'newbalanceOrig': float(request.form['newbalanceOrig']),
                            'oldbalanceDest': float(request.form['oldbalanceDest']),
                            'newbalanceDest': float(request.form['newbalanceDest']),
                            'type': request.form['type']
                        }

                    # Print the input data for debugging
                    print(f"Received input data: {input_data}")

                    # Add data to database
                    add_raw_data(db_session, input_data)

                    # Retrieve the raw data for preprocessing
                    raw_data_entry = get_latest_raw_data(db_session)

                    # Convert the SQLAlchemy object to a dictionary and exclude internal state
                    raw_data_dict = {column.name: getattr(raw_data_entry, column.name)
                                     for column in raw_data_entry.__table__.columns}

                    # Print the retrieved raw data for debugging
                    print(f"Retrieved raw data: {raw_data_dict}")

                    # Convert to DataFrame
                    raw_data_df = pd.DataFrame([raw_data_dict])
                    print(f"Retrieved raw data DataFrame: {raw_data_df}")

                    # Process the data
                    processed_data_df = preprocess_input_data(raw_data_df)

                    # Convert processed data DataFrame to dictionary format
                    processed_data_dict = processed_data_df.to_dict(orient='records')[0]

                    # Add data to the database
                    add_processed_data(db_session, processed_data_dict)

                    # Commit the session to save the processed data
                    db_session.commit()

                    return redirect(url_for('predict'))

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    return f"An error occurred: {str(e)}", 500

            return render_template('dashboard.html')
        elif training_status['status'] == 'in_progress':
            flash('Model training is still in progress. Please wait or check back later.')
            return redirect(url_for('train'))
        else:
            flash('Model not trained yet. Redirecting to training page.')
            return redirect(url_for('train'))
    else:
        flash('You are not logged in.')
        return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    from app.models.model import make_predictions_and_check_drift
    from app.db import get_latest_processed_data
    db_session = get_session()
    try:
        print("Received prediction request")

        # Retrieve the latest processed data entry
        processed_data_df = get_latest_processed_data(db_session)

        # Call the shared prediction function
        prediction = make_predictions(processed_data_df)
        #prediction, drift_detected = make_predictions_and_check_drift(processed_data_df)

        return render_template('prediction.html', prediction=int(prediction), drift_detected=drift_detected)
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500

