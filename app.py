import random
from chatbot.handler import chat_blueprint
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
from symptoms import *
from ollama import chat
from typing import List
import bleach
from flask_socketio import SocketIO, emit
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = Flask(__name__)

app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_123")  # Fallback for development
socketio = SocketIO(app, cors_allowed_origins="*")

sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")

Rf = pickle.load(open('model/RandomForest.pkl', 'rb'))

# Here we make a dictionary of symptoms and diseases and preprocess it

symptoms_list = symptoms_list
diseases_list = diseases_list


symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}


# Here we created a function (information) to extract information from all the datasets

def information(predicted_dis):
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']
    disease_desciption = " ".join([w for w in disease_desciption])

    disease_precautions = precautions[precautions['Disease'] == predicted_dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]

    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']
    disease_medications = [med for med in disease_medications.values]

    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']
    disease_diet = [die for die in disease_diet.values]

    disease_workout = workout[workout['disease'] == predicted_dis]['workout']

    return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout


# This is the function that passes the user input symptoms to our Model
def predicted_value(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))
    for i in patient_symptoms:
        i_vector[symptoms_list_processed[i]] = 1
    return diseases_list[Rf.predict([i_vector])[0]]


# Function to correct the spellings of the symptom (if any)
def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    # If the similarity score is above a certain threshold, consider it a match
    if score >= 80:
        return closest_match
    else:
        return None


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('Symptome.html', message=message)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            patient_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            patient_symptoms = [symptom.strip("[]' ") for symptom in patient_symptoms]

            # Correct the spelling of symptoms
            corrected_symptoms = []
            for symptom in patient_symptoms:
                corrected_symptom = correct_spelling(symptom)
                if corrected_symptom:
                    corrected_symptoms.append(corrected_symptom)
                else:
                    message = f"Symptom '{symptom}' not found in the database."
                    return render_template('Symptome.html', message=message)

            # Predict the disease using corrected symptoms
            predicted_disease = predicted_value(corrected_symptoms)
            dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            # converting the string into a list format before returning to the frontend
            medication_list = ast.literal_eval(medications[0])
            medications = []
            for item in medication_list:
                medications.append(item)

            diet_list = ast.literal_eval(rec_diet[0])
            rec_diet = []
            for item in diet_list:
                rec_diet.append(item)
            return render_template('Symptome.html', symptoms=corrected_symptoms, predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('Symptome.html')

@app.route('/services')
def services():
    return render_template('Symptome.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analytic')
def analytics():
    return render_template('analytics.html')


# Route for predicting diabetes risk with LLM integration
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    # Fetch parameters from the request
    pregnancies = data['pregnancies']
    glucose = data['glucose']
    blood_pressure = data['blood_pressure']
    skin_thickness = data['skin_thickness']
    insulin = data['insulin']
    bmi = data['bmi']
    diabetes_pedigree = data['diabetes_pedigree']
    age = data['age']

    # Prepare prompt for the model
    prompt = f"""
    Given the following health data:
    - Pregnancies: {pregnancies}
    - Glucose: {glucose}
    - Blood Pressure: {blood_pressure}
    - Skin Thickness: {skin_thickness}
    - Insulin: {insulin}
    - BMI: {bmi}
    - Diabetes Pedigree: {diabetes_pedigree}
    - Age: {age}

    Predict the risk of diabetes (Low, Moderate, or High) and give a confidence percentage.
    answer in 40-50 words
    """

    # Make the request to the model
    prediction = ""
    try:
        stream = chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        # Collect all chunks of the response
        for chunk in stream:
            if isinstance(chunk, tuple):
                # If chunk is a tuple, get the first element
                chunk_content = chunk[0]
            else:
                # If chunk is a dict, get the message content
                chunk_content = chunk.get('message', {}).get('content', '')

            prediction += str(chunk_content)

    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

    # Return prediction
    return jsonify({
        'prediction': prediction
    })


# Route for calculating BMI
@app.route('/calculate_bmi', methods=['POST'])
def calculate_bmi():
    weight = float(request.json['weight'])
    height = float(request.json['height'])
    bmi = weight / (height ** 2)
    return jsonify({'bmi': f"BMI: {bmi:.2f}"})


# Route for predicting heart disease risk with LLM integration
@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    data = request.json
    # Extract parameters for prediction
    age = data['age']
    cholesterol = data['cholesterol']
    blood_pressure = data['blood_pressure']
    exercise = data['exercise']

    # Prepare the prompt for the model
    prompt = f"""
    Given the following health data:
    - Age: {age}
    - Cholesterol: {cholesterol}
    - Blood Pressure: {blood_pressure}
    - Exercise: {exercise}

    Predict the risk of heart disease (Low, Moderate, or High) and provide reasoning.
    Be consice and answer in 20-30 words
    """

    # Make the request to the model
    prediction = ""
    try:
        stream = chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        # Collect all chunks of the response
        for chunk in stream:
            if isinstance(chunk, tuple):
                # If chunk is a tuple, get the first element
                chunk_content = chunk[0]
            else:
                # If chunk is a dict, get the message content
                chunk_content = chunk.get('message', {}).get('content', '')

            prediction += str(chunk_content)

    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

    return jsonify({'prediction': prediction})



@app.route('/dashboard')
def dashboard():
    # Sample data for dashboard
    # You can replace these values with real data from your form submissions or predictions
    diabetes_data = {
        'pregnancies': 2,
        'glucose': 140,
        'blood_pressure': 80,
        'skin_thickness': 20,
        'insulin': 90,
        'bmi': 29.4,
        'diabetes_pedigree': 0.5,
        'age': 45,
        'prediction': "High risk (Confidence: 85%)"
    }

    heart_disease_data = {
        'age': 55,
        'cholesterol': 230,
        'blood_pressure': 140,
        'exercise': "No",
        'prediction': "Moderate risk (Confidence: 75%)"
    }

    bmi_data = {
        'weight': 70,
        'height': 1.75,
        'bmi': 22.9  # Calculated BMI
    }

    return render_template(
        'dashboard.html',
        diabetes_data=diabetes_data,
        heart_disease_data=heart_disease_data,
        bmi_data=bmi_data
    )


def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    return bleach.clean(text, tags=[], strip=True)


@app.route('/Health')
def Health():
    return render_template('Health.html')


@app.route('/correlate_health', methods=['POST'])
def correlate_health():
    try:
        data = request.json

        # Sanitize inputs
        medications = [sanitize_input(med) for med in data.get('medications', [])]
        allergies = [sanitize_input(allergy) for allergy in data.get('allergies', [])]
        genetic_history = sanitize_input(data.get('genetic_history', ""))

        # Create prompt for LLaMA
        prompt = f"""
        A patient has provided the following health data:
        - Current Medications: {', '.join(medications)}
        - Known Allergies: {', '.join(allergies)}
        - Genetic Disease History: {genetic_history}

        Correlate the medications, allergies, and genetic history to identify potential risks, 
        drug interactions, and hereditary concerns. Provide a summary of potential health risks 
        and suggest precautions. Answer in 50-70 words.
        """

        # Get response from LLaMA
        correlation_analysis = ""
        try:
            stream = chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )

            for chunk in stream:
                if isinstance(chunk, tuple):
                    chunk_content = chunk[0]
                else:
                    chunk_content = chunk.get('message', {}).get('content', '')
                correlation_analysis += str(chunk_content)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"correlation_analysis": correlation_analysis})

    except Exception as e:
        return jsonify({"error": "Invalid request"}), 400


@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/locator')
def locator():
    return render_template('locator.html')



@app.route('/virtual-assistant')
def virtual_assistant():
    return render_template('virtual-assistant.html')

# @app.route('/')
# def index():
#     return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.debug('Client connected')
    emit('connection_success', {'data': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect(sid):  # Fixed: Added sid parameter
    logger.debug(f'Client disconnected: {sid}')

@socketio.on('offer')
def handle_offer(data):
    logger.debug('Handling offer')
    emit('offer', data, broadcast=True, include_self=False)

@socketio.on('answer')
def handle_answer(data):
    logger.debug('Handling answer')
    emit('answer', data, broadcast=True, include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    logger.debug('Handling ICE candidate')
    emit('ice-candidate', data, broadcast=True, include_self=False)

@socketio.on_error()
def error_handler(e):
    logger.error(f'SocketIO error: {str(e)}')
    emit('error', {'message': 'An error occurred during signaling'})




app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Register blueprint
app.register_blueprint(chat_blueprint)

if __name__ == '__main__':
    app.run(debug=True)