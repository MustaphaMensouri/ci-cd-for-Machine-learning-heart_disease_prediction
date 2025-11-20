import gradio as gr
import joblib
import pandas as pd

# Load your model
model = joblib.load("./Model/heart_disease_model.pkl")

def predict_heart_disease(age, gender, weight, height, bmi, smoking, alcohol_intake, 
                         physical_activity, diet, stress_level, hypertension, 
                         diabetes, hyperlipidemia, family_history, previous_heart_attack,
                         systolic_bp, diastolic_bp, heart_rate, blood_sugar_fasting, 
                         cholesterol_total):
    """Predict heart disease based on patient features.
    
    Args:
        Various patient health parameters
        
    Returns:
        str: Predicted heart disease risk (0 or 1)
    """
    # Create a DataFrame with the same column names and order as training data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'BMI': [bmi],
        'Smoking': [smoking],
        'Alcohol_Intake': [alcohol_intake],
        'Physical_Activity': [physical_activity],
        'Diet': [diet],
        'Stress_Level': [stress_level],
        'Hypertension': [hypertension],
        'Diabetes': [diabetes],
        'Hyperlipidemia': [hyperlipidemia],
        'Family_History': [family_history],
        'Previous_Heart_Attack': [previous_heart_attack],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Heart_Rate': [heart_rate],
        'Blood_Sugar_Fasting': [blood_sugar_fasting],
        'Cholesterol_Total': [cholesterol_total]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    try:
        probability = model.predict_proba(input_data)[0]
        confidence = f"Confidence: {max(probability):.2%}"
    except AttributeError:
        confidence = ""
    
    label = "Heart Disease Risk: " + ("HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✓")
    
    return f"{label}\n{confidence}" if confidence else label

inputs = [
    gr.Slider(18, 100, step=1, value=50, label="Age"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Slider(40, 150, step=1, value=70, label="Weight (kg)"),
    gr.Slider(140, 210, step=1, value=170, label="Height (cm)"),
    gr.Slider(15, 50, step=0.1, value=25, label="BMI"),
    gr.Radio(["Never", "Former", "Current"], label="Smoking Status"),
    gr.Radio(["Low", "Moderate", "High"], label="Alcohol Intake", value="Low"),
    gr.Radio(["Sedentary", "Moderate", "Active"], label="Physical Activity"),
    gr.Radio(["Healthy", "Average", "Unhealthy"], label="Diet"),
    gr.Radio(["Low", "Medium", "High"], label="Stress Level"),
    gr.Radio([0, 1], label="Hypertension (0=No, 1=Yes)", value=0),
    gr.Radio([0, 1], label="Diabetes (0=No, 1=Yes)", value=0),
    gr.Radio([0, 1], label="Hyperlipidemia (0=No, 1=Yes)", value=0),
    gr.Radio([0, 1], label="Family History (0=No, 1=Yes)", value=0),
    gr.Radio([0, 1], label="Previous Heart Attack (0=No, 1=Yes)", value=0),
    gr.Slider(90, 200, step=1, value=120, label="Systolic BP (mmHg)"),
    gr.Slider(60, 120, step=1, value=80, label="Diastolic BP (mmHg)"),
    gr.Slider(50, 120, step=1, value=75, label="Heart Rate (bpm)"),
    gr.Slider(70, 200, step=1, value=100, label="Blood Sugar Fasting (mg/dL)"),
    gr.Slider(150, 300, step=1, value=200, label="Total Cholesterol (mg/dL)"),
]

outputs = gr.Textbox(label="Prediction Result")

examples = [
    [59, "Female", 53, 158, 35.7, "Never", "Low", "Sedentary", "Average", "High", 0, 1, 0, 0, 0, 150, 94, 78, 85, 163],
    [73, "Male", 91, 158, 34.8, "Never", "Low", "Active", "Unhealthy", "Low", 1, 0, 0, 1, 0, 146, 62, 65, 132, 232],
    [33, "Male", 69, 197, 39.7, "Former", "Moderate", "Moderate", "Average", "Low", 0, 0, 1, 0, 0, 178, 62, 97, 154, 250],
]

title = "Heart Disease Risk Prediction"
description = "Enter patient health metrics to predict heart disease risk. Fill in all fields for accurate prediction."
article = "This app is part of a CI/CD pipeline for Machine Learning. It demonstrates automated training, evaluation, and deployment of models using GitHub Actions."

gr.Interface(
    fn=predict_heart_disease,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
