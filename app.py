import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Apply dark background and full-width padding */
    .stApp {
        background-color: #1e1e2f;
        padding: 2rem 3rem;
    }

    /* Remove default Streamlit padding/margin */
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Title and Section Styling */
    .stTitle {
        color: #ffffff;
        font-size: 2.5rem;
        text-align: left;
    }

    .stExpanderHeader {
        background-color: #333a49;
        color: #ffffff;
    }

    .stExpanderContent {
        background-color: #2c3144;
    }

    .stButton>button {
        background-color: #f2b632;
        color: #1e1e2f;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #e0a429;
    }

    .stSlider, .stNumberInput, .stSelectbox {
        background-color: #2c3144;
        color: white;
        border: none;
        border-radius: 5px;
    }

    .stSlider>div>div>div {
        color: white !important;
    }

    .stNumberInput>div>input {
        color: white !important;
    }

    /* Expander & Inputs to take full width */
    .css-1d391kg, .css-1lcbmhc, .css-1v3fvcr {
        max-width: 100% !important;
        margin: 0 !important;
    }

    /* Force full width for main container */
    section[data-testid="stSidebar"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)



models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
}
feature_names = joblib.load("models/feature_names.pkl")
scaler = joblib.load("models/scaler.pkl")  


st.title("Stress Level Prediction App")
st.write("Enter your personal, lifestyle, and medical information to predict stress level")


selected_model_name = st.selectbox("Select Algorithm", list(models.keys()))
model = models[selected_model_name]


with st.expander("Personal Information", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    
    col4, col5, col6 = st.columns(3)
    with col4:
        occupation = st.selectbox(
            "Occupation (exact name)",
            [
                "Software Engineer", "Marketing Manager", "Data Scientist", "Teacher",
                "Doctor", "Graphic Designer", "Civil Engineer", "Business Owner",
                "Nurse", "Student", "Other"
            ]
        )
    with col5:
        sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 12.0, 7.0)
    with col6:
        travel_time = st.slider("Travel Time (hours)", 0.0, 5.0, 1.0)
    
    col7, col8, col9 = st.columns(3)
    with col7:
        sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 3)
    with col8:
        work_hours = st.number_input("Work Hours per day", 0.0, 16.0, 8.0)
    with col9:
        social = st.slider("Social Interactions (1-10)", 1, 10, 5)


with st.expander(" Lifestyle", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        wake_up_time = st.number_input("Wake Up Time (minutes from 00:00)", 0, 1440, 420)
    with col2:
        bed_time = st.number_input("Bed Time (minutes from 00:00)", 0, 1440, 1320)
    with col3:
        physical_activity = st.slider("Physical Activity Level", 0.0, 5.0, 2.0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        screen_time = st.slider("Screen Time (hours)", 0.0, 10.0, 4.0)
    with col5:
        caffeine = st.slider("Caffeine Intake", 0, 5, 1)
    with col6:
        alcohol = st.slider("Alcohol Intake", 0, 5, 0)
    
    col7, col8, col9 = st.columns(3)
    with col7:
        smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
    with col8:
        meditation = st.selectbox("Meditation Practice", ["Yes", "No"])
    with col9:
        exercise_type = st.selectbox(
            "Exercise Type",
            ["Cardio", "Yoga", "Strength Training", "Aerobics", "Walking", "Pilates"]
        )


with st.expander(" Medical Information", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        blood_pressure = st.number_input("Blood Pressure", 80, 200, 120)
    with col2:
        cholesterol = st.number_input("Cholesterol Level", 100, 300, 180)
    with col3:
        blood_sugar = st.number_input("Blood Sugar Level", 60, 200, 90)


if st.button(" Predict Stress Level"):
   
    input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

  
    numerical_features = {
        "Age": age,
        "Sleep_Duration": sleep_duration,
        "Sleep_Quality": sleep_quality,
        "Wake_Up_Time": wake_up_time,
        "Bed_Time": bed_time,
        "Physical_Activity": physical_activity,
        "Screen_Time": screen_time,
        "Caffeine_Intake": caffeine,
        "Alcohol_Intake": alcohol,
        "Work_Hours": work_hours,
        "Travel_Time": travel_time,
        "Social_Interactions": social,
        "Blood_Pressure": blood_pressure,
        "Cholesterol_Level": cholesterol,
        "Blood_Sugar_Level": blood_sugar
    }
    
    for col, val in numerical_features.items():
        if col in input_data.columns:
            input_data[col] = val

    
    cat_features = {
        "Gender": gender,
        "Marital_Status": marital_status,
        "Smoking_Habit": smoking,
        "Meditation_Practice": meditation,
        "Exercise_Type": exercise_type,
        "Occupation": occupation
    }
    
    for col, val in cat_features.items():
        col_name = f"{col}_{val}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

    
    input_data_scaled = scaler.transform(input_data)
    

    input_data_scaled = pd.DataFrame(input_data_scaled, columns=feature_names)


    
    print(input_data_scaled)
    prediction = model.predict(input_data_scaled)[0]
    
    stress_map = {
        1: "Low Stress",
        2: " Medium Stress",
        3: " High Stress"
    }
    
    st.subheader("Prediction Result")
    st.success(stress_map.get(prediction, "Unknown"))