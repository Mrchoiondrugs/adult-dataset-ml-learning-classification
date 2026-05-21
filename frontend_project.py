import streamlit as st
import requests

# Set page style and title
st.set_page_config(page_title="Income Prediction Dashboard", page_icon="💰", layout="centered")

st.title("💰 Adult Income Prediction Dashboard")
st.markdown("Enter the individual's details below to predict their income bracket using the live KNN Model.")

# Base URL pointing to your working FastAPI backend on port 8000
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.subheader("👤 Personal Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    relationship = st.selectbox("Relationship Status", ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried", "Other-relative"])

with col2:
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    education = st.selectbox("Highest Education Level", ["Bachelors", "Some-college", "HS-grad", "Masters", "Doctorate", "Assoc-acdm", "Assoc-voc", "Prof-school", "11th", "10th", "9th", "7th-8th", "12th", "Preschool"])
    educational_num = st.slider("Years of Education completed", min_value=1, max_value=16, value=13)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Canada", "Germany", "India", "Philippines", "United-Kingdom", "Other"])

st.subheader("💼 Employment & Financials")
col3, col4 = st.columns(2)

with col3:
    workclass = st.selectbox("Employment Type (Workclass)", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Prof-specialty", "Craft-repair", "Sales", "Adm-clerical", "Tech-support", "Machine-op-inspct", "Farming-fishing", "Transport-moving", "Handlers-cleaners", "Other-service"])
    hours_per_week = st.number_input("Work Hours Per Week", min_value=1, max_value=100, value=40)

with col4:
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=100000, value=0)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=150000)

st.markdown("---")

# Predict Button
if st.button("🚀 Predict Income Bracket", use_container_width=True):
    # Package inputs exactly matching the DataInput Pydantic model fields
    payload = {
        "age": int(age),
        "workclass": workclass,
        "fnlwgt": int(fnlwgt),
        "education": education,
        "educational_num": int(educational_num),
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital_gain": int(capital_gain),
        "capital_loss": int(capital_loss),
        "hours_per_week": int(hours_per_week),
        "native_country": native_country
    }
    
    with st.spinner("Talking to KNN model server..."):
        try:
            # Send data to FastAPI
            response = requests.post(FASTAPI_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                
                # Render clean outcome alerts depending on what model says
                if prediction == ">50K":
                    st.success(f"🎉 **Prediction Outcome:** High Income Bracket (**{prediction}**/year)")
                else:
                    st.info(f"📋 **Prediction Outcome:** Standard Income Bracket (**{prediction}**/year)")
            else:
                st.error(f"❌ Server error (Status Code: {response.status_code}). Check your FastAPI console output.")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to FastAPI! Ensure your terminal running uvicorn on port 8000 is still active.")