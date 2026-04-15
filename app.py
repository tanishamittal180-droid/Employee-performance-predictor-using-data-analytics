import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Employee Performance Predictor", layout="centered")

st.title("👩‍💼 Employee Performance Predictor")
st.write("Predict whether an employee is a High, Medium, or Low performer")

# -------------------------------
# LOAD OR CREATE MODEL
# -------------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)

    n = 500

    data = pd.DataFrame({
        'age': np.random.randint(22, 60, n),
        'experience': np.random.randint(1, 20, n),
        'department': np.random.choice(['HR', 'IT', 'Sales'], n),
        'salary': np.random.randint(20000, 100000, n),
        'projects': np.random.randint(1, 10, n),
        'training_hours': np.random.randint(0, 50, n),
        'attendance': np.random.uniform(0.7, 1.0, n),
        'feedback_score': np.random.uniform(1, 5, n)
    })

    # Create target
    conditions = [
        (data['projects'] > 6) & (data['feedback_score'] > 4),
        (data['projects'] > 3),
    ]
    choices = ['High', 'Medium']

    data['performance'] = np.select(conditions, choices, default='Low')

    # Encode
    le_dept = LabelEncoder()
    le_perf = LabelEncoder()

    data['department'] = le_dept.fit_transform(data['department'])
    data['performance'] = le_perf.fit_transform(data['performance'])

    X = data.drop('performance', axis=1)
    y = data['performance']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le_dept, le_perf


model, le_dept, le_perf = train_model()

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Enter Employee Details")

age = st.slider("Age", 22, 60, 30)
experience = st.slider("Experience (years)", 1, 20, 5)
department = st.selectbox("Department", ["HR", "IT", "Sales"])
salary = st.slider("Salary", 20000, 100000, 40000)
projects = st.slider("Projects Completed", 1, 10, 3)
training_hours = st.slider("Training Hours", 0, 50, 10)
attendance = st.slider("Attendance Rate", 0.7, 1.0, 0.9)
feedback_score = st.slider("Feedback Score", 1.0, 5.0, 3.0)

# Encode department
dept_encoded = le_dept.transform([department])[0]

# Create dataframe
input_data = pd.DataFrame({
    'age': [age],
    'experience': [experience],
    'department': [dept_encoded],
    'salary': [salary],
    'projects': [projects],
    'training_hours': [training_hours],
    'attendance': [attendance],
    'feedback_score': [feedback_score]
})

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Performance"):
    prediction = model.predict(input_data)
    result = le_perf.inverse_transform(prediction)[0]

    st.subheader(f"📊 Predicted Performance: {result}")

    # Insight messages
    if result == "High":
        st.success("Excellent employee! Consider promotion 🚀")
    elif result == "Medium":
        st.warning("Average performance. Needs improvement 📈")
    else:
        st.error("Low performance. Training required ⚠️")