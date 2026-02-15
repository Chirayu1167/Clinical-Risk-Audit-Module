import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance

DATA_FILENAME = "diabetes_dataset.csv"
MODEL_FILENAME = "calibrated_model.joblib"
SCALER_FILENAME = "scaler.joblib"
FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'hbA1c_level', 'blood_glucose_level']

def train_and_save_model():
    if not os.path.exists(DATA_FILENAME):
        st.error(f"File {DATA_FILENAME} not found.")
        return None, None
    
    df = pd.read_csv(DATA_FILENAME)
    df.columns = df.columns.str.strip()
    
    X = df[FEATURES].copy()
    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])
    y = df['diabetes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    constraints = [0, 1, 0, 0, 1, 1, 1]

    base_model = HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=50,
        max_leaf_nodes=15,
        monotonic_cst=constraints
    )

    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_scaled, y)
    
    joblib.dump(calibrated_model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    return calibrated_model, scaler

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
    else:
        model, scaler = train_and_save_model()
    
    df = pd.read_csv(DATA_FILENAME)
    df.columns = df.columns.str.strip()
    X_raw = df[FEATURES].copy()
    X_raw['gender'] = LabelEncoder().fit_transform(X_raw['gender'])
    X_train_scaled = scaler.transform(X_raw)
    y_train = df['diabetes']
    
    return model, scaler, X_train_scaled, y_train

MODEL, SCALER, X_TRAIN, Y_TRAIN = load_assets()

def audit_model():
    st.subheader("Model Technical Audit")
    
    with st.spinner("Analyzing features..."):
        result = permutation_importance(MODEL, X_TRAIN, Y_TRAIN, n_repeats=5, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': result.importances_mean
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

    st.write("Calibration Curve (Reliability Diagram):")
    prob_pos = MODEL.predict_proba(X_TRAIN)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(Y_TRAIN, prob_pos, n_bins=10)
    
    
    
    fig, ax = plt.subplots()
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_ylabel("Actual Fraction")
    ax.set_xlabel("Predicted Probability")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

st.title("Clinical Risk Audit Module")

tab1, tab2 = st.tabs(["Assessment", "Audit"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("risk_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 1, 120, 55)
            bmi = st.number_input("BMI", 10.0, 60.0, 29.8)
            hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 6.6)
            glucose = st.number_input("Glucose (mg/dL)", 50, 400, 155)
            hyper = st.checkbox("Hypertension")
            heart = st.checkbox("Heart Disease History", value=True)
            submit = st.form_submit_button("Run Analysis")

    if submit:
        input_raw = pd.DataFrame([[
            1 if gender == "Male" else 0, age, 1 if hyper else 0,
            1 if heart else 0, bmi, hba1c, glucose
        ]], columns=FEATURES)
        
        input_scaled = SCALER.transform(input_raw)
        prob = MODEL.predict_proba(input_scaled)[0][1]
        
        with col2:
            st.metric("Diabetes Probability", f"{prob:.1%}")
            if hba1c >= 6.5:
                st.error("Clinical Alert: HbA1c is in the diabetic range.")

with tab2:
    if st.button("Generate Audit Plots"):
        audit_model()

st.sidebar.markdown("### Reliability Stress Test")
if st.sidebar.button("Test Glucose Monotonicity"):
    test_range = np.linspace(70, 300, 20)
    test_data = pd.DataFrame([[0, 40, 0, 0, 25.0, 5.5, g] for g in test_range], columns=FEATURES)
    test_scaled = SCALER.transform(test_data)
    test_probs = MODEL.predict_proba(test_scaled)[:, 1]
    
    fig2, ax2 = plt.subplots()
    ax2.plot(test_range, test_probs, marker='o')
    ax2.set_xlabel("Glucose")
    ax2.set_ylabel("Risk Score")
    st.sidebar.pyplot(fig2, clear_figure=True)
