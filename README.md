# 🩺 Clinical Risk Audit Module

**An interactive Streamlit app to assess diabetes risk using patient clinical data.**  
It combines **machine learning predictions**, **model auditing**, and **monotonicity stress tests** to deliver **transparent, reliable, and interpretable risk assessments**.  

---

## 🌟 Features

### 1️⃣ Patient Risk Assessment
- Enter clinical data: **gender, age, BMI, HbA1c, blood glucose, hypertension, heart disease history**.  
- Get a **predicted probability of diabetes**.  
- **Clinical alerts** highlight high-risk indicators (e.g., HbA1c ≥ 6.5%).  

### 2️⃣ Model Audit & Transparency
- **Permutation-based feature importance** shows which clinical factors most influence predictions.  
- **Calibration curves** visualize model reliability and probability accuracy.  

### 3️⃣ Monotonicity Stress Test
- Test model behavior under controlled scenarios (e.g., increasing glucose).  
- Ensures predicted risk **rises logically** with key clinical features.  

### 4️⃣ Robust Machine Learning Engine
- Calibrated **HistGradientBoostingClassifier** with **monotonic constraints**.  
- Proper **feature scaling** and **feature ordering** for consistent predictions.  
- Cached model and scaler for **fast app startup**.  

---

## 🛠 Technology Stack
- **Python 3.x** – core language  
- **Streamlit** – interactive web interface  
- **scikit-learn** – ML models, calibration, and auditing  
- **pandas & numpy** – data handling and manipulation  
- **matplotlib** – plotting and visualization  
