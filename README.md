Clinical Risk Audit Module

An interactive Streamlit application for assessing diabetes risk based on patient clinical data. Combines machine learning predictions, feature auditing, and monotonicity stress tests to provide a transparent and reliable risk assessment tool.

Features

Patient Risk Assessment
Input clinical data such as gender, age, BMI, HbA1c, blood glucose, hypertension, and heart disease history to get a predicted probability of diabetes. Clinical alerts highlight high-risk indicators like elevated HbA1c.

Model Audit & Transparency

Permutation-based feature importance to understand which factors most influence predictions.

Calibration curves to visualize the reliability of predicted probabilities.

Monotonicity Stress Test
Test model behavior under controlled scenarios (e.g., increasing glucose) to verify that predicted risk rises logically with key clinical features.

Robust Machine Learning Engine

Uses a calibrated HistGradientBoostingClassifier with monotonic constraints.

Features scaling and proper feature ordering for consistent predictions.

Cached model and scaler for fast app startup.

Technology Stack

Python 3.x

Streamlit for interactive web interface

scikit-learn for machine learning and calibration

pandas & numpy for data handling

matplotlib for plotting and visualization
