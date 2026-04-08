# VID DRIVER PULSE – Predictive Analytics for Driver Test Outcomes

A minimalistic machine learning system that predicts whether a driver candidate will **PASS** or **FAIL** their driving test, using a Random Forest classifier. Built with Flask (backend) and vanilla HTML/CSS/JS (frontend). Developed for the Vehicle Inspectorate Department (VID), Zimbabwe.

## Features
- **Predict** pass/fail outcome based on candidate data (age, gender, test station, vehicle type, licence type, manoeuvre, training hours, attempt number).
- **Confidence score** for each prediction.
- **Interactive web interface** with real-time form submission and result display.
- **Model training** script (`model.py`) that generates a `model.pkl` file (synthetic data included, or use your own CSV).
- **REST API** endpoint `/predict` for programmatic access.
