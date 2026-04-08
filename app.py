import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Global variables
model = None
label_encoders = {}
feature_columns = None

# ------------------------------------------------------------
# Data Preprocessing & Feature Engineering
# ------------------------------------------------------------
def engineer_features(df):
    """Create age groups and other derived features."""
    bins = [16, 25, 35, 45, 55, 100]
    labels = ['16-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

def encode_categorical(df, fit=False):
    """Label encode categorical columns. If fit=True, store encoders."""
    categorical_cols = ['gender', 'test_station', 'vehicle_type', 'licence_type', 'test_manoeuvre', 'age_group']
    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le:
                # Handle unseen labels: set to most frequent or 0
                df[col + '_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df[col + '_encoded'] = 0
    return df

def prepare_features(input_dict):
    """Convert raw input dict to feature vector for prediction."""
    df = pd.DataFrame([input_dict])
    df = engineer_features(df)
    df = encode_categorical(df, fit=False)
    # Ensure all expected feature columns exist
    expected = ['age', 'gender_encoded', 'test_station_encoded', 'vehicle_type_encoded',
                'licence_type_encoded', 'test_manoeuvre_encoded', 'training_hours',
                'attempt_number', 'age_group_encoded']
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    return df[expected].values[0]

# ------------------------------------------------------------
# Model Training (if model.pkl not found)
# ------------------------------------------------------------
def train_model(csv_path=None):
    """Train a Random Forest model. If csv_path is None, generate synthetic data."""
    global model, feature_columns, label_encoders

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Generate synthetic dataset similar to VID structure
        np.random.seed(42)
        n = 2000
        data = {
            'age': np.random.randint(18, 65, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'test_station': np.random.choice(['Harare', 'Bulawayo', 'Mutare', 'Gweru'], n),
            'vehicle_type': np.random.choice(['Small car', 'Light truck', 'Heavy vehicle'], n),
            'licence_type': np.random.choice(['Class 1', 'Class 2', 'Class 3', 'Class 4'], n),
            'test_manoeuvre': np.random.choice(['Parking', 'Three-point turn', 'Highway merge', 'Emergency stop'], n),
            'training_hours': np.random.randint(0, 60, n),
            'attempt_number': np.random.choice([1, 2, 3, 4], n, p=[0.6, 0.25, 0.1, 0.05]),
        }
        df = pd.DataFrame(data)
        # Create synthetic target: pass if training_hours > 20 and attempt_number == 1 etc.
        df['result'] = ((df['training_hours'] > 20) & (df['attempt_number'] == 1) |
                        (df['training_hours'] > 35) | (df['age'] < 30)).astype(int)
        # Add some noise
        noise = np.random.choice([0, 1], n, p=[0.85, 0.15])
        df['result'] = df['result'] ^ noise  # flip 15% of outcomes

    # Preprocess
    df = engineer_features(df)
    df = encode_categorical(df, fit=True)

    # Define features and target
    feature_cols = ['age', 'gender_encoded', 'test_station_encoded', 'vehicle_type_encoded',
                    'licence_type_encoded', 'test_manoeuvre_encoded', 'training_hours',
                    'attempt_number', 'age_group_encoded']
    X = df[feature_cols]
    y = df['result']
    feature_columns = feature_cols

    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate (optional print)
    acc = rf.score(X_test, y_test)
    print(f"Model trained. Accuracy: {acc:.3f}")

    # Save model and encoders
    joblib.dump({'model': rf, 'encoders': label_encoders, 'feature_columns': feature_columns},
                'model.pkl')
    return rf

# ------------------------------------------------------------
# Load or train model on startup
# ------------------------------------------------------------
def load_or_train_model():
    global model, label_encoders, feature_columns
    if os.path.exists('model.pkl'):
        data = joblib.load('model.pkl')
        model = data['model']
        label_encoders = data['encoders']
        feature_columns = data['feature_columns']
        print("Model loaded from model.pkl")
    else:
        print("model.pkl not found. Training new model...")
        model = train_model()  # train on synthetic data
        # optionally try to load a real CSV: train_model('data/vid_records.csv')

load_or_train_model()

# ------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Expects JSON with candidate data. Returns prediction and confidence."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    required_fields = ['age', 'gender', 'test_station', 'vehicle_type', 'licence_type',
                       'test_manoeuvre', 'training_hours', 'attempt_number']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        features = prepare_features(data)
        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        confidence = max(proba)
        result = 'PASS' if pred == 1 else 'FAIL'
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 3),
            'probabilities': {'fail': proba[0], 'pass': proba[1]}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional: retrain endpoint
@app.route('/retrain', methods=['POST'])
def retrain():
    """Upload a CSV file to retrain the model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        # Save temporarily
        temp_path = 'temp_upload.csv'
        file.save(temp_path)
        global model, label_encoders, feature_columns
        model = train_model(temp_path)
        os.remove(temp_path)
        return jsonify({'message': 'Model retrained successfully', 'accuracy': model.score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
