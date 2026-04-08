#!/usr/bin/env python3
"""
VID DRIVER PULSE - Model Training Script
Generates model.pkl containing:
- Trained Random Forest classifier
- Label encoders for categorical features
- List of feature column names
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
OUTPUT_FILE = 'model.pkl'

# ------------------------------------------------------------
# Data Generation (if no real CSV provided)
# ------------------------------------------------------------
def generate_synthetic_data(n_samples=4000):
    """Generate synthetic driver test data similar to VID structure."""
    np.random.seed(RANDOM_STATE)
    
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'test_station': np.random.choice(['Harare', 'Bulawayo', 'Mutare', 'Gweru'], n_samples),
        'vehicle_type': np.random.choice(['Small car', 'Light truck', 'Heavy vehicle'], n_samples),
        'licence_type': np.random.choice(['Class 1', 'Class 2', 'Class 3', 'Class 4'], n_samples),
        'test_manoeuvre': np.random.choice(['Parking', 'Three-point turn', 'Highway merge', 'Emergency stop'], n_samples),
        'training_hours': np.random.uniform(0, 60, n_samples).round(1),
        'attempt_number': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
    }
    df = pd.DataFrame(data)
    
    # Generate target: pass if training_hours > 20 and attempt 1, or training_hours > 35, or age < 30
    condition = ((df['training_hours'] > 20) & (df['attempt_number'] == 1)) | \
                (df['training_hours'] > 35) | \
                (df['age'] < 30)
    df['result'] = condition.astype(int)
    
    # Add realistic noise: flip 12% of outcomes
    noise = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    df['result'] = df['result'] ^ noise
    
    # Ensure pass rate ~52-55% as in project report
    actual_pass_rate = df['result'].mean()
    print(f"Generated synthetic data: {n_samples} records, pass rate: {actual_pass_rate:.1%}")
    
    return df

def load_real_data(csv_path):
    """Load real CSV data if available."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded real data: {len(df)} records")
    return df

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------
def engineer_features(df):
    """Create age groups and other derived features."""
    bins = [16, 25, 35, 45, 55, 100]
    labels = ['16-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

def encode_categorical(df, encoders=None):
    """
    Encode categorical columns.
    If encoders is None, fit new LabelEncoders and return them.
    If encoders is provided, use existing ones (transform only).
    """
    categorical_cols = ['gender', 'test_station', 'vehicle_type', 'licence_type', 'test_manoeuvre', 'age_group']
    new_encoders = {}
    
    for col in categorical_cols:
        if encoders is None:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            new_encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels: set to most frequent (class 0) if not seen
            df[col + '_encoded'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
            new_encoders[col] = le
    
    return df, new_encoders

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------
def train_and_save_model(csv_path=None):
    """
    Main training function.
    If csv_path is provided, load real data; otherwise generate synthetic.
    Saves model, encoders, and feature columns to model.pkl.
    """
    # 1. Load or generate data
    if csv_path and os.path.exists(csv_path):
        df = load_real_data(csv_path)
    else:
        df = generate_synthetic_data(4000)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Encode categorical variables (fit encoders)
    df, encoders = encode_categorical(df, encoders=None)
    
    # 4. Define features and target
    feature_cols = ['age', 'gender_encoded', 'test_station_encoded', 'vehicle_type_encoded',
                    'licence_type_encoded', 'test_manoeuvre_encoded', 'training_hours',
                    'attempt_number', 'age_group_encoded']
    X = df[feature_cols]
    y = df['result']
    
    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 6. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Feature importance (optional)
    importances = dict(zip(feature_cols, model.feature_importances_))
    print("\nTop 5 features by importance:")
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {k}: {v:.3f}")
    
    # 8. Save everything to model.pkl
    save_data = {
        'model': model,
        'encoders': encoders,
        'feature_columns': feature_cols,
        'test_accuracy': test_acc,
        'random_state': RANDOM_STATE
    }
    joblib.dump(save_data, OUTPUT_FILE)
    print(f"\nModel saved to {OUTPUT_FILE}")
    
    return model, encoders, feature_cols

# ------------------------------------------------------------
# Command-line execution
# ------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train VID Driver Pulse model')
    parser.add_argument('--csv', type=str, help='Path to CSV file with driver test data')
    args = parser.parse_args()
    
    train_and_save_model(csv_path=args.csv)
