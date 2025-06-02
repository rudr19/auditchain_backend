"""
Script to create and save the initial Isolation Forest model
Run this once to create the initial model file
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

def create_initial_model():
    """Create initial model with synthetic data"""
    print("Creating initial Isolation Forest model...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal invoices (90%)
    normal_data = {
        'amount': np.random.lognormal(mean=8, sigma=1, size=int(n_samples * 0.9)),
        'day_of_month': np.random.randint(1, 29, size=int(n_samples * 0.9)),
        'month': np.random.randint(1, 13, size=int(n_samples * 0.9)),
        'year': np.random.choice([2022, 2023, 2024], size=int(n_samples * 0.9)),
        'vendor_length': np.abs(np.random.normal(15, 5, size=int(n_samples * 0.9)))
    }
    
    # Anomalous invoices (10% - potential fraud)
    anomaly_data = {
        'amount': np.concatenate([
            np.random.uniform(50000, 200000, size=int(n_samples * 0.05)),  # Very high amounts
            np.random.uniform(0.1, 10, size=int(n_samples * 0.05))  # Very low amounts
        ]),
        'day_of_month': np.random.randint(1, 29, size=int(n_samples * 0.1)),
        'month': np.random.randint(1, 13, size=int(n_samples * 0.1)),
        'year': np.random.choice([2022, 2023, 2024], size=int(n_samples * 0.1)),
        'vendor_length': np.concatenate([
            np.random.uniform(1, 5, size=int(n_samples * 0.05)),  # Very short names
            np.random.uniform(40, 80, size=int(n_samples * 0.05))  # Very long names
        ])
    }
    
    # Combine all data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    model.fit(features_scaled)
    
    # Save model and scaler
    with open("models/isolation_forest.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open("models/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model created and saved successfully!")
    print(f"Training data size: {len(df)} samples")
    print(f"Features: {list(df.columns)}")
    
    # Test the model
    test_features = [[5000, 15, 6, 2024, 12]]  # Normal invoice
    test_features_scaled = scaler.transform(test_features)
    score = model.decision_function(test_features_scaled)[0]
    
    print(f"Test prediction score: {score:.3f}")
    print(f"Is anomaly: {score < -0.1}")

if __name__ == "__main__":
    create_initial_model()
