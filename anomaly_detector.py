import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Fraud detection using Isolation Forest"""
    
    def __init__(self, model_path: str = "models/isolation_forest.pkl"):
        """
        Initialize the anomaly detector
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.scaler_path = "models/scaler.pkl"
        self.model = None
        self.scaler = None
        self.feature_columns = ['amount', 'day_of_month', 'month', 'year', 'vendor_length']
        self.fraud_threshold = -0.1  # Threshold for fraud detection
        
        # Load existing model if available
        self._load_model()
        
        # If no model exists, create a default one
        if self.model is None:
            self._create_default_model()
    
    def _load_model(self) -> bool:
        """Load saved model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Model and scaler loaded successfully")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False
    
    def _save_model(self):
        """Save model and scaler to disk"""
        try:
            os.makedirs("models", exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("Model and scaler saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _create_default_model(self):
        """Create a default model with synthetic data"""
        print("Creating default model with synthetic data...")
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Normal invoices
        normal_amounts = np.random.lognormal(mean=8, sigma=1, size=int(n_samples * 0.9))
        normal_days = np.random.randint(1, 29, size=int(n_samples * 0.9))
        normal_months = np.random.randint(1, 13, size=int(n_samples * 0.9))
        normal_years = np.random.choice([2022, 2023, 2024], size=int(n_samples * 0.9))
        normal_vendor_lengths = np.random.normal(15, 5, size=int(n_samples * 0.9))
        
        # Anomalous invoices (potential fraud)
        anomaly_amounts = np.concatenate([
            np.random.uniform(50000, 200000, size=int(n_samples * 0.05)),  # Very high amounts
            np.random.uniform(0.1, 10, size=int(n_samples * 0.05))  # Very low amounts
        ])
        anomaly_days = np.random.randint(1, 29, size=int(n_samples * 0.1))
        anomaly_months = np.random.randint(1, 13, size=int(n_samples * 0.1))
        anomaly_years = np.random.choice([2022, 2023, 2024], size=int(n_samples * 0.1))
        anomaly_vendor_lengths = np.concatenate([
            np.random.uniform(1, 5, size=int(n_samples * 0.05)),  # Very short names
            np.random.uniform(40, 80, size=int(n_samples * 0.05))  # Very long names
        ])
        
        # Combine data
        amounts = np.concatenate([normal_amounts, anomaly_amounts])
        days = np.concatenate([normal_days, anomaly_days])
        months = np.concatenate([normal_months, anomaly_months])
        years = np.concatenate([normal_years, anomaly_years])
        vendor_lengths = np.concatenate([normal_vendor_lengths, anomaly_vendor_lengths])
        
        # Create DataFrame
        df = pd.DataFrame({
            'amount': amounts,
            'day_of_month': days,
            'month': months,
            'year': years,
            'vendor_length': np.abs(vendor_lengths)
        })
        
        # Train model
        self._train_isolation_forest(df)
        self._save_model()
    
    def predict_fraud(self, invoice_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict if an invoice is fraudulent
        
        Args:
            invoice_fields: Dictionary containing invoice fields
            
        Returns:
            Dictionary with anomaly score and fraud flag
        """
        try:
            # Extract features
            features = self._extract_features(invoice_fields)
            
            if features is None:
                return {
                    "anomaly_score": 0.0,
                    "is_fraud": False,
                    "confidence": "low",
                    "reason": "Insufficient data for analysis"
                }
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict anomaly score
            anomaly_score = self.model.decision_function(features_scaled)[0]
            
            # Determine if fraud
            is_fraud = anomaly_score < self.fraud_threshold
            
            # Determine confidence level
            confidence = self._get_confidence_level(anomaly_score)
            
            # Get reason for fraud detection
            reason = self._get_fraud_reason(invoice_fields, anomaly_score)
            
            return {
                "anomaly_score": round(anomaly_score, 3),
                "is_fraud": is_fraud,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "anomaly_score": 0.0,
                "is_fraud": False,
                "confidence": "low",
                "reason": f"Prediction error: {str(e)}"
            }
    
    def _extract_features(self, invoice_fields: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from invoice fields"""
        try:
            # Amount feature
            amount = invoice_fields.get('amount')
            if amount is None:
                return None
            
            # Date features
            date_str = invoice_fields.get('date')
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    day_of_month = date_obj.day
                    month = date_obj.month
                    year = date_obj.year
                except:
                    # Use current date as fallback
                    now = datetime.now()
                    day_of_month = now.day
                    month = now.month
                    year = now.year
            else:
                # Use current date as fallback
                now = datetime.now()
                day_of_month = now.day
                month = now.month
                year = now.year
            
            # Vendor name length feature
            vendor = invoice_fields.get('vendor', '')
            vendor_length = len(vendor) if vendor else 0
            
            features = [
                float(amount),
                float(day_of_month),
                float(month),
                float(year),
                float(vendor_length)
            ]
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def _get_confidence_level(self, anomaly_score: float) -> str:
        """Determine confidence level based on anomaly score"""
        if anomaly_score < -0.3:
            return "high"
        elif anomaly_score < -0.1:
            return "medium"
        elif anomaly_score < 0.1:
            return "low"
        else:
            return "very_low"
    
    def _get_fraud_reason(self, invoice_fields: Dict[str, Any], anomaly_score: float) -> str:
        """Generate reason for fraud detection"""
        reasons = []
        
        amount = invoice_fields.get('amount', 0)
        vendor = invoice_fields.get('vendor', '')
        
        if amount and amount > 50000:
            reasons.append("unusually high amount")
        elif amount and amount < 10:
            reasons.append("unusually low amount")
        
        if len(vendor) < 3:
            reasons.append("suspiciously short vendor name")
        elif len(vendor) > 50:
            reasons.append("unusually long vendor name")
        
        if anomaly_score < self.fraud_threshold:
            if not reasons:
                reasons.append("pattern deviates significantly from normal invoices")
        else:
            return "Normal invoice pattern detected"
        
        return "Potential fraud indicators: " + ", ".join(reasons) if reasons else "Normal invoice"
    
    def train_model(self, df: pd.DataFrame) -> str:
        """
        Train the fraud detection model with new data
        
        Args:
            df: DataFrame with training data
            
        Returns:
            Training status message
        """
        try:
            # Validate required columns
            required_columns = ['amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Prepare features
            df = self._prepare_training_data(df)
            
            # Train model
            self._train_isolation_forest(df)
            
            # Save model
            self._save_model()
            
            return f"Model trained successfully with {len(df)} samples"
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data by extracting features"""
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Ensure amount column exists
        if 'amount' not in df_processed.columns:
            raise ValueError("Amount column is required")
        
        # Extract date features if date column exists
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
            df_processed['day_of_month'] = df_processed['date'].dt.day
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['year'] = df_processed['date'].dt.year
        else:
            # Use current date as default
            now = datetime.now()
            df_processed['day_of_month'] = now.day
            df_processed['month'] = now.month
            df_processed['year'] = now.year
        
        # Vendor length feature
        if 'vendor' in df_processed.columns:
            df_processed['vendor_length'] = df_processed['vendor'].str.len().fillna(0)
        else:
            df_processed['vendor_length'] = 10  # Default length
        
        # Select only feature columns
        feature_df = df_processed[self.feature_columns].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        return feature_df
    
    def _train_isolation_forest(self, df: pd.DataFrame):
        """Train the Isolation Forest model"""
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(df)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        self.model.fit(features_scaled)
        
        print(f"Model trained with {len(df)} samples")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_type": "Isolation Forest",
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "feature_columns": self.feature_columns,
            "fraud_threshold": self.fraud_threshold
        }
        
        if self.model is not None:
            info.update({
                "n_estimators": getattr(self.model, 'n_estimators', 'Unknown'),
                "contamination": getattr(self.model, 'contamination', 'Unknown'),
                "model_path": self.model_path
            })
        
        return info
