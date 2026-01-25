import logging
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_scorer")

def log_error(event, error, extra=None):
        payload = {
            "event": event,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        if extra:
            payload.update(extra)
        logger.error(json.dumps(payload))

class CreditScorer:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.fill_medians = {}
        self.target_column = 'Credit_Score'
        self.model_registry = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced"
            )
        }

    def convert_credit_age(self, age_str):
        try:
            if pd.isnull(age_str):
                return np.nan
            parts = age_str.strip().split()
            return int(parts[0]) * 12 + int(parts[2])
        except:
            return np.nan

    def preprocess(self, data, fit=False):
        start_time = time.time()
        try:
            data = data.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Month'], errors='ignore')
            y = data[self.target_column] if self.target_column in data.columns else None
            X = data.drop(columns=[self.target_column], errors='ignore')

            if 'Credit_History_Age' in X.columns:
                X['Credit_History_Age'] = X['Credit_History_Age'].apply(self.convert_credit_age)

            known_num_cols = [
                "Monthly_Inhand_Salary", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
                "Amount_invested_monthly", "Monthly_Balance", "Age", "Annual_Income",
                "Num_of_Loan", "Changed_Credit_Limit", "Outstanding_Debt",
                "Credit_History_Age", "Num_Credit_Card"
            ]
            for col in known_num_cols:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')

            # Feature engineering
            if "Outstanding_Debt" in X.columns and "Annual_Income" in X.columns:
                X["Debt_to_Income_Ratio"] = X["Outstanding_Debt"] / (X["Annual_Income"] + 1)
            if "Changed_Credit_Limit" in X.columns and "Outstanding_Debt" in X.columns:
                X["Credit_Usage"] = X["Changed_Credit_Limit"] / (X["Outstanding_Debt"] + 1)
            if "Outstanding_Debt" in X.columns and "Num_Credit_Card" in X.columns:
                X["Avg_Debt_per_Card"] = X["Outstanding_Debt"] / (X["Num_Credit_Card"] + 1)

            numerical_cols = [col for col in X.columns if X[col].dtype != 'object']
            categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

            for col in numerical_cols:
                if fit:
                    mean_value = X[col].mean()
                    if pd.isna(mean_value):
                        mean_value = 0
                    self.fill_medians[col] = mean_value
                X[col] = X[col].fillna(self.fill_medians.get(col, 0))

            for col in categorical_cols:
                if fit:
                    mode_value = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                    self.label_encoders[f"{col}_fill"] = mode_value
                mode_value = self.label_encoders.get(f"{col}_fill", "Unknown")
                X[col] = X[col].fillna(mode_value).astype(str)

            for col in categorical_cols:
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        try:
                            X[col] = le.transform(X[col])
                        except ValueError:
                            X[col] = X[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                            X[col] = le.transform(X[col])

            if fit:
                self.scaler = StandardScaler()
                X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            else:
                X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            
            latency = (time.time() - start_time) * 1000

            logger.info(json.dumps({
                "event": "data_preprocessed",
                "fit": fit,
                "num_rows": X.shape[0],
                "num_features": X.shape[1],
                "latency_ms": float(latency),
                "timestamp": datetime.utcnow().isoformat()
            }))

            return X, y
        except Exception as e:
            log_error(
                event="data_preprocess_failed",
                error=e,
                extra={"fit": fit}
            )
            raise

    def train(self, data):
        start_time = time.time()
        try:
            X, y = self.preprocess(data, fit=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

            nan_rows = X_train.isnull().any(axis=1)
            if nan_rows.sum() > 0:
                X_train = X_train[~nan_rows]
                y_train = y_train[~nan_rows]
            if X_train.shape[0] == 0:
                raise ValueError("Todos los datos de entrenamiento fueron eliminados por NaNs.")

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            if self.model_type in ["random_forest", "logistic_regression", "decision_tree"]:
                classes = np.unique(y_train)
                weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
                class_weights = dict(zip(classes, weights))
                if self.model_type == "random_forest":
                    self.model = RandomForestClassifier(
                        n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=5,
                        class_weight="balanced", random_state=42
                    )
                elif self.model_type == "logistic_regression":
                    self.model = LogisticRegression(max_iter=1000, class_weight=class_weights)
                elif self.model_type == "decision_tree":
                    self.model = DecisionTreeClassifier(class_weight=class_weights)
            else:
                self.model = self.model_registry[self.model_type]

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            latency = (time.time() - start_time) * 1000

            logger.info(json.dumps({
                "event": "model_trained",
                "model_type": self.model_type,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": float(acc),
                "latency_ms": latency,
                "timestamp": datetime.utcnow().isoformat()
            }))

            return self.model
        except Exception as e:
            log_error(
                event="model_training_failed",
                error=e,
                extra={"model_type": self.model_type}
            )
            raise

    def predict(self, data):
        start_time = time.time()
        try:
            X, _ = self.preprocess(data, fit=False)
            preds = self.model.predict(X)
            prediction_dist = {
                str(k): int(v)
                for k, v in pd.Series(preds).value_counts().items()
            }


            latency = (time.time() - start_time) * 1000

            logger.info(json.dumps({
                "event": "model_inference",
                "model_type": self.model_type,
                "num_records": len(X),
                "prediction_distribution": prediction_dist,
                "latency_ms": latency,
                "timestamp": datetime.utcnow().isoformat()
            }))
            return preds
        except Exception as e:
            log_error(
                event="model_inference_failed",
                error=e,
                extra={"num_records": len(data)}
            )
            raise

    def predict_proba(self, data):
        start_time = time.time()
        try:
            X, _ = self.preprocess(data, fit=False)

            if not hasattr(self.model, "predict_proba"):
                raise ValueError("Modelo sin soporte predict_proba")

            probs = self.model.predict_proba(X)
            max_conf = probs.max(axis=1)

            latency = (time.time() - start_time) * 1000

            logger.info(json.dumps({
                "event": "model_confidence",
                "avg_confidence": float(np.mean(max_conf)),
                "min_confidence": float(np.min(max_conf)),
                "max_confidence": float(np.max(max_conf)),
                "latency_ms": latency,
                "timestamp": datetime.utcnow().isoformat()
            }))

            return probs

        except Exception as e:
            log_error(
                event="predict_proba_failed",
                error=e
            )
            raise


    def save(self, filename):
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.label_encoders,
                'fill_medians': self.fill_medians
            }, f"{filename}.joblib")
            logger.info(json.dumps({
                "event": "model_saved",
                "model_name": filename,
                "timestamp": datetime.utcnow().isoformat()
            }))
        except Exception as e:
            log_error("model_save_failed", e)
            raise

    def load(self, filename):
        try:
            data = joblib.load(f"{filename}.joblib")
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoders = data['encoders']
            self.fill_medians = data['fill_medians']
            logger.info(json.dumps({
                "event": "model_loaded",
                "model_name": filename,
                "timestamp": datetime.utcnow().isoformat()
            }))
        except Exception as e:
            log_error("model_load_failed", e)
            raise




