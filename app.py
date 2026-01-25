import logging
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_scoring_app")

from credit_scorer import CreditScorer

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

class CreditScoringApp:
    def __init__(self, train_path, test_path, model_type="random_forest"):
        self.train_path = train_path
        self.test_path = test_path
        self.model_type = model_type
        self.model = CreditScorer(model_type=model_type)
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.cm_train = None
        self.acc_train = None

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_path, low_memory=False)
            self.test_data = pd.read_csv(self.test_path, low_memory=False)

            logger.info(json.dumps({
                "event": "data_loaded",
                "train_rows": self.train_data.shape[0],
                "train_columns": self.train_data.shape[1],
                "test_rows": self.test_data.shape[0],
                "test_columns": self.test_data.shape[1],
                "timestamp": datetime.utcnow().isoformat()
            }))

        except Exception as e:
            log_error(
                event="data_load_failed",
                error=e,
                extra={"train_path": self.train_path, "test_path": self.test_path}
            )
            raise



    def train_model(self):
        try:
            self.model.train(self.train_data)

            self.X_train, _ = self.model.preprocess(self.train_data, fit=False)
            y_pred_train = self.model.predict(self.train_data)
            y_true_train = self.train_data["Credit_Score"]

            self.acc_train = accuracy_score(y_true_train, y_pred_train)

            logger.info(json.dumps({
                "event": "model_trained",
                "model_type": self.model_type,
                "train_accuracy": float(self.acc_train),
                "timestamp": datetime.utcnow().isoformat()
            }))

        except Exception as e:
            log_error(
                event="model_training_failed",
                error=e,
                extra={"model_type": self.model_type}
            )
            raise



    def plot_feature_importance(self):
        if hasattr(self.model.model, "feature_importances_"):
            importances = self.model.model.feature_importances_
            features = self.X_train.columns
            indices = importances.argsort()[-10:]

            plt.figure(figsize=(10, 5))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            logger.info(json.dumps({
                "event": "feature_importance",
                "top_features": dict(zip(
                    [self.X_train.columns[i] for i in indices],
                    importances[indices]
                )),
                "timestamp": datetime.utcnow().isoformat()
            }))


    def save_model(self, name="credit_model"):
        self.model.save(name)
        logger.info(json.dumps({
            "event": "model_saved",
            "model_name": name,
            "timestamp": datetime.utcnow().isoformat()
        }))


    def load_model(self, name="credit_model"):
        self.model.load(name)
        logger.info(json.dumps({
            "event": "model_loaded",
            "model_name": name,
            "timestamp": datetime.utcnow().isoformat()
        }))


    def evaluate_test_set(self):
        if "Credit_Score" in self.test_data.columns:
            X_test = self.test_data.drop(columns=["Credit_Score"])
            y_test = self.test_data["Credit_Score"]
        else:
            X_test = self.test_data
            y_test = None

        y_pred = self.model.predict(X_test)
        print("Predicciones realizadas.")

        if y_test is not None:
            labels = unique_labels(y_test, y_pred)
            cm_test = confusion_matrix(y_test, y_pred, labels=labels)
            acc_test = accuracy_score(y_test, y_pred)
            logger.info(json.dumps({
                "event": "model_evaluated",
                "test_accuracy": float(acc_test),
                "prediction_distribution": dict(pd.Series(y_pred).value_counts()),
                "timestamp": datetime.utcnow().isoformat()
            }))

        else:
            print("Test.csv no contiene la columna 'Credit_Score'; no se puede calcular accuracy.")

        # Guardar predicciones + probabilidades disponibles
        result = self.test_data.copy()
        result["Predicted_Credit_Score"] = y_pred

        try:
            proba = self.model.predict_proba(X_test)
            class_names = [f"Prob_{cls}" for cls in self.model.model.classes_]
            proba_df = pd.DataFrame(proba, columns=class_names)
            result = pd.concat([result.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        except:
            pass

        result.to_csv("predicciones.csv", index=False)
        print("📄 Predicciones guardadas en predicciones.csv")



if __name__ == "__main__":
    app = CreditScoringApp("train.csv", "test.csv", model_type="random_forest")
    app.load_data()
    app.train_model()
    app.save_model("credit_model")
    app.load_model("credit_model")
    app.evaluate_test_set()
