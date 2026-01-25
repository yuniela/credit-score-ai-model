
# 📊 Credit Score AI Library

Una librería modular de Inteligencia Artificial para la **clasificación del score crediticio** basada en modelos de Machine Learning supervisado. Diseñada para ser reutilizable, personalizable y portable mediante Docker.

---

## ✅ Características

- Clasificación multiclase del `Credit_Score` (`Poor`, `Standard`, `Good`)
- Modelos disponibles:
  - Regresión logística
  - Árboles de decisión
  - Random Forest
- Preprocesamiento completo:
  - Limpieza y normalización
  - Codificación de variables categóricas
  - Ingeniería de características
- Balanceo de clases con **SMOTE**
- Evaluación con métricas y visualizaciones:
  - Matriz de confusión
  - Curvas ROC
  - Importancia de características
- Exportación de predicciones con probabilidades
- Ejecutable en cualquier entorno vía **Docker**

---

## 📁 Estructura del Proyecto

```
credit-score-ai/
├── app.py                # Clase de aplicación principal
├── credit_scorer.py      # Clase con toda la lógica de IA
├── train.csv             # Dataset de entrenamiento
├── test.csv              # Dataset de prueba
├── predicciones.csv      # Archivo generado con predicciones
├── requirements.txt      # Dependencias del proyecto
└── Dockerfile            # Imagen Docker para ejecución
```

---

## ⚙️ Requisitos

- Python 3.8 o superior
- Librerías de Python:
  - `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`
  - `matplotlib`, `seaborn`, `joblib`

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Desde Python (modo local)

```python
from app import CreditScoringApp

app = CreditScoringApp("train.csv", "test.csv", model_type="random_forest")
app.load_data()
app.train_model()
app.save_model("credit_model")
app.load_model("credit_model")
app.evaluate_test_set()
```

## 🧪 Opción 1Crear y usar un entorno virtual en Python

### 1. Crear el entorno virtual

```bash
python -m venv venv
```
1. **Activar el entorno virtual**

En Windowns CMD
```bash
venv\Scripts\activate
```
En macOS/Linux
```bash
source venv/bin/activate
```
2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```
3. **Probar librería**
```bash
python ./app.py
```
4. **Desactivar entorno**
```bash
deactivate
```

## 🧪 Opción 2 Desde Docker

1. **Construir la imagen**

```bash
docker build -t credit-score-app .
```

2. **Ejecutar la aplicación**

```bash
docker run --rm -v $(pwd):/app credit-score-app
```

Esto generará `predicciones.csv` en tu carpeta local.

---

## 📤 Salidas del sistema

- **`predicciones.csv`**: incluye predicción y probabilidades por clase.
- **Métricas impresas** en consola: `accuracy`, `precision`, `recall`, `F1`.
- **Gráficos mostrados**:
  - Matriz de confusión (normalizada)
  - Top 10 features (para árboles)
  - ROC multiclase (si aplica)

---

## 🧠 Modelos soportados

Puedes cambiar el modelo en el parámetro `model_type`:

```python
model_type="logistic_regression"
model_type="decision_tree"
model_type="random_forest"
```

---

## 🧪 Dataset de ejemplo

Este proyecto utiliza el dataset `Credit Score Classification` disponible en Kaggle:  
https://www.kaggle.com/datasets/parisroshan/credit-score-classification

---

## 📌 Autor

Desarrollado por **Evelyn Solórzano Burgos** como parte de un proyecto de integración de IA y software modular.



