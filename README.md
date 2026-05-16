# 🔧 Equipment Failure & Maintenance Cost Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue?logo=jupyter)](./maintenance_cost_prediction.ipynb)

A production-ready **machine learning solution** for predicting equipment maintenance costs with **99.7% accuracy**. This project demonstrates industrial data analysis, feature engineering, and ML model deployment in industrial maintenance systems.

## ⭐ Key Features

- **🎯 High Accuracy**: 99.7% R² Score (Lasso Regression)
- **📊 Complete Pipeline**: From raw data to predictions
- **📈 Interactive Notebook**: Jupyter notebook with full workflow
- **🔄 Feature Engineering**: 17 advanced features from raw data
- **📉 Model Comparison**: 5 different algorithms evaluated
- **📱 Easy to Use**: Simple prediction interface
- **🎨 Visualizations**: Performance plots and analysis charts
- **⚡ Production Ready**: Serialized models and preprocessors
- **🚀 FastAPI Integration**: REST API with MongoDB auto-save
- **📦 Automatic Processing**: Background tasks for non-blocking predictions

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [FastAPI REST API](#-fastapi-rest-api---new)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features](#-features)
- [Examples](#-examples)

---

## 🚀 Quick Start

### 1. Clone & Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/BellilxDhaker/equipment-failure-predictor.git
cd equipment-failure-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook maintenance_cost_prediction.ipynb
```

### 2. Or Use Pre-Trained Model

```python
import joblib

# Load trained model
model = joblib.load('models/best_maintenance_cost_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Make predictions (see examples below)
```

---

## 🚀 FastAPI REST API

**Production-ready API with automatic MongoDB integration and non-blocking predictions!**

### Quick API Setup

```bash
# 1. Configure MongoDB in .env file
# Edit .env and set your MongoDB connection:
#   MONGO_URI=mongodb://localhost:27017/maintenance_db
#   MONGO_DB_NAME=maintenance_db

# 2. Run the API
run_api.bat          # Windows
./run_api.sh         # Linux/Mac

# 3. Access API Documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Key Features

- ✅ **Automatic MongoDB Integration** - Data retrieval and auto-save
- ✅ **Non-Blocking Predictions** - Background tasks for batch processing
- ✅ **Batch Processing** - Handle multiple records at once
- ✅ **REST Endpoints** - Easy integration with any application
- ✅ **Background Task Queue** - Predictions run async without interruption
- ✅ **Interactive Documentation** - Built-in Swagger UI

### API Endpoints Summary

| Endpoint               | Method | Purpose                                     |
| ---------------------- | ------ | ------------------------------------------- |
| `/health`              | GET    | Check API and database status               |
| `/model-info`          | GET    | Get ML model details                        |
| `/predict`             | POST   | Single record prediction                    |
| `/predict-batch`       | POST   | Batch predictions with auto-save to MongoDB |
| `/retrieve-pending`    | GET    | Process all pending records automatically   |
| `/sync-predictions`    | POST   | Sync entire database                        |
| `/predictions/{limit}` | GET    | Retrieve recent predictions                 |

### Python Client Example

```python
from src.api_client import MaintenanceAPIClient

# Initialize client
client = MaintenanceAPIClient("http://localhost:8000")

# Single prediction
record = {
    "typeMaintenance": "Corrective",
    "priorite": "Critique",
    "type": "Compresseur",
    "marque": "Siemens",
    "etat": "En Maintenance",
    "heuresFonctionnement": 15000,
    "equipment_age_days": 730,
    "nombre_pannes_historique": 5,
    "cout_maintenance_moyen_historique": 5000,
    "dureeEstimee": 6.5,
    "coutEstime": 6500,
    "garantie": False
}

result = client.predict_single(record)
print(f"Predicted Cost: ${result['predicted_cost']:.2f}")
print(f"Difference: {result['percentage_difference']:+.1f}%")

# Batch prediction (auto-saves to MongoDB)
records = [record]  # Can add multiple
batch_result = client.predict_batch(records)
print(f"Processed {batch_result['successful_predictions']} records")

# Process pending records from database
client.retrieve_pending()  # Returns immediately, runs in background

# Get recent predictions
predictions = client.get_recent_predictions(limit=10)
```

### Workflow Diagram

```
HTTP Request with Maintenance Data
           ↓
    FastAPI Endpoint
           ↓
    Load ML Model
           ↓
    Feature Engineering
           ↓
    Make Prediction
           ↓
    Add to Background Queue
           ↓
 ╔─ API Response ────────────────────────┐
 │  (Immediate - Non-blocking)           │
 │  ✓ Returns prediction result          │
 │  ✓ Returns immediately                │
 │                                        │
 │  Background Task:                     │
 │  → Connect to MongoDB                 │
 │  → Save prediction + metadata         │
 │  → Update equipment record            │
 └────────────────────────────────────────┘
```

**See [API_SETUP.md](API_SETUP.md) for complete API documentation.**

---

### 2. Or Use Pre-Trained Model

```python
import joblib

# Load trained model
model = joblib.load('models/best_maintenance_cost_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Make predictions (see examples below)
```

---

## 📁 Project Structure

```
equipment-failure-predictor/
├── 📓 maintenance_cost_prediction.ipynb    ⭐ Run this first!
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── PROJECT_STRUCTURE.md                    # Detailed structure guide
│
├── data/
│   ├── raw/                               # 🗂️ Original CSV files (4 essential tables)
│   │   ├── equipements.csv               # Equipment specs (500 records)
│   │   ├── maintenances.csv              # Maintenance records (2,000 records)
│   │   ├── interventions.csv             # Interventions (3,000 records)
│   │   └── pannes.csv                    # Equipment failures (1,500 records)
│   ├── processed/                         # Cleaned & engineered data
│   │   └── maintenance_cost_ml_dataset.csv  # ML-ready dataset
│   └── predictions/                       # Model predictions
│       └── example_predictions.csv
│
├── models/                                 # 🤖 ML artifacts
│   ├── best_maintenance_cost_model.pkl    # Trained Lasso model
│   ├── feature_scaler.pkl                 # StandardScaler
│   ├── label_encoders.pkl                 # Categorical encoders
│   └── model_summary.json                 # Model metadata
│
├── src/                                    # 💻 Python scripts
│   ├── generate_maintenance_dataset.py    # Data generation
│   ├── train_maintenance_cost_model.py    # Model training
│   └── predict_maintenance_cost.py        # Prediction utility
│
└── visualizations/                         # 📈 Performance plots
    ├── model_comparison.png
    ├── prediction_vs_actual.png
    ├── residuals_plot.png
    └── feature_correlation.png
```

---

## 📊 Dataset Description

### Data Tables Used

| Table                 | Records | Description                                 |
| --------------------- | ------- | ------------------------------------------- |
| **equipements.csv**   | 500     | Equipment inventory with specs, age, brand  |
| **maintenances.csv**  | 2,000   | Maintenance operations (TARGET: `coutReel`) |
| **interventions.csv** | 3,000   | Maintenance interventions & details         |
| **pannes.csv**        | 1,500   | Equipment failure history                   |

### Target Variable

- **`coutReel`** - Actual maintenance cost (what we predict)

### Maintenance Cost Statistics

```
Count:      506 completed maintenance records
Mean:       $4,367.84
Min:        $167.03
Max:        $19,895.69
Std Dev:    $3,441.98
```

---

## 🤖 Model Performance

### Best Model: Lasso Regression

| Metric                      | Value              |
| --------------------------- | ------------------ |
| **R² Score**                | **0.9970** (99.7%) |
| **Mean Absolute Error**     | **$112.33**        |
| **Root Mean Squared Error** | **$166.70**        |
| **Cross-Val R² (mean)**     | **0.9968**         |

### Model Comparison

| Model                   | R² Score   | MAE         | RMSE        |
| ----------------------- | ---------- | ----------- | ----------- |
| **Lasso Regression** ✅ | **0.9970** | **$112.33** | **$166.70** |
| Ridge Regression        | 0.9970     | $112.97     | $167.14     |
| Linear Regression       | 0.9970     | $112.23     | $166.96     |
| Random Forest           | 0.9956     | $118.83     | $203.19     |
| Gradient Boosting       | 0.9931     | $132.32     | $254.71     |

---

## 🔍 Top Features (by Importance)

1. **coutEstime** (0.999) - Estimated cost
2. **cout_maintenance_moyen_historique** (0.784) - Historical avg
3. **dureeEstimee** (0.444) - Estimated duration
4. **typeMaintenance** (0.433) - Type of work
5. **is_critical_priority** (0.427) - Critical flag
6. **is_corrective** (0.399) - Corrective flag
7. **is_preventive** (0.349) - Preventive flag
8. **equipment_age_years** (0.143) - Equipment age
9. Equipment type & brand
10. Operating hours

---

## � Installation

### Requirements

- Python 3.8+
- pip

### Step 1: Clone Repository

```bash
git clone https://github.com/BellilxDhaker/equipment-failure-predictor.git
cd equipment-failure-predictor
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3: (Optional) Install Jupyter

```bash
pip install jupyter notebook
```

---

## 💻 Usage

### Option 1: Interactive Jupyter Notebook (Recommended) 📓

```bash
jupyter notebook maintenance_cost_prediction.ipynb
```

This notebook includes:

- ✅ Data loading & exploration
- ✅ Feature engineering
- ✅ Model training (5 algorithms)
- ✅ Model evaluation & comparison
- ✅ Visualizations
- ✅ Predictions on new data
- ✅ Step-by-step explanations

### Option 2: Use Pre-Trained Model

```python
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load('models/best_maintenance_cost_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Create a maintenance record
data = {
    'typeMaintenance': 'Corrective',
    'priorite': 'Critique',
    'type': 'Compresseur',
    'marque': 'Siemens',
    'etat': 'En Maintenance',
    'heuresFonctionnement': 15000,
    'equipment_age_days': 730,
    'nombre_pannes_historique': 5,
    'cout_maintenance_moyen_historique': 5000,
    'dureeEstimee': 6.5,
    'coutEstime': 6500,
    'garantie': False
}

# Prepare features
df = pd.DataFrame([data])

# Feature engineering
df['is_critical_priority'] = (df['priorite'] == 'Critique').astype(int)
df['is_corrective'] = (df['typeMaintenance'] == 'Corrective').astype(int)
df['is_preventive'] = (df['typeMaintenance'] == 'Préventive').astype(int)
df['has_garantie'] = df['garantie'].astype(int)
df['equipment_age_years'] = df['equipment_age_days'] / 365.25
df['hours_per_day'] = df['heuresFonctionnement'] / (df['equipment_age_days'] + 1)

# Encode categorical variables
for col in ['typeMaintenance', 'priorite', 'type', 'marque', 'etat']:
    df[f'{col}_encoded'] = label_encoders[col].transform(df[col])

# Select features in correct order
feature_columns = [
    'typeMaintenance_encoded', 'priorite_encoded', 'type_encoded',
    'marque_encoded', 'etat_encoded', 'heuresFonctionnement',
    'equipment_age_days', 'equipment_age_years',
    'nombre_pannes_historique', 'cout_maintenance_moyen_historique',
    'dureeEstimee', 'coutEstime', 'is_critical_priority',
    'is_corrective', 'is_preventive', 'has_garantie', 'hours_per_day'
]

X = df[feature_columns]
X_scaled = scaler.transform(X)

# Make prediction
predicted_cost = model.predict(X_scaled)[0]
print(f"Predicted Maintenance Cost: ${predicted_cost:,.2f}")
```

### Option 3: Run Python Scripts

```bash
# Train the model from scratch
python src/train_maintenance_cost_model.py

# Make predictions
python src/predict_maintenance_cost.py

# Generate new dataset
python src/generate_maintenance_dataset.py
```

---

## 📚 Examples

### Prediction Scenario 1: Critical Corrective

**Input:**

```
Maintenance Type:  Corrective
Priority:          Critique
Equipment:         Compresseur (Siemens)
Age:               2.0 years
Operating Hours:   15,000
Historical Fails:  5
Estimated Cost:    $6,500.00
```

**Prediction:**

```
✅ PREDICTED COST: $6,876.01
   Difference: +$376.01 (+5.8%)
```

### Prediction Scenario 2: Preventive Maintenance

**Input:**

```
Maintenance Type:  Préventive
Priority:          Moyenne
Equipment:         Pompe (ABB)
Age:               1.0 year
Operating Hours:   8,000
Historical Fails:  1
Estimated Cost:    $2,200.00
```

**Prediction:**

```
✅ PREDICTED COST: $2,306.03
   Difference: +$106.03 (+4.8%)
```

### Prediction Scenario 3: Predictive Maintenance

**Input:**

```
Maintenance Type:  Prédictive
Priority:          Haute
Equipment:         Générateur (Schneider)
Age:               4.0 years
Operating Hours:   25,000
Historical Fails:  8
Estimated Cost:    $8,000.00
```

**Prediction:**

```
✅ PREDICTED COST: $8,456.45
   Difference: +$456.45 (+5.7%)
```

---

## 💡 Key Insights

### What the Model Learned

1. **Estimated cost is the strongest predictor** (99.9% correlation)
   - The model identifies cost adjustment factors

2. **Historical data patterns matter**
   - Equipment with more failures → higher maintenance costs
   - Historical average is highly predictive

3. **Maintenance type drives costs**
   - Corrective: ~150% vs baseline
   - Preventive: ~70% vs baseline
   - Predictive: ~100% vs baseline

4. **Priority level multiplier**
   - Critical: 2.0x cost multiplier
   - Haute: 1.3x cost multiplier
   - Moyenne: 1.0x (baseline)
   - Basse: 0.8x cost multiplier

5. **Equipment age impact**
   - Older equipment → higher costs
   - ~50% increase over 10 years

6. **Model stability**
   - Cross-validation R²: 0.9968
   - Low variance → generalizes well

---

## 📖 Features

### Feature Engineering (17 Features)

- **Encoded Features**: Type, Priority, Equipment type, Brand, State
- **Numeric Features**: Hours, Age, Historical failures, Cost, Duration
- **Binary Flags**: Critical priority, Corrective, Preventive, Warranty
- **Derived Features**: Equipment age in years, Hours per day

---

## 🎯 Use Cases

| Use Case                    | Benefit                                         |
| --------------------------- | ----------------------------------------------- |
| **Budget Planning**         | Predict actual costs vs estimates for budgeting |
| **Resource Allocation**     | Identify high-cost maintenance operations       |
| **Preventive Optimization** | Schedule maintenance to minimize costs          |
| **Equipment Lifecycle**     | Track cost trends over equipment lifetime       |
| **Decision Support**        | Compare predicted vs estimated before approval  |
| **Cost Anomaly Detection**  | Flag unexpected maintenance cost outliers       |

---

## 📊 Performance Visualizations

The project includes several analysis visualizations:

- **Model Comparison Chart**: R² scores across 5 algorithms
- **Predictions vs Actual**: Model accuracy scatter plot
- **Residuals Plot**: Error distribution analysis
- **Feature Correlation**: Top features by importance

View them in `visualizations/` directory or generate in Jupyter notebook.

---

## 📝 Data Dictionary

### Maintenance Dataset Columns

| Column                 | Type      | Description                               |
| ---------------------- | --------- | ----------------------------------------- |
| `id_maintenance`       | int       | Unique maintenance ID                     |
| `typeMaintenance`      | str       | Type: Préventive, Corrective, Prédictive  |
| `priorite`             | str       | Priority: Basse, Moyenne, Haute, Critique |
| `coutEstime`           | float     | Estimated cost ($)                        |
| **`coutReel`**         | **float** | **Actual cost ($) - TARGET**              |
| `dureeEstimee`         | float     | Estimated duration (hours)                |
| `heuresFonctionnement` | int       | Equipment operating hours                 |
| `equipment_age_days`   | int       | Equipment age in days                     |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License .

---

## 📈 Project Stats

- **Accuracy**: 99.7% R² Score
- **Models Evaluated**: 5 different algorithms
- **Features Used**: 17 engineered features
- **Training Data**: 506 maintenance records
- **Processing**: Feature scaling + categorical encoding
- **Library**: scikit-learn, pandas, numpy

---

## Checklist for Getting Started

- [ ] Clone the repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Open Jupyter notebook (`jupyter notebook maintenance_cost_prediction.ipynb`)
- [ ] Run cells sequentially (1-11)
- [ ] Make predictions on new data
- [ ] Explore visualizations
