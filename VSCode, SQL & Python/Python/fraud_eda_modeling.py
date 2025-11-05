"""
====================================================================
💡 SMART BUDGET / Risk & Fraud Detection — Enhanced Python Pipeline
--------------------------------------------------------------------
Author: ChatGPT (adapted for Aïmane)
Date: 2025-11-03
Purpose:
    This Python script connects to PostgreSQL, performs Exploratory Data Analysis (EDA),
    trains machine learning models for fraud detection, evaluates them,
    and exports all results (summaries, KPIs, and model metrics)
    directly to CSVs for Power BI dashboards.

Key Outputs (for Power BI):
    ✅ model_performance_summary.csv   → Model metrics (Accuracy, ROC-AUC, etc.)
    ✅ fraud_summary_by_day.csv        → Daily trends
    ✅ fraud_summary_by_hour.csv       → Hourly behavior
    ✅ fraud_summary_by_user.csv       → User-level risk
    ✅ kpi_summary.csv                 → Single file for KPI cards
====================================================================
"""

# === IMPORTS ==========================================================
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
from datetime import datetime


# === CONFIGURATION & SETUP ============================================
"""
Here we:
- Load database credentials from a .env file
- Create folders for outputs and logs
- Prepare a connection to PostgreSQL
"""

load_dotenv()

# Database connection settings (editable in .env)
DB_USER = os.getenv('PG_USER', 'postgres')
DB_PASS = os.getenv('PG_PASS', None)
DB_HOST = os.getenv('PG_HOST', 'localhost')
DB_PORT = os.getenv('PG_PORT', '5432')
DB_NAME = os.getenv('PG_DB', 'cdb')
DB_SCHEMA = os.getenv('PG_SCHEMA', 'public')

# Output directories
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './outputs')
LOG_DIR = os.getenv('LOG_DIR', './logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging to track model training & export steps
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'fraud_modeling.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ensure password is provided
if not DB_PASS:
    raise SystemExit("❌ Missing PG_PASS in .env file")

# Connect to PostgreSQL
try:
    engine = create_engine(
        f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}',
        echo=False
    )
    with engine.connect() as conn:
        conn.execute(text(f"SET search_path TO {DB_SCHEMA};"))
        print("✅ Connected to database:", DB_NAME)
except Exception as e:
    print("❌ Database connection failed:", e)
    raise SystemExit(e)


# === LOAD DATA ========================================================
"""
We now load the cleaned dataset produced by fraud_pipeline.sql.
This table contains synthetic card IDs, timestamps, and engineered features.
"""
query = "SELECT * FROM fraud.transactions_clean;"
df = pd.read_sql(query, engine, parse_dates=['transaction_ts'])
print("Loaded:", len(df), "rows")


# === BASIC EXPLORATORY ANALYSIS ======================================
"""
We calculate overall fraud statistics and transaction behavior.
These metrics will feed into Power BI KPI cards.
"""
fraud_counts = df['class'].value_counts().sort_index()
total_tx = len(df)
fraud_tx = fraud_counts.get(1, 0)
fraud_rate = fraud_tx / total_tx
avg_amount = df['amount'].mean()
max_amount = df['amount'].max()


# === CORRELATION MATRIX ==============================================
"""
We measure how strongly each numerical variable relates to others.
This helps identify important fraud predictors (like night transactions).
"""
num_cols = [
    'amount', 'transaction_amount_log', 'transaction_hour', 'is_night_transaction',
    'transactions_per_user_last_24h', 'avg_amount_last_7days'
] + [f'v{i}' for i in range(1, 29)]

corr = df[num_cols].corr().reset_index().melt(id_vars='index')
corr.columns = ['variable_1', 'variable_2', 'correlation_value']
corr.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix_long.csv'), index=False)


# === MODELING PIPELINE ===============================================
"""
We now train multiple models to detect fraud:
- Logistic Regression (simple baseline)
- Random Forest (robust tree-based model)
- XGBoost (high-performance boosting model)

Steps:
1️⃣ Prepare data (scaling & splitting)
2️⃣ Balance classes using SMOTE (since fraud is rare)
3️⃣ Train each model and compute key metrics
"""

# Select input features
features = [
    'transaction_amount_log', 'transaction_hour', 'is_night_transaction',
    'transactions_per_user_last_24h', 'avg_amount_last_7days'
] + [f'v{i}' for i in range(1, 29)]

X = df[features].copy()
X['is_night_transaction'] = X['is_night_transaction'].astype(int)
y = df['class'].astype(int)

# Split data into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Normalize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance (SMOTE = Synthetic Minority Oversampling)
try:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
except:
    X_train_res, y_train_res = X_train_scaled, y_train


# === MODEL EVALUATION FUNCTION =======================================
"""
This helper function trains a model, makes predictions,
and returns accuracy, precision, recall, F1, and ROC-AUC.
"""
def evaluate_model(clf, X_t, X_v, y_t, y_v, name="model"):
    y_pred = clf.predict(X_v)
    y_proba = clf.predict_proba(X_v)[:,1] if hasattr(clf, "predict_proba") else None
    acc = accuracy_score(y_v, y_pred)
    prec = precision_score(y_v, y_pred, zero_division=0)
    rec = recall_score(y_v, y_pred)
    f1 = f1_score(y_v, y_pred)
    roc = roc_auc_score(y_v, y_proba) if y_proba is not None else np.nan
    return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}


# === TRAIN MODELS =====================================================
"""
Train and compare models on the same dataset.
We'll keep all results for Power BI visualization.
"""
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
xgb = XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1)

models = [("LogisticRegression", lr), ("RandomForest", rf), ("XGBoost", xgb)]
results = []

for name, model in models:
    model.fit(X_train_res, y_train_res)
    results.append(evaluate_model(model, X_train_res, X_test_scaled, y_train_res, y_test, name))

# Save all model performance metrics
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)


# === FEATURE IMPORTANCE EXPORT =======================================
"""
We record which variables were most influential for Random Forest and XGBoost.
This helps explain “why” a transaction was flagged.
"""
rf_imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
xgb_imp = pd.DataFrame({'feature': features, 'importance': xgb.feature_importances_}).sort_values('importance', ascending=False)
rf_imp.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importances.csv'), index=False)
xgb_imp.to_csv(os.path.join(OUTPUT_DIR, 'xgb_feature_importances.csv'), index=False)


# === EXPORT SUMMARIES FOR POWER BI ====================================
"""
We aggregate transactions into:
- Hourly trends (fraud spikes during the night)
- Daily trends (activity over time)
- User-level risk summaries (per card)
These are directly usable in Power BI visuals.
"""
df['date'] = df['transaction_ts'].dt.date

# Hourly summary
summary_by_hour = df.groupby(['date', 'transaction_hour']).agg(
    total_transactions=('id', 'count'),
    fraud_transactions=('class', 'sum'),
    avg_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    min_amount=('amount', 'min')
).reset_index()
summary_by_hour['fraud_rate'] = summary_by_hour['fraud_transactions'] / summary_by_hour['total_transactions']
summary_by_hour['is_night'] = np.where(summary_by_hour['transaction_hour'].between(0,5), 1, 0)
summary_by_hour.to_csv(os.path.join(OUTPUT_DIR, 'fraud_summary_by_hour.csv'), index=False)

# Daily summary
summary_by_day = df.groupby('date').agg(
    total_transactions=('id','count'),
    fraud_transactions=('class','sum'),
    avg_amount=('amount','mean'),
    max_amount=('amount','max'),
    min_amount=('amount','min')
).reset_index()
summary_by_day['fraud_rate'] = summary_by_day['fraud_transactions'] / summary_by_day['total_transactions']
summary_by_day.to_csv(os.path.join(OUTPUT_DIR, 'fraud_summary_by_day.csv'), index=False)

# User summary
summary_by_user = df.groupby('card_id').agg(
    total_transactions=('id', 'count'),
    fraud_transactions=('class', 'sum'),
    avg_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    last_tx_ts=('transaction_ts', 'max')
).reset_index()
summary_by_user['fraud_rate'] = summary_by_user['fraud_transactions'] / summary_by_user['total_transactions']
summary_by_user['is_risky'] = np.where(summary_by_user['fraud_rate'] > fraud_rate, 1, 0)
summary_by_user.to_csv(os.path.join(OUTPUT_DIR, 'fraud_summary_by_user.csv'), index=False)


# === KPI SUMMARY (for Power BI Cards) ================================
"""
This creates a single-row CSV with business and model KPIs:
✅ total transactions, fraud rate, average spend
✅ best model name and accuracy
✅ peak fraud day/hour
✅ run timestamp
No DAX or Power BI formulas needed.
"""
best_model = res_df.sort_values('roc_auc', ascending=False).iloc[0]

kpi_data = {
    'total_transactions': total_tx,
    'fraud_transactions': fraud_tx,
    'fraud_rate': round(fraud_rate * 100, 3),
    'avg_transaction_amount': round(avg_amount, 2),
    'max_transaction_amount': round(max_amount, 2),
    'unique_users': df['card_id'].nunique(),
    'peak_fraud_hour': int(summary_by_hour.loc[summary_by_hour['fraud_rate'].idxmax(), 'transaction_hour']),
    'peak_fraud_date': str(summary_by_day.loc[summary_by_day['fraud_rate'].idxmax(), 'date']),
    'best_model': best_model['name'],
    'best_model_roc_auc': round(best_model['roc_auc'], 4),
    'best_model_accuracy': round(best_model['accuracy'], 4),
    'best_model_f1': round(best_model['f1'], 4),
    'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
pd.DataFrame([kpi_data]).to_csv(os.path.join(OUTPUT_DIR, 'kpi_summary.csv'), index=False)


# === FEATURE DESCRIPTION EXPORT =======================================
"""
We describe every column in plain language for documentation.
This helps non-technical users understand the dataset schema in Power BI.
"""
feature_descriptions = [
    {'column':'id','description':'Unique row ID'},
    {'column':'transaction_ts','description':'Transaction timestamp'},
    {'column':'card_id','description':'Synthetic user/card ID'},
    {'column':'amount','description':'Original transaction amount'},
    {'column':'transaction_amount_log','description':'Log-transformed transaction amount'},
    {'column':'transaction_hour','description':'Hour of day (0–23)'},
    {'column':'is_night_transaction','description':'1 if transaction hour between 0–5'},
    {'column':'transactions_per_user_last_24h','description':'# of user transactions in last 24h'},
    {'column':'avg_amount_last_7days','description':'User avg amount over past 7 days'},
    {'column':'class','description':'Fraud label: 1=fraud, 0=legit'}
] + [{'column': f'v{i}', 'description': f'Anonymized PCA feature V{i}'} for i in range(1,29)]

pd.DataFrame(feature_descriptions).to_csv(os.path.join(OUTPUT_DIR, 'feature_descriptions.csv'), index=False)


# === FINAL OUTPUT MESSAGE =============================================
print("\n✅ All Power BI export files saved to:", OUTPUT_DIR)
