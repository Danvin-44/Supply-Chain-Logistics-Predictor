import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import shap
import gradio as gr
import warnings

# Suppress warnings for clean deployment logs
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Data Pipeline
# ---------------------------------------------------------
print("Initializing data pipeline...")
df = pd.read_csv('scofdr.csv')

target_col = 'delayed'
cols_to_drop = [target_col, 'order_id', 'order_date']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore') 

numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[categorical_cols] = X[categorical_cols].fillna('Unknown')
X = pd.get_dummies(X, drop_first=True) 
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ---------------------------------------------------------
# 2. Model Optimization
# ---------------------------------------------------------
print("Starting Optuna hyperparameter tuning...")
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'random_state': 42
    }
    model = XGBClassifier(**param)
    model.fit(X_train_smote, y_train_smote)
    return model.score(X_test, y_test)

study = optuna.create_study(direction='maximize')
# Reduced trials slightly for faster server startup on Hugging Face
study.optimize(objective, n_trials=10) 

print(f"Optimal parameters: {study.best_params}")
best_model = XGBClassifier(**study.best_params, random_state=42)
best_model.fit(X_train_smote, y_train_smote)

# ---------------------------------------------------------
# 3. Cost-Matrix Threshold Optimization
# ---------------------------------------------------------
print("Calculating business-optimized probability threshold...")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Adjusted cost matrix: False Positive = $150, False Negative = $500
costs = []
thresholds = np.arange(0.1, 0.9, 0.05)

for thresh in thresholds:
    y_pred_custom = (y_pred_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()
    total_cost = (fp * 150) + (fn * 500)
    costs.append(total_cost)

optimal_threshold = thresholds[np.argmin(costs)]
print(f"Target threshold set to: {optimal_threshold:.2f}")

# ---------------------------------------------------------
# 4. Gradio Interface Definition
# ---------------------------------------------------------
print("Configuring Gradio interface...")
explainer = shap.TreeExplainer(best_model)

def predict_and_explain(test_row_index):
    row_data = X_test.iloc[[test_row_index]]
    actual_status = "Delayed" if y_test.iloc[test_row_index] == 1 else "On-Time"
    
    prob = best_model.predict_proba(row_data)[0, 1]
    prediction = "Delay Risk Flagged" if prob >= optimal_threshold else "On-Time Expected"
    
    shap_vals = explainer(row_data)
    
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.title(f"Model Inference: {prediction}")
    plt.tight_layout()
    
    plot_path = f"shap_local.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    report = (
        f"Ground Truth: {actual_status}\n"
        f"Model Output: {prediction} (Probability: {prob:.1%})\n\n"
        f"Threshold Logic:\n"
        f"Cost-matrix optimized threshold set at {optimal_threshold:.1%}. "
        f"Values exceeding this threshold trigger a delay warning based on the current FP/FN penalty configuration."
    )
    return report, plot_path

max_index = len(X_test) - 1

interface = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Slider(0, max_index, step=1, label="Test Set Index (Order ID Proxy)"),
    outputs=[
        gr.Textbox(label="Inference Results", lines=5),
        gr.Image(label="SHAP Value Breakdown")
    ],
    title="Logistics Delay Prediction Engine",
    description="End-to-end classification pipeline utilizing XGBoost and SMOTE. The model incorporates a custom business cost-matrix to optimize the decision threshold, visualized via local SHAP explanations.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch()