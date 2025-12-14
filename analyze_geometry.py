import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# --- CONFIG ---
INPUT_FILE = "data/phase4_geometry_features.jsonl"
REPORT_DIR = "reports/phase4_pairwise_separation"

def load_data():
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def plot_scatter(df, x_col, y_col, output_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='label', palette={0: 'blue', 1: 'red'}, alpha=0.7)
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def train_eval(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    except:
        print("ROC-AUC:   N/A")

def main():
    print("=== GEOFENCE-LLM Phase 4.2: Feature Interaction & Separability Analysis ===")
    
    ensure_dir(REPORT_DIR)
    
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records.")
    
    features = ['tortuosity', 'directional_mean', 'directional_var', 'energy_drift', 'velocity_variance']
    
    # 2. Correlation Analysis
    print("\n--- 1. Feature Correlation Analysis ---")
    corr = df[features].corr()
    print(corr.round(4))
    
    # 3. Pairwise Scatter Plots
    print("\n--- 2. Pairwise Separability Check ---")
    pairs = [
        ('energy_drift', 'velocity_variance'),
        ('directional_mean', 'velocity_variance'),
        ('energy_drift', 'directional_var'),
        ('tortuosity', 'velocity_variance')
    ]
    
    for x, y in pairs:
        path = os.path.join(REPORT_DIR, f"{x}_vs_{y}.png")
        plot_scatter(df, x, y, path)
        
    # 4. Modeling (Linear vs Nonlinear)
    print("\n--- Modeling Setup ---")
    X = df[['energy_drift', 'velocity_variance', 'directional_var']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Linear
    print("\n--- 3. Simple Linear Separability Test (Logistic Regression) ---")
    logreg = LogisticRegression(random_state=42)
    train_eval(logreg, X_train, X_test, y_train, y_test, "Logistic Regression")
    
    # Nonlinear
    print("\n--- 4. Nonlinear Separability Test (Decision Tree depth=3) ---")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    train_eval(dt, X_train, X_test, y_train, y_test, "Decision Tree (d=3)")
    
    # Explanation / Qualitative
    print("\n--- Summary ---")
    print("Feature importance (DT):")
    for name, imp in zip(X.columns, dt.feature_importances_):
        print(f"  {name}: {imp:.4f}")
        
    print("\nPhase 4.2 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
