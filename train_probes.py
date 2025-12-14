import json
import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# --- CONFIG ---
INPUT_FILE = "data/phase5_2_geometry_features.jsonl"
MODELS_DIR = "models/phase5_2_probes"
FEATURES = [
    "directional_mean",
    "directional_var",
    "velocity_variance",
    "energy_drift",
    "tortuosity"
]
SEED = 42

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file missing: {INPUT_FILE}")
        sys.exit(1)
    
    X = []
    y = []
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Extract fixed features in order
                row = [rec[f] for f in FEATURES]
                X.append(row)
                y.append(rec['label'])
                
    return np.array(X), np.array(y)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "cm": cm.tolist()
    }

def main():
    print("=== GEOFENCE-LLM Phase 5.1: Distilled Probe Training ===")
    
    # 1. Setup
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # 2. Load Data
    X, y = load_data()
    print(f"Loaded {len(X)} samples. Features: {len(FEATURES)}")
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 4. Standardize (for LR and MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # --- MODEL 1: Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=SEED)
    lr.fit(X_train_scaled, y_train)
    evaluate_model("Logistic Regression", lr, X_test_scaled, y_test)
    
    # Feature Importance (Coefficients)
    print("LR Coefficients:")
    for name, coef in zip(FEATURES, lr.coef_[0]):
        print(f"  {name}: {coef:.4f}")
        
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    
    # --- MODEL 2: Shallow MLP ---
    # Input -> 16 -> 1. sklearn MLPClassifier automatically handles the output layer.
    # hidden_layer_sizes=(16,) means one hidden layer with 16 neurons.
    print("\nTraining Shallow MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=SEED,
        early_stopping=True
    )
    mlp.fit(X_train_scaled, y_train)
    evaluate_model("Shallow MLP", mlp, X_test_scaled, y_test)
    joblib.dump(mlp, os.path.join(MODELS_DIR, "shallow_mlp.joblib"))
    
    # --- MODEL 3: Decision Tree ---
    # Unscaled data for Trees (though sklearn trees handle unscaled fine, usually better to keep original for interpretability)
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=SEED
    )
    dt.fit(X_train, y_train) # Use unscaled X_train
    evaluate_model("Decision Tree", dt, X_test, y_test) # Use unscaled X_test
    
    # Feature Importance
    print("DT Feature Importances:")
    for name, imp in zip(FEATURES, dt.feature_importances_):
        if imp > 0:
            print(f"  {name}: {imp:.4f}")
            
    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree.joblib"))
    
    print(f"\nModels saved to {MODELS_DIR}")
    print("Phase 5.1 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
