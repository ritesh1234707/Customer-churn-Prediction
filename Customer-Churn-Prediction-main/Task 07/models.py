"""
Models Module
-------------
Responsible for:
1. Initializing and training the three required classifiers:
   - Logistic Regression (with class_weight='balanced')
   - Random Forest (with class_weight='balanced_subsample')
   - XGBoost (with scale_pos_weight dynamically computed from training targets)
2. Handling class-imbalance natively during model training.

Author: Senior Machine Learning Engineer
Date: 2026-07-01
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def train_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> dict:
    """
    Trains Logistic Regression, Random Forest, and XGBoost models on the training data.
    Implements native class imbalance handling for each algorithm.
    
    Args:
        X_train: Standardized training features DataFrame.
        y_train: Training labels Series.
        random_state: Seed for reproducibility.
        
    Returns:
        A dictionary containing the trained model objects.
    """
    print("\n[Model Training] Starting model training pipeline...")
    
    # 1. Logistic Regression
    print("[Model Training] Training Logistic Regression (class_weight='balanced')...")
    lr_model = LogisticRegression(
        class_weight='balanced', 
        random_state=random_state, 
        max_iter=1000
    )
    lr_model.fit(X_train, y_train)
    
    # 2. Random Forest
    print("[Model Training] Training Random Forest (class_weight='balanced_subsample')...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced_subsample',
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # 3. XGBoost
    # Calculate scale_pos_weight dynamically: negative_cases / positive_cases
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    
    if pos_count == 0:
        raise ValueError("Training targets contain zero positive (churn) labels. Cannot compute scale_pos_weight.")
        
    scale_pos_weight = neg_count / pos_count
    print(f"[Model Training] Training XGBoost (scale_pos_weight={scale_pos_weight:.4f})...")
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    print("[Model Training] All models trained successfully.\n")
    
    return {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

if __name__ == "__main__":
    # Test execution
    from data_loader import load_and_preprocess_data
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data()
    models = train_models(X_train, y_train)
    for name, model in models.items():
        print(f"Trained {name} object: {type(model)}")
