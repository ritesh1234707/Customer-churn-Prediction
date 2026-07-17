"""
Evaluation & Business Simulation Module
----------------------------------------
Responsible for:
1. Generating a side-by-side performance comparison matrix (ROC-AUC, Precision, Recall, F1) for all models.
2. Plotting ROC curves and saving the result as a styled PNG image.
3. Simulating business financial savings under different classification thresholds to optimize target selection.

Author: Senior Machine Learning Engineer
Date: 2026-07-01
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

def evaluate_performance(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Computes performance metrics (ROC-AUC, Precision, Recall, F1-Score) for all trained models.
    Uses default threshold of 0.5 for classification metrics.
    
    Returns:
        pd.DataFrame: A comparison matrix of the models.
    """
    metrics_list = []
    
    for name, model in models.items():
        # Predict class probabilities and hard class labels
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_probs)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_list.append({
            'Model': name,
            'ROC-AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
    df_metrics = pd.DataFrame(metrics_list).set_index('Model')
    return df_metrics

def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = "roc_curves.png"):
    """
    Plots the ROC curves of all models side-by-side on a single chart and saves it to disk.
    Applies custom styling to create a premium, clean design.
    """
    plt.figure(figsize=(10, 8))
    
    # Premium color palette
    colors = {
        'Logistic Regression': '#4A90E2',  # Slate Blue
        'Random Forest': '#E67E22',        # Coral / Orange
        'XGBoost': '#2ECC71'              # Emerald Green
    }
    
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc = roc_auc_score(y_test, y_probs)
        
        color = colors.get(name, '#7F8C8D')
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {auc:.4f})')
        
    # Baseline diagonal line (Random guessing)
    plt.plot([0, 1], [0, 1], color='#BDC3C7', linestyle='--', lw=1.5, label='Random Guessing (AUC = 0.50)')
    
    # Styling details
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7')
    plt.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='#BDC3C7')
    
    # Clean background layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] ROC curves plotted and saved to: {save_path}")

def simulate_business_savings(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    clv: float = 500.0, 
    retention_cost: float = 20.0, 
    success_rate: float = 0.5
) -> dict:
    """
    Simulates financial savings across decision thresholds from 0.0 to 1.0.
    
    Args:
        model: The trained classifier to use for probabilities.
        X_test: Test features.
        y_test: Test labels.
        clv: Customer Lifetime Value (loss incurred if customer churns and is not targeted).
        retention_cost: Cost of running the retention program per targeted customer.
        success_rate: Probability that targeting a true churner successfully saves them.
        
    Returns:
        dict: Summary results, including baseline cost, optimal threshold, maximum savings,
              and detailed threshold-by-threshold dataframe.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.0, 1.01, 0.05)
    results = []
    
    # Baseline cost: loss if we target nobody. All actual churners churn.
    actual_churners = sum(y_test == 1)
    baseline_cost = actual_churners * clv
    
    for t in thresholds:
        # Avoid standard confusion matrix to handle edge thresholds (0.0, 1.0) safely
        tp = np.sum((y_probs >= t) & (y_test == 1))
        fp = np.sum((y_probs >= t) & (y_test == 0))
        fn = np.sum((y_probs < t) & (y_test == 1))
        tn = np.sum((y_probs < t) & (y_test == 0))
        
        # Financial Math:
        # Cost = (FN * CLV) + (TP * [Retention Cost + (1 - success_rate) * CLV]) + (FP * Retention Cost)
        # We can simplify this to:
        cost_model = (fn * clv) + (tp * (retention_cost + (1 - success_rate) * clv)) + (fp * retention_cost)
        savings = baseline_cost - cost_model
        
        # Calculate classification metrics at this specific threshold
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'Threshold': round(t, 2),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Targeted': tp + fp,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Total Cost ($)': cost_model,
            'Savings ($)': savings
        })
        
    df_results = pd.DataFrame(results)
    
    # Identify optimal threshold based on maximum savings
    optimal_idx = df_results['Savings ($)'].idxmax()
    optimal_row = df_results.loc[optimal_idx]
    
    # Default threshold (0.50) statistics
    default_row = df_results[df_results['Threshold'] == 0.50].iloc[0]
    
    # Target everyone (0.00) statistics
    target_all_row = df_results[df_results['Threshold'] == 0.00].iloc[0]
    
    summary = {
        'baseline_cost': baseline_cost,
        'actual_churners': actual_churners,
        'optimal_threshold': optimal_row['Threshold'],
        'max_savings': optimal_row['Savings ($)'],
        'optimal_cost': optimal_row['Total Cost ($)'],
        'optimal_recall': optimal_row['Recall'],
        'optimal_precision': optimal_row['Precision'],
        'optimal_targeted': int(optimal_row['Targeted']),
        'default_savings': default_row['Savings ($)'],
        'default_recall': default_row['Recall'],
        'default_precision': default_row['Precision'],
        'target_all_savings': target_all_row['Savings ($)'],
        'df_results': df_results
    }
    
    return summary
