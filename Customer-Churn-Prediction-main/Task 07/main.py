"""
Central Orchestrator
--------------------
Orchestrates the entire machine learning pipeline:
1. Loads and preprocesses data (synthetic or local CSV).
2. Trains Logistic Regression, Random Forest, and XGBoost with native class imbalance handling.
3. Evaluates all models and prints a side-by-side metric comparison.
4. Plots ROC curves and saves the visualization.
5. Identifies the best-performing model using ROC-AUC.
6. Simulates business value and financial savings across various decision thresholds.
7. Prints a summary report with strategic recommendations.

Author: Senior Machine Learning Engineer
Date: 2026-07-01
"""

import sys
import traceback
import pandas as pd
from data_loader import load_and_preprocess_data
from models import train_models
from evaluation import evaluate_performance, plot_roc_curves, simulate_business_savings

def run_pipeline():
    print("=" * 70)
    print("        CUSTOMER CHURN PREDICTION & FINANCIAL OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    try:
        X_train, X_test, y_train, y_test, scaler, df_raw = load_and_preprocess_data()
    except Exception as e:
        print(f"[Error] Failed to load and preprocess data: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 2. Train Models
    try:
        models = train_models(X_train, y_train)
    except Exception as e:
        print(f"[Error] Failed to train machine learning models: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Evaluate Models
    try:
        print("[Pipeline] Evaluating model performance on test split...")
        df_metrics = evaluate_performance(models, X_test, y_test)
        
        print("\n" + "-" * 50)
        print("          SIDE-BY-SIDE MODEL EVALUATION MATRIX")
        print("    (Metrics computed at default decision threshold of 0.5)")
        print("-" * 50)
        print(df_metrics.to_string(formatters={
            'ROC-AUC': '{:,.4f}'.format,
            'Precision': '{:,.4%}'.format,
            'Recall': '{:,.4%}'.format,
            'F1-Score': '{:,.4%}'.format
        }))
        print("-" * 50)
        
    except Exception as e:
        print(f"[Error] Failed to evaluate models: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # 4. Plot ROC Curves
    try:
        roc_plot_path = "roc_curves.png"
        plot_roc_curves(models, X_test, y_test, save_path=roc_plot_path)
    except Exception as e:
        print(f"[Warning] Failed to plot ROC curves: {e}")
        
    # 5. Identify Best Model (by ROC-AUC)
    best_model_name = df_metrics['ROC-AUC'].idxmax()
    best_model = models[best_model_name]
    print(f"\n[Pipeline] Best model selected based on ROC-AUC: **{best_model_name}** "
          f"(AUC: {df_metrics.loc[best_model_name, 'ROC-AUC']:.4f})")
    
    # 6. Business Value & CLV Optimization Simulation
    try:
        print(f"\n[Pipeline] Simulating Customer Lifetime Value (CLV) savings with {best_model_name}...")
        
        # Simulation parameters
        CLV = 500.0          # Loss if False Negative
        RET_COST = 20.0      # Spent on True Positive or False Positive
        SUCCESS_RATE = 0.50  # Chance promo saves actual churner
        
        sim_results = simulate_business_savings(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            clv=CLV,
            retention_cost=RET_COST,
            success_rate=SUCCESS_RATE
        )
        
        # Display table of threshold results (subset for clarity)
        df_grid = sim_results['df_results']
        display_cols = ['Threshold', 'TP', 'FP', 'FN', 'TN', 'Targeted', 'Precision', 'Recall', 'Savings ($)']
        
        print("\n" + "-" * 80)
        print("                 DECISION THRESHOLD SHIFTING SIMULATION GRID")
        print("-" * 80)
        print(df_grid[display_cols].to_string(index=False, formatters={
            'Precision': '{:,.1%}'.format,
            'Recall': '{:,.1%}'.format,
            'Savings ($)': '${:,.2f}'.format
        }))
        print("-" * 80)
        
        # Print financial comparisons
        baseline_cost = sim_results['baseline_cost']
        max_savings = sim_results['max_savings']
        opt_thresh = sim_results['optimal_threshold']
        opt_cost = sim_results['optimal_cost']
        opt_precision = sim_results['optimal_precision']
        opt_recall = sim_results['optimal_recall']
        opt_targeted = sim_results['optimal_targeted']
        
        default_savings = sim_results['default_savings']
        target_all_savings = sim_results['target_all_savings']
        
        print("\n" + "=" * 70)
        print("                        FINANCIAL OPTIMIZATION REPORT")
        print("=" * 70)
        print(f"Baseline Cost (Do Nothing):             ${baseline_cost:,.2f} (Loss from {sim_results['actual_churners']} churners)")
        print(f"Target All Strategy (Threshold = 0.0):   Savings of ${target_all_savings:,.2f}")
        print(f"Default Model Strategy (Threshold = 0.5): Savings of ${default_savings:,.2f}")
        print(f"Optimal Model Strategy (Threshold = {opt_thresh:.2f}): Savings of ${max_savings:,.2f}  <-- [MAX SAVINGS]")
        print("-" * 70)
        print(f"Optimized Strategy Details at Threshold = {opt_thresh:.2f}:")
        print(f"  * Total Customers Targeted:   {opt_targeted} of {len(y_test)} ({opt_targeted/len(y_test):.1%})")
        print(f"  * Model Precision:            {opt_precision:.1%}")
        print(f"  * Model Recall:               {opt_recall:.1%}")
        print(f"  * Remaining Churn Loss + Cost: ${opt_cost:,.2f}")
        print(f"  * Net Business Value Added:   ${max_savings:,.2f} ({max_savings / baseline_cost:.1%} reduction in churn loss)")
        print("=" * 70)
        
        # Strategic Recommendation output
        print("\n[Strategic Recommendations]")
        print(f"1. **Deploy {best_model_name}** as the primary classifier for the customer retention campaign.")
        print(f"2. **Shift the decision threshold from the standard 0.50 to {opt_thresh:.2f}**.")
        if opt_thresh < 0.50:
            print(f"   - Since the cost of targeting (${RET_COST}) is low relative to the CLV loss (${CLV}), "
                  f"it is financially optimal to accept a lower threshold ({opt_thresh:.2f}) to capture more "
                  f"potential churners (Recall: {opt_recall:.1%}), even if it increases false positives.")
        else:
            print(f"   - The model has higher precision at {opt_thresh:.2f}, allowing the business to run "
                  f"a highly targeted campaign with minimal incentive waste.")
        print(f"3. **Expected Business Impact**: Saving ${max_savings:,.2f} on this test cohort "
              f"alone, which represents a {max_savings / baseline_cost:.1%} cost reduction compared to a do-nothing baseline.")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"[Error] Failed to execute business savings simulation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
