# Customer-churn-Predictio

Customer Churn Prediction & Financial Optimization
A production-grade binary classification framework designed to identify at-risk customers, mitigate class imbalance, and maximize financial retention savings using Logistic Regression, Random Forest, and XGBoost.

Project Overview
In customer retention, standard accuracy is a trap. If 95% of your customer base is loyal, a broken model that predicts "no churn" for every single user will achieve 95% accuracy while letting your business bleed revenue.

This project solves this by optimizing for ROC-AUC, Precision, and Recall, backed by a financial simulation layer. Instead of using a default 
0.5
 classification threshold, the pipeline shifts the threshold dynamically to find the exact mathematical sweet spot where retention spend minimizes customer acquisition losses.

Key Features
Imbalance Mitigation Engine: Tailored strategies per algorithm (SMOTE for Logistic Regression, balanced_subsample for Random Forest, and dynamic scale_pos_weight calculation for XGBoost).
Modular Architecture: Clean separation of concerns across data engineering, model training, evaluation, and business logic.
CLV Financial Simulator: Converts standard abstract ML metrics into concrete business metrics (Dollars Saved).
Project Structure
├── data_loader.py    # Synthetic data generation, stratified splitting, & scaling
├── models.py         # Model initialization & imbalanced-class training pipelines
├── evaluation.py     # Metric matrices, visual curves, & threshold optimization
└── main.py           # Master orchestration script

Getting Started
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:

pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib
2. Execution
Run the master execution script from your terminal to trigger the entire pipeline:

python main.py
Methodology & Components
1. Data Pipeline (data_loader.py)
Generates structural, non-linear data simulating realistic customer behaviors (tenure, usage drops, customer service complaints) with an intentional 9:1 class imbalance. It enforces a strict stratified split before scaling to prevent information leakage from SMOTE into the evaluation sets.

2. The Model Matrix (models.py)
Trains three distinct structural architectures configured specifically to combat skewed distributions:

Algorithm	Imbalance Strategy	Best For
Logistic Regression	SMOTE Oversampling	Baseline interpretability & linear relationships
Random Forest	class_weight='balanced_subsample'	Capturing complex feature interactions robustly
XGBoost	scale_pos_weight = ratio	High-performance gradient boosting on weak signals
3. Business Value Simulation (evaluation.py)
This component applies a financial cost-benefit matrix to model probabilities:

False Negative Cost: Lost Customer Lifetime Value ($500)
False Positive Cost: Wasted retention incentive spend ($20)
Incentive Success Rate: 50% effectiveness at stopping a true churner.
The simulator sweeps thresholds from 
0.0
 to 
1.0
 to isolate the operational setting that yields peak profitability.

Example Output
Upon execution, the terminal will render your comparative performance and financial optimization target:

============================================================
              MODEL PERFORMANCE COMPARISON                  
============================================================
                               ROC-AUC  Precision    Recall  F1-Score
Logistic Regression (SMOTE)     0.782      0.312     0.741     0.439
Random Forest (Balanced)       0.864      0.684     0.523     0.593
XGBoost (Scale Pos Weight)      0.891      0.641     0.712     0.675

============================================================
            FINANCIAL THRESHOLD OPTIMIZATION                
============================================================
[INFO] Default Threshold (0.50) Net Savings: $14,200.00
[SUCCESS] Optimal Business Threshold Found at: 0.35
[SUCCESS] Maximum Preventative Net Savings: $18,450.00

[INFO] Default Threshold (0.50) Net Savings: $14,200.00
[SUCCESS] Optimal Business Threshold Found at: 0.35
[SUCCESS] Maximum Preventative Net Savings: $18,450.00

Customization
To swap out the synthetic generator for production enterprise features, adjust the data ingest block within data_loader.py:

# Inside data_loader.py -> load_data()
df = pd.read_csv("your_enterprise_churn_data.csv")
X = df.drop("churn_label", axis=1)
y = df["churn_label"]
