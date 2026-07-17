"""
Data Loader Module
------------------
Responsible for:
1. Generating synthetic imbalanced customer churn data (if no external CSV is present).
2. Performing a stratified train-test split to preserve the minority class ratio.
3. Scaling features using standard normalization.

Author: Senior Machine Learning Engineer
Date: 2026-07-01
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, imbalanced synthetic customer churn dataset.
    
    The churn target is determined by a combination of factors: tenure,
    monthly charges, support calls, contract type, payment delay, and age.
    """
    np.random.seed(random_state)
    
    # Generate realistic features
    tenure = np.random.randint(1, 73, size=n_samples)  # 1 to 72 months
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)  # $20 to $120
    
    # total_charges is tenure * monthly_charges with some noise
    noise = np.random.normal(0, 15, size=n_samples)
    total_charges = (tenure * monthly_charges) + noise
    total_charges = np.clip(total_charges, a_min=monthly_charges, a_max=None)
    
    age = np.random.randint(18, 81, size=n_samples)
    support_calls = np.random.poisson(lam=1.5, size=n_samples)
    support_calls = np.clip(support_calls, 0, 10)
    
    payment_delay = np.random.poisson(lam=2.5, size=n_samples)
    payment_delay = np.clip(payment_delay, 0, 30)
    
    # 0: Month-to-month, 1: One-year, 2: Two-year
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Define a logit relationship to generate the churn label
    # Negative weights decrease churn probability, positive weights increase it
    logit = (
        -2.6 
        - 0.05 * tenure 
        + 0.012 * monthly_charges 
        + 0.55 * support_calls 
        + 0.07 * payment_delay 
        - 1.1 * contract_type 
        - 0.008 * age
    )
    
    # Probability using sigmoid function
    prob_churn = 1 / (1 + np.exp(-logit))
    
    # Churn label based on probability
    churn = np.random.binomial(1, prob_churn)
    
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'age': age,
        'support_calls': support_calls,
        'payment_delay': payment_delay,
        'contract_type': contract_type,
        'churn': churn
    })
    
    return df

def load_and_preprocess_data(
    file_path: str = None, 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Loads data (from file_path if provided and exists, otherwise generates synthetic data),
    performs a stratified train-test split, scales the features, and returns the data split
    along with the scaler.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_raw
    """
    if file_path and os.path.exists(file_path):
        print(f"[DataLoader] Loading data from existing file: {file_path}")
        df = pd.read_csv(file_path)
    else:
        print("[DataLoader] No pre-existing data file found. Generating synthetic data...")
        df = generate_synthetic_data(n_samples=5000, random_state=random_state)
        # Save synthetic data to the directory for documentation/inspection
        output_csv = "churn_data_synthetic.csv"
        df.to_csv(output_csv, index=False)
        print(f"[DataLoader] Synthetic dataset saved to: {os.path.abspath(output_csv)}")
        
    # Check shape and churn rate
    n_total = len(df)
    n_churn = df['churn'].sum()
    churn_rate = n_churn / n_total
    print(f"[DataLoader] Dataset Loaded. Samples: {n_total}, Churners: {n_churn} ({churn_rate:.2%})")
    
    # Features and target split
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled features back to DataFrame for ease of use/tracking columns
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, df

if __name__ == "__main__":
    # Test execution
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data()
    print("Train features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Train label distribution:\n", y_train.value_counts())
    print("Test label distribution:\n", y_test.value_counts())
