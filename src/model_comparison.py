"""
PHASE 6: MODEL COMPARISON
=========================

Trains multiple classification models and compares their performance:
- Baseline Random Forest (no optimization)
- Improved Random Forest (with SMOTE)
- Logistic Regression
- Naive Bayes

Output: Comprehensive comparison metrics and trained models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import json

# ============================================================================
# STEP 1: LOAD DATA & INITIALIZATION
# ============================================================================

def load_and_prepare_data(data_path):
    """Load dataset and prepare features/target"""
    print("=" * 80)
    print("PHASE 6: MODEL COMPARISON - INITIALIZATION")
    print("=" * 80)
    
    data = pd.read_csv(data_path)
    print(f"\n✓ Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"  Features: {list(data.columns)}")
    
    X = data[['LOC', 'CBO', 'RFC', 'WMC']]
    y = data['defect']
    
    print(f"\n✓ Class distribution:")
    print(f"  Clean (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  Buggy (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split data and apply scaling"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Train-test split completed:")
    print(f"  Training set: {X_train_scaled.shape[0]} samples")
    print(f"  Testing set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# STEP 2: BASELINE MODEL (Random Forest - No Optimization)
# ============================================================================

def train_baseline_rf(X_train, X_test, y_train, y_test):
    """Train baseline Random Forest without any optimization"""
    print("\n" + "=" * 80)
    print("MODEL 1: BASELINE RANDOM FOREST (No Optimization)")
    print("=" * 80)
    
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Baseline RF'
    metrics['model_type'] = 'RandomForest'
    
    print("\n✓ Baseline Random Forest trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 3: IMPROVED MODEL (Random Forest with SMOTE)
# ============================================================================

def train_improved_rf(X_train, X_test, y_train, y_test):
    """Train improved Random Forest with SMOTE and hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("MODEL 2: IMPROVED RANDOM FOREST (with SMOTE & Tuning)")
    print("=" * 80)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"✓ SMOTE applied:")
    print(f"  Before: {(y_train == 0).sum()} clean, {(y_train == 1).sum()} buggy")
    print(f"  After: {(y_train_balanced == 0).sum()} clean, {(y_train_balanced == 1).sum()} buggy")
    
    # Train with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Improved RF'
    metrics['model_type'] = 'RandomForest'
    
    print("\n✓ Improved Random Forest trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 4: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model"""
    print("\n" + "=" * 80)
    print("MODEL 3: LOGISTIC REGRESSION")
    print("=" * 80)
    
    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Logistic Regression'
    metrics['model_type'] = 'Linear'
    
    print("\n✓ Logistic Regression trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 5: NAIVE BAYES
# ============================================================================

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Train Gaussian Naive Bayes model"""
    print("\n" + "=" * 80)
    print("MODEL 4: GAUSSIAN NAIVE BAYES")
    print("=" * 80)
    
    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    model = GaussianNB()
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Naive Bayes'
    metrics['model_type'] = 'Probabilistic'
    
    print("\n✓ Gaussian Naive Bayes trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 6: EVALUATION FUNCTION
# ============================================================================

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Comprehensive model evaluation using multiple metrics
    
    Returns dict with metrics:
    - Accuracy: Overall correctness
    - Precision: True positives / Predicted positives
    - Recall: True positives / Actual positives (IMPORTANT for bug detection)
    - F1-Score: Harmonic mean of precision and recall
    - AUC-ROC: Area under ROC curve
    """
    from sklearn.metrics import roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'cm': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # Calculate AUC-ROC if probabilities available
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


# ============================================================================
# STEP 7: PRINT DETAILED COMPARISON
# ============================================================================

def print_model_comparison(all_metrics):
    """Print detailed comparison of all models"""
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    
    df_metrics = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1']:.4f}",
            'AUC-ROC': f"{m.get('auc_roc', 0):.4f}",
        }
        for m in all_metrics
    ])
    
    print("\n" + str(df_metrics.to_string(index=False)))
    
    # Highlight key improvements
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    baseline_recall = all_metrics[0]['recall']
    improved_recall = all_metrics[1]['recall']
    
    print(f"\n✓ Recall Improvement (Bug Detection):")
    print(f"  Baseline RF: {baseline_recall:.4f}")
    print(f"  Improved RF: {improved_recall:.4f}")
    print(f"  Improvement: +{(improved_recall - baseline_recall):.4f} ({(improved_recall - baseline_recall) / baseline_recall * 100:.1f}%)")
    
    # Find best model for different priorities
    by_recall = max(all_metrics, key=lambda x: x['recall'])
    by_f1 = max(all_metrics, key=lambda x: x['f1'])
    
    print(f"\n✓ Best Model by Recall (Bug Detection): {by_recall['model_name']} ({by_recall['recall']:.4f})")
    print(f"✓ Best Model by F1-Score (Balance): {by_f1['model_name']} ({by_f1['f1']:.4f})")
    
    return df_metrics


# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

def save_models_and_results(models_dict, metrics_list, scaler, output_dir='models'):
    """Save all trained models and results"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    for name, model in models_dict.items():
        joblib.dump(model, f"{output_dir}/{name.replace(' ', '_')}.pkl")
    
    # Save scaler
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    # Save metrics as JSON
    with open(f"{output_dir}/comparison_metrics.json", 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"\n✓ Models saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete model comparison pipeline"""
    
    # Load and prepare data
    X, y = load_and_prepare_data("data/processed/cleaned_dataset.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    # Train all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    models_dict = {}
    predictions_dict = {}
    all_metrics = []
    
    # Model 1: Baseline
    model_1, metrics_1, pred_1, proba_1 = train_baseline_rf(X_train, X_test, y_train, y_test)
    models_dict['Baseline RF'] = model_1
    predictions_dict['Baseline RF'] = {'pred': pred_1, 'proba': proba_1}
    all_metrics.append(metrics_1)
    
    # Model 2: Improved
    model_2, metrics_2, pred_2, proba_2 = train_improved_rf(X_train, X_test, y_train, y_test)
    models_dict['Improved RF'] = model_2
    predictions_dict['Improved RF'] = {'pred': pred_2, 'proba': proba_2}
    all_metrics.append(metrics_2)
    
    # Model 3: Logistic Regression
    model_3, metrics_3, pred_3, proba_3 = train_logistic_regression(X_train, X_test, y_train, y_test)
    models_dict['Logistic Regression'] = model_3
    predictions_dict['Logistic Regression'] = {'pred': pred_3, 'proba': proba_3}
    all_metrics.append(metrics_3)
    
    # Model 4: Naive Bayes
    model_4, metrics_4, pred_4, proba_4 = train_naive_bayes(X_train, X_test, y_train, y_test)
    models_dict['Naive Bayes'] = model_4
    predictions_dict['Naive Bayes'] = {'pred': pred_4, 'proba': proba_4}
    all_metrics.append(metrics_4)
    
    # Compare models
    df_comparison = print_model_comparison(all_metrics)
    
    # Save results
    save_models_and_results(models_dict, all_metrics, scaler)
    
    # Return for Phase 7 (Visualization)
    return {
        'metrics': all_metrics,
        'predictions': predictions_dict,
        'models': models_dict,
        'y_test': y_test,
        'X_test': X_test,
        'scaler': scaler,
        'comparison_df': df_comparison
    }


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 80)
    print("✓ PHASE 6 COMPLETED: Model Comparison")
    print("=" * 80)
