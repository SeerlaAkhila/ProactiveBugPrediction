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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss
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
    print(f"\n? Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"  Features: {list(data.columns)}")
    
    X = data[['LOC', 'CBO', 'RFC', 'WMC']]
    y = data['defect']
    
    print(f"\n? Class distribution:")
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
    
    print(f"\n? Train-test split completed:")
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
    
    print("\n? Baseline Random Forest trained successfully")
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
    
    print(f"? SMOTE applied:")
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
    
    print("\n? Improved Random Forest trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 4: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model"""
    print("\n" + "=" * 80)
    print("MODEL 3: LOGISTIC REGRESSION")
    print("=" * 80)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Logistic Regression'
    metrics['model_type'] = 'Linear'
    
    print("\n? Logistic Regression trained successfully")
    return model, metrics, y_pred, y_pred_proba


# ============================================================================
# STEP 5: NAIVE BAYES
# ============================================================================

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Train Gaussian Naive Bayes model"""
    print("\n" + "=" * 80)
    print("MODEL 4: GAUSSIAN NAIVE BAYES")
    print("=" * 80)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    metrics['model_name'] = 'Naive Bayes'
    metrics['model_type'] = 'Probabilistic'
    
    print("\n? Gaussian Naive Bayes trained successfully")
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
    - Brier Score: Probability calibration quality (lower is better)
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
        metrics['brier'] = brier_score_loss(y_true, y_pred_proba)
    
    return metrics


# ============================================================================
# STEP 7: STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================

def get_model_specs(random_state=42):
    """Return model configurations used across holdout and CV evaluation."""
    return {
        'Baseline RF': {
            'model_type': 'RandomForest',
            'use_smote': False,
            'estimator': RandomForestClassifier(random_state=random_state, n_jobs=-1)
        },
        'Improved RF': {
            'model_type': 'RandomForest',
            'use_smote': True,
            'estimator': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        },
        'Logistic Regression': {
            'model_type': 'Linear',
            'use_smote': False,
            'estimator': LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        },
        'Naive Bayes': {
            'model_type': 'Probabilistic',
            'use_smote': False,
            'estimator': GaussianNB()
        }
    }


def tune_decision_threshold(y_true, y_proba, optimize_for='f1'):
    """
    Tune decision threshold on validation probabilities.
    """
    if optimize_for not in {'accuracy', 'precision', 'recall', 'f1'}:
        raise ValueError(f"Unsupported optimization metric: {optimize_for}")

    candidate_thresholds = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 37),
        np.round(y_proba, 6),
        np.array([0.5])
    ]))

    best_threshold = 0.5
    best_score = None
    for threshold in candidate_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        score = (
            metrics[optimize_for],
            metrics['recall'],
            metrics['precision'],
            -abs(float(threshold) - 0.5)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def run_stratified_cross_validation(
    X,
    y,
    n_splits=5,
    random_state=42,
    inner_val_size=0.2,
    threshold_metric='f1'
):
    """
    Evaluate all models using stratified k-fold CV.

    Per fold order:
    Split -> Scale -> SMOTE (train only) -> Train -> Evaluate
    """
    print("\n" + "=" * 80)
    print("STRATIFIED K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(
        f"Using {n_splits}-fold stratified CV with fold-wise scaling, "
        "model-specific imbalance handling, Platt calibration, and tuned thresholds"
    )

    model_specs = get_model_specs(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    tracked_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'brier', 'threshold']
    fold_scores = {
        model_name: {metric: [] for metric in tracked_metrics}
        for model_name in model_specs
    }

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_outer = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Inner validation split for threshold tuning (production-aligned behavior)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=inner_val_size,
            random_state=random_state,
            stratify=y_train_outer
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        for model_name, spec in model_specs.items():
            X_fit, y_fit = X_train_scaled, y_train

            if spec['use_smote']:
                smote = SMOTE(random_state=random_state)
                X_fit, y_fit = smote.fit_resample(X_train_scaled, y_train)

            base_model = clone(spec['estimator'])
            base_model.fit(X_fit, y_fit)

            # Platt scaling with CalibratedClassifierCV (sigmoid).
            # Use prefit mode when available; fallback to cv=3 for compatibility.
            calibrated_model = None
            try:
                calibrated_model = CalibratedClassifierCV(
                    estimator=base_model,
                    method='sigmoid',
                    cv='prefit'
                )
                calibrated_model.fit(X_val_scaled, y_val)
            except TypeError:
                calibrated_model = CalibratedClassifierCV(
                    base_estimator=base_model,
                    method='sigmoid',
                    cv='prefit'
                )
                calibrated_model.fit(X_val_scaled, y_val)
            except Exception:
                try:
                    calibrated_model = CalibratedClassifierCV(
                        estimator=clone(spec['estimator']),
                        method='sigmoid',
                        cv=3
                    )
                    calibrated_model.fit(X_fit, y_fit)
                except TypeError:
                    calibrated_model = CalibratedClassifierCV(
                        base_estimator=clone(spec['estimator']),
                        method='sigmoid',
                        cv=3
                    )
                    calibrated_model.fit(X_fit, y_fit)

            y_val_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
            tuned_threshold = tune_decision_threshold(
                y_val,
                y_val_proba,
                optimize_for=threshold_metric
            )

            y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_proba >= tuned_threshold).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_proba)

            for metric in tracked_metrics:
                if metric == 'threshold':
                    fold_scores[model_name][metric].append(float(tuned_threshold))
                else:
                    fold_scores[model_name][metric].append(float(metrics.get(metric, np.nan)))

        print(f"? Completed fold {fold_idx}/{n_splits}")

    cv_summary = {}
    for model_name, metrics_dict in fold_scores.items():
        cv_summary[model_name] = {}
        for metric, values in metrics_dict.items():
            values_arr = np.array(values, dtype=float)
            cv_summary[model_name][f'{metric}_mean'] = float(np.nanmean(values_arr))
            cv_summary[model_name][f'{metric}_std'] = float(np.nanstd(values_arr))

    return cv_summary


# ============================================================================
# STEP 8: PRINT DETAILED COMPARISON
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
            'Brier': f"{m.get('brier', 0):.4f}",
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
    
    print(f"\n? Recall Improvement (Bug Detection):")
    print(f"  Baseline RF: {baseline_recall:.4f}")
    print(f"  Improved RF: {improved_recall:.4f}")
    print(f"  Improvement: +{(improved_recall - baseline_recall):.4f} ({(improved_recall - baseline_recall) / baseline_recall * 100:.1f}%)")
    
    # Find best model for different priorities
    by_recall = max(all_metrics, key=lambda x: x['recall'])
    by_f1 = max(all_metrics, key=lambda x: x['f1'])
    
    print(f"\n? Best Model by Recall (Bug Detection): {by_recall['model_name']} ({by_recall['recall']:.4f})")
    print(f"? Best Model by F1-Score (Balance): {by_f1['model_name']} ({by_f1['f1']:.4f})")
    
    return df_metrics


# ============================================================================
# STEP 9: PRINT CROSS-VALIDATION COMPARISON
# ============================================================================

def print_cross_validation_comparison(cv_summary):
    """Print CV mean +/- std comparison table and best model by recall + stability."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION MODEL COMPARISON (MEAN +/- STD)")
    print("=" * 80)

    rows = []
    for model_name, m in cv_summary.items():
        rows.append({
            'Model': model_name,
            'Accuracy': f"{m['accuracy_mean']:.4f} +/- {m['accuracy_std']:.4f}",
            'Precision': f"{m['precision_mean']:.4f} +/- {m['precision_std']:.4f}",
            'Recall': f"{m['recall_mean']:.4f} +/- {m['recall_std']:.4f}",
            'F1-Score': f"{m['f1_mean']:.4f} +/- {m['f1_std']:.4f}",
            'AUC-ROC': f"{m['auc_roc_mean']:.4f} +/- {m['auc_roc_std']:.4f}",
            'Brier': f"{m['brier_mean']:.4f} +/- {m['brier_std']:.4f}",
            'Threshold': f"{m['threshold_mean']:.4f} +/- {m['threshold_std']:.4f}",
        })

    df_cv = pd.DataFrame(rows)
    print("\n" + str(df_cv.to_string(index=False)))

    # Select best model by defect-detection priority:
    # 1) higher recall mean, 2) lower recall std (stability), 3) higher F1 mean.
    best_name, best_metrics = max(
        cv_summary.items(),
        key=lambda item: (item[1]['recall_mean'], -item[1]['recall_std'], item[1]['f1_mean'])
    )

    print("\n" + "=" * 80)
    print("CV-BASED RECOMMENDATION")
    print("=" * 80)
    print(
        f"Best model by Recall + Stability: {best_name} "
        f"(Recall={best_metrics['recall_mean']:.4f} +/- {best_metrics['recall_std']:.4f}, "
        f"F1={best_metrics['f1_mean']:.4f} +/- {best_metrics['f1_std']:.4f})"
    )

    return df_cv


# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

def save_models_and_results(models_dict, metrics_list, scaler, cv_summary=None, output_dir='models'):
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

    # Save cross-validation summary as JSON
    if cv_summary is not None:
        with open(f"{output_dir}/cross_validation_metrics.json", 'w') as f:
            json.dump(cv_summary, f, indent=2)
    
    print(f"\n? Models saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete model comparison pipeline"""
    
    # Load and prepare data
    X, y = load_and_prepare_data("data/processed/cleaned_dataset.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Cross-validation evaluation for reliable model ranking
    cv_summary = run_stratified_cross_validation(X, y, n_splits=5, random_state=42)
    
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
    df_cv = print_cross_validation_comparison(cv_summary)
    
    # Save results
    save_models_and_results(models_dict, all_metrics, scaler, cv_summary=cv_summary)
    
    # Return for Phase 7 (Visualization)
    return {
        'metrics': all_metrics,
        'predictions': predictions_dict,
        'models': models_dict,
        'y_test': y_test,
        'X_test': X_test,
        'scaler': scaler,
        'comparison_df': df_comparison,
        'cv_summary': cv_summary,
        'cv_df': df_cv
    }


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 80)
    print("? PHASE 6 COMPLETED: Model Comparison")
    print("=" * 80)

