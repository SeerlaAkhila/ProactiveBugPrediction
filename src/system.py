"""
PHASE 9: SYSTEM DESIGN - MODULAR ARCHITECTURE
==============================================

Implements a complete, modular Bug Prediction System with:
1. DataPreprocessor - Handles data loading, cleaning, scaling
2. ModelTrainer - Trains multiple models with SMOTE
3. ModelEvaluator - Comprehensive evaluation metrics
4. RiskClassifier - Risk categorization
5. PredictionEngine - Single/batch predictions
6. SystemOrchestrator - Coordinates all modules

Architecture:
    ┌─────────────────────────────┐
    │   SystemOrchestrator        │
    │  (Main Coordination Hub)    │
    └──────────┬──────────────────┘
               │
      ┌────────┴────────┬────────────┬────────────┬──────────┐
      │                 │            │            │          │
    ┌─┴─┐        ┌──────┴───┐   ┌────┴───┐  ┌────┴──┐  ┌────┴─┐
    │DP │        │  Model   │   │ Eval   │  │ Risk  │  │Predict│
    │   │        │ Trainer  │   │uator   │  │Class  │  │Engine │
    └───┘        └──────────┘   └────────┘  └───────┘  └───────┘
     
Features:
- Consistent API across all modules
- Error handling and logging
- Model persistence (save/load)
- Batch processing support
- Configurable parameters
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, brier_score_loss
)
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name, log_dir='logs'):
    """Setup logging for the system"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(f'{log_dir}/{name}.log', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# ============================================================================
# MODULE 1: DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """
    Handles all data loading, cleaning, and preprocessing tasks
    
    Responsibilities:
    - Load dataset from CSV
    - Handle missing values
    - Feature extraction & validation
    - Train-validation-test split
    - Feature scaling
    """
    
    REQUIRED_FEATURES = ['LOC', 'CBO', 'RFC', 'WMC', 'defect']
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42, scale=True):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scale = scale
        self.scaler = None
        self.logger = setup_logger('DataPreprocessor')
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV"""
        try:
            data = pd.read_csv(filepath)
            self.logger.info(f"[OK] Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
            return data
        except Exception as e:
            self.logger.error(f"✗ Error loading data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate dataset has required features"""
        missing = set(self.REQUIRED_FEATURES) - set(data.columns)
        if missing:
            self.logger.error(f"[ERROR] Missing features: {missing}")
            return False
        self.logger.info("[OK] Data validation passed")
        return True
    
    def prepare_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features/target and enforce basic quality constraints."""
        feature_cols = ['LOC', 'CBO', 'RFC', 'WMC']
        X = data[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(data['defect'], errors='coerce')

        valid_mask = X.notna().all(axis=1) & y.isin([0, 1])
        invalid_rows = int((~valid_mask).sum())
        if invalid_rows:
            self.logger.warning(f"[WARN] Dropping {invalid_rows} row(s) with missing/non-numeric values")

        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].astype(int).copy()

        # Negative static code metrics are usually invalid in this domain.
        non_negative_mask = (X >= 0).all(axis=1)
        negative_rows = int((~non_negative_mask).sum())
        if negative_rows:
            self.logger.warning(f"[WARN] Dropping {negative_rows} row(s) with negative metric values")
            X = X.loc[non_negative_mask].copy()
            y = y.loc[non_negative_mask].copy()
        
        self.logger.info(f"[OK] Features prepared: {X.shape}")
        self.logger.info(f"  Class distribution - Clean: {(y == 0).sum()}, Buggy: {(y == 1).sum()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split into train, validation, and test sets"""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_train_val
        )

        self.logger.info("[OK] Train-val-test split completed:")
        self.logger.info(f"  Training: {X_train.shape[0]} samples")
        self.logger.info(f"  Validation: {X_val.shape[0]} samples")
        self.logger.info(f"  Testing: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess(self, filepath: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        # Load and validate
        data = self.load_data(filepath)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Prepare features
        X, y = self.prepare_features_and_target(data)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Scale
        if self.scale:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            X_test_scaled = X_test.values
        
        feature_ranges = {
            col: {'min': float(X[col].min()), 'max': float(X[col].max())}
            for col in X.columns
        }

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_full': X,
            'y_full': y,
            'scaler': self.scaler,
            'feature_names': ['LOC', 'CBO', 'RFC', 'WMC'],
            'feature_ranges': feature_ranges
        }


# ============================================================================
# MODULE 2: MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """
    Trains multiple classification models with SMOTE for class imbalance
    
    Responsibilities:
    - Train Baseline Random Forest
    - Train Improved Random Forest (with SMOTE)
    - Train Logistic Regression
    - Train Naive Bayes
    - Apply SMOTE when needed
    """
    
    def __init__(self, random_state=42, smote_models=None):
        self.random_state = random_state
        self.logger = setup_logger('ModelTrainer')
        self.smote = None
        self.smote_models = set(smote_models or {'Improved RF'})
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """Apply SMOTE for class balancing"""
        self.smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = self.smote.fit_resample(X_train, y_train)
        
        self.logger.info(f"[OK] SMOTE applied:")
        self.logger.info(f"  Before: {(y_train == 0).sum()} clean, {(y_train == 1).sum()} buggy")
        self.logger.info(f"  After: {(y_balanced == 0).sum()} clean, {(y_balanced == 1).sum()} buggy")
        
        return X_balanced, y_balanced
    
    def train_baseline_rf(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Baseline Random Forest"""
        model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        model.fit(X_train, y_train)
        self.logger.info("[OK] Baseline Random Forest trained")
        return model
    
    def train_improved_rf(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Improved Random Forest with SMOTE"""
        if 'Improved RF' in self.smote_models:
            X_balanced, y_balanced = self.apply_smote(X_train, y_train)
        else:
            X_balanced, y_balanced = X_train, y_train
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_balanced, y_balanced)
        self.logger.info("[OK] Improved Random Forest trained")
        return model
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Logistic Regression with class balancing."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.logger.info("[OK] Logistic Regression trained")
        return model
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Gaussian Naive Bayes (without synthetic oversampling)."""
        model = GaussianNB()
        model.fit(X_train, y_train)
        self.logger.info("[OK] Gaussian Naive Bayes trained")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all models"""
        self.logger.info("Starting model training...")
        
        models = {
            'Baseline RF': self.train_baseline_rf(X_train, y_train),
            'Improved RF': self.train_improved_rf(X_train, y_train),
            'Logistic Regression': self.train_logistic_regression(X_train, y_train),
            'Naive Bayes': self.train_naive_bayes(X_train, y_train),
        }
        
        self.logger.info(f"[OK] All models trained: {len(models)} models")
        return models

    def train_model_by_name(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Train one supported model by display name."""
        trainers = {
            'Baseline RF': self.train_baseline_rf,
            'Improved RF': self.train_improved_rf,
            'Logistic Regression': self.train_logistic_regression,
            'Naive Bayes': self.train_naive_bayes,
        }

        if model_name not in trainers:
            raise ValueError(f"Unsupported model for training: {model_name}")

        return trainers[model_name](X_train, y_train)


# ============================================================================
# MODULE 3: MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """
    Evaluates models using comprehensive metrics
    
    Metrics:
    - Accuracy
    - Precision
    - Recall (critical for bug detection)
    - F1-Score
    - AUC-ROC
    - Confusion Matrix
    """
    
    def __init__(self):
        self.logger = setup_logger('ModelEvaluator')
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
        """Evaluate a single model"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'cm': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['brier'] = brier_score_loss(y_true, y_proba)
        
        return metrics

    def _fit_platt_scaler(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Fit a lightweight Platt-scaling calibrator on validation probabilities.
        """
        proba = np.asarray(y_proba).reshape(-1, 1)
        y = np.asarray(y_true).astype(int)

        if len(np.unique(y)) < 2 or len(np.unique(np.round(proba, 6))) < 5:
            return None

        calibrator = LogisticRegression(
            random_state=42,
            max_iter=500,
            class_weight='balanced'
        )
        calibrator.fit(proba, y)
        return calibrator

    def _apply_platt_scaler(self, calibrator, y_proba: np.ndarray) -> np.ndarray:
        """Apply fitted Platt scaler to probabilities."""
        if calibrator is None:
            return y_proba
        proba = np.asarray(y_proba).reshape(-1, 1)
        return calibrator.predict_proba(proba)[:, 1]

    def tune_decision_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        optimize_for: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Find a probability threshold that improves classification decisions.

        The default objective is F1 to balance recall and precision while still
        surfacing threshold metadata for downstream consumers.
        """
        if y_proba is None:
            return {
                'threshold': None,
                'optimized_for': optimize_for,
                'metrics': None
            }

        if optimize_for not in {'accuracy', 'precision', 'recall', 'f1'}:
            raise ValueError(f"Unsupported optimization metric: {optimize_for}")

        candidate_thresholds = np.unique(np.concatenate([
            np.linspace(0.05, 0.95, 37),
            np.round(y_proba, 6),
            np.array([0.5])
        ]))

        best_result = None
        for threshold in candidate_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self.evaluate(y_true, y_pred, y_proba)
            metrics['threshold'] = float(threshold)

            score = (
                metrics[optimize_for],
                metrics['recall'],
                metrics['precision'],
                -abs(threshold - 0.5)
            )

            if best_result is None or score > best_result['score']:
                best_result = {
                    'threshold': float(threshold),
                    'optimized_for': optimize_for,
                    'metrics': metrics,
                    'score': score
                }

        return {
            'threshold': best_result['threshold'],
            'optimized_for': best_result['optimized_for'],
            'metrics': best_result['metrics']
        }
    
    def evaluate_all_models(
        self,
        models: Dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimize_threshold_for: str = 'f1',
        calibrate_probabilities: bool = True
    ) -> Dict:
        """Evaluate all trained models"""
        self.logger.info("Evaluating all models...")
        
        results = {}
        for model_name, model in models.items():
            y_proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            default_y_pred = model.predict(X_test)
            default_metrics = self.evaluate(y_test, default_y_pred, y_proba_test)

            if y_proba_val is not None and y_proba_test is not None:
                calibrator = None
                calibrated_val = y_proba_val
                calibrated_test = y_proba_test
                if calibrate_probabilities:
                    calibrator = self._fit_platt_scaler(y_val, y_proba_val)
                    calibrated_val = self._apply_platt_scaler(calibrator, y_proba_val)
                    calibrated_test = self._apply_platt_scaler(calibrator, y_proba_test)

                threshold_info = self.tune_decision_threshold(
                    y_val,
                    calibrated_val,
                    optimize_for=optimize_threshold_for
                )
                tuned_threshold = threshold_info['threshold']
                tuned_y_pred_test = (calibrated_test >= tuned_threshold).astype(int)
                metrics = self.evaluate(y_test, tuned_y_pred_test, calibrated_test)
                metrics['default_metrics'] = default_metrics
                metrics['default_threshold'] = 0.5
                metrics['decision_threshold'] = tuned_threshold
                metrics['threshold_optimized_for'] = threshold_info['optimized_for']
                metrics['threshold_selected_on'] = 'validation'
                metrics['probability_calibrated'] = bool(calibrator is not None)
                metrics['threshold_gain'] = {
                    key: metrics[key] - default_metrics[key]
                    for key in ['accuracy', 'precision', 'recall', 'f1']
                }
            else:
                metrics = dict(default_metrics)
                metrics['default_metrics'] = dict(default_metrics)
                metrics['default_threshold'] = None
                metrics['decision_threshold'] = None
                metrics['threshold_optimized_for'] = None
                metrics['threshold_selected_on'] = None
                metrics['probability_calibrated'] = False
                metrics['threshold_gain'] = {
                    key: 0.0 for key in ['accuracy', 'precision', 'recall', 'f1']
                }

            results[model_name] = metrics
            
            self.logger.info(
                f"[OK] {model_name}: Accuracy={metrics['accuracy']:.4f}, "
                f"Recall={metrics['recall']:.4f}, Threshold={metrics['decision_threshold']}"
            )
        
        return results

    def evaluate_with_stratified_kfold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        trainer: ModelTrainer,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Cross-validated model evaluation to reduce single-split variance.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores = {}

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for model_name in ['Baseline RF', 'Improved RF', 'Logistic Regression', 'Naive Bayes']:
                model = trainer.train_model_by_name(model_name, X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                metrics = self.evaluate(y_test, y_pred, y_proba)

                if model_name not in fold_scores:
                    fold_scores[model_name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc_roc': []}
                for key in ['accuracy', 'precision', 'recall', 'f1']:
                    fold_scores[model_name][key].append(metrics[key])
                fold_scores[model_name]['auc_roc'].append(metrics.get('auc_roc', np.nan))

            self.logger.info(f"[OK] Completed CV fold {fold_idx}/{n_splits}")

        summary = {}
        for model_name, scores in fold_scores.items():
            summary[model_name] = {}
            for metric_name, values in scores.items():
                arr = np.asarray(values, dtype=float)
                summary[model_name][f'{metric_name}_mean'] = float(np.nanmean(arr))
                summary[model_name][f'{metric_name}_std'] = float(np.nanstd(arr))

        return summary


# ============================================================================
# MODULE 4: RISK CLASSIFIER
# ============================================================================

class RiskClassifier:
    """
    Classifies predictions into risk categories:
    - LOW (0.0 - 0.3)
    - MEDIUM (0.3 - 0.7)
    - HIGH (0.7 - 1.0)
    """
    
    THRESHOLDS = {
        'low': 0.3,
        'medium': 0.7,
        'high': 1.0
    }
    
    def __init__(self):
        self.logger = setup_logger('RiskClassifier')
    
    def classify(self, probability: float) -> str:
        """Classify single probability to risk level"""
        if probability < self.THRESHOLDS['low']:
            return 'LOW'
        elif probability < self.THRESHOLDS['medium']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def classify_batch(self, probabilities: np.ndarray) -> List[str]:
        """Classify batch of probabilities"""
        return [self.classify(p) for p in probabilities]
    
    def get_risk_distribution(self, probabilities: np.ndarray) -> Dict:
        """Get distribution of risk levels"""
        risks = self.classify_batch(probabilities)
        total = len(risks)
        
        return {
            'low': risks.count('LOW'),
            'medium': risks.count('MEDIUM'),
            'high': risks.count('HIGH'),
            'low_pct': risks.count('LOW') / total * 100,
            'medium_pct': risks.count('MEDIUM') / total * 100,
            'high_pct': risks.count('HIGH') / total * 100,
        }


# ============================================================================
# MODULE 5: PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """
    Unified prediction interface for single and batch predictions
    
    Features:
    - Single sample prediction
    - Batch predictions
    - Probability estimates
    - Risk classification
    """
    
    def __init__(
        self,
        model,
        scaler,
        risk_classifier=None,
        decision_threshold=0.5,
        model_name=None,
        feature_names=None,
        feature_ranges=None,
        ood_z_threshold=4.0
    ):
        self.model = model
        self.scaler = scaler
        self.risk_classifier = risk_classifier or RiskClassifier()
        self.decision_threshold = float(decision_threshold) if decision_threshold is not None else 0.5
        self.model_name = model_name
        self.feature_names = feature_names or ['LOC', 'CBO', 'RFC', 'WMC']
        self.feature_ranges = feature_ranges or {}
        self.ood_z_threshold = float(ood_z_threshold)
        self.logger = setup_logger('PredictionEngine')

    def _assess_input_distribution(self, X_raw: np.ndarray) -> Dict[str, Any]:
        """Assess whether raw input appears outside training distribution."""
        warnings = []
        out_of_range_features = []

        for idx, feature in enumerate(self.feature_names):
            if feature not in self.feature_ranges:
                continue
            value = float(X_raw[0, idx])
            min_val = float(self.feature_ranges[feature]['min'])
            max_val = float(self.feature_ranges[feature]['max'])
            if value < min_val or value > max_val:
                out_of_range_features.append(feature)

        if out_of_range_features:
            warnings.append(
                "Input is outside training distribution; prediction may be unreliable"
            )

        max_abs_zscore = None
        if self.scaler is not None:
            z_values = self.scaler.transform(X_raw)
            max_abs_zscore = float(np.max(np.abs(z_values)))
            if max_abs_zscore > self.ood_z_threshold:
                warnings.append(
                    f"Input is far from training centroid (max |z|={max_abs_zscore:.2f})"
                )

        return {
            'out_of_distribution': bool(warnings),
            'ood_warnings': warnings,
            'out_of_range_features': out_of_range_features,
            'max_abs_zscore': max_abs_zscore
        }
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict for single sample"""
        # Reshape for single sample
        X_raw = np.asarray(features, dtype=float).reshape(1, -1)
        reliability = self._assess_input_distribution(X_raw)
        X = X_raw
        
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        probability = self.model.predict_proba(X)[0][1] if hasattr(self.model, 'predict_proba') else None
        if probability is not None:
            prediction = int(probability >= self.decision_threshold)
            risk = self.risk_classifier.classify(probability)
        else:
            prediction = int(self.model.predict(X)[0])
            risk = None
        
        return {
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'risk_level': risk,
            'label': 'Buggy' if prediction == 1 else 'Clean',
            'decision_threshold': self.decision_threshold,
            'out_of_distribution': reliability['out_of_distribution'],
            'ood_warnings': reliability['ood_warnings'],
            'out_of_range_features': reliability['out_of_range_features'],
            'max_abs_zscore': reliability['max_abs_zscore']
        }
    
    def predict_batch(self, X: np.ndarray) -> pd.DataFrame:
        """Predict for batch of samples"""
        X_raw = np.asarray(X, dtype=float)
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X_raw)
        else:
            X = X_raw
        
        probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        if probabilities is not None:
            predictions = (probabilities >= self.decision_threshold).astype(int)
            risks = self.risk_classifier.classify_batch(probabilities)
        else:
            predictions = self.model.predict(X)
            risks = None
        
        batch_results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': risks,
            'label': ['Buggy' if p == 1 else 'Clean' for p in predictions],
            'decision_threshold': self.decision_threshold
        })

        if self.feature_ranges:
            ood_flags = []
            for row in X_raw:
                reliability = self._assess_input_distribution(row.reshape(1, -1))
                ood_flags.append(reliability['out_of_distribution'])
            batch_results['out_of_distribution'] = ood_flags

        return batch_results


# ============================================================================
# MODULE 6: SYSTEM ORCHESTRATOR
# ============================================================================

class BugPredictionSystem:
    """
    Complete Bug Prediction System - Orchestrates all modules
    
    Workflow:
    1. Data Preprocessing
    2. Model Training
    3. Model Evaluation
    4. Risk Classification
    5. Prediction & Export
    """
    
    def __init__(self, config=None):
        self.logger = setup_logger('BugPredictionSystem')
        self.config = config or {}
        
        # Initialize modules
        self.preprocessor = DataPreprocessor(
            test_size=self.config.get('test_size', 0.2),
            val_size=self.config.get('val_size', 0.2),
            random_state=self.config.get('random_state', 42),
            scale=self.config.get('scale', True)
        )
        self.trainer = ModelTrainer(
            random_state=self.config.get('random_state', 42),
            smote_models=self.config.get('smote_models', {'Improved RF'})
        )
        self.evaluator = ModelEvaluator()
        self.risk_classifier = RiskClassifier()
        
        # Storage
        self.models = {}
        self.evaluation_results = {}
        self.threshold_metadata = {}
        self.data = {}
        self.cv_results = {}
        self.scaler = None
    
    def run_complete_pipeline(self, data_path: str) -> Dict:
        """Execute complete system pipeline"""
        print("\n" + "=" * 80)
        print("BUG PREDICTION SYSTEM - COMPLETE PIPELINE")
        print("=" * 80)
        
        # Step 1: Preprocess
        print("\n[1/4] Data Preprocessing...")
        self.data = self.preprocessor.preprocess(data_path)
        self.scaler = self.data['scaler']
        
        # Step 2: Train models
        print("[2/4] Model Training...")
        self.models = self.trainer.train_all_models(self.data['X_train'], self.data['y_train'])
        
        # Step 3: Evaluate models
        print("[3/4] Model Evaluation...")
        self.evaluation_results = self.evaluator.evaluate_all_models(
            self.models,
            self.data['X_val'],
            self.data['y_val'],
            self.data['X_test'],
            self.data['y_test'],
            optimize_threshold_for=self.config.get('threshold_metric', 'f1'),
            calibrate_probabilities=self.config.get('calibrate_probabilities', True)
        )

        self.threshold_metadata = {
            model_name: {
                'decision_threshold': metrics.get('decision_threshold'),
                'default_threshold': metrics.get('default_threshold'),
                'threshold_optimized_for': metrics.get('threshold_optimized_for'),
                'threshold_selected_on': metrics.get('threshold_selected_on'),
                'threshold_gain': metrics.get('threshold_gain', {})
            }
            for model_name, metrics in self.evaluation_results.items()
        }
        
        # Step 4: Risk classification
        print("[4/4] Risk Classification...")
        risk_results = {}
        for model_name, model in self.models.items():
            y_proba = model.predict_proba(self.data['X_test'])[:, 1]
            risk_results[model_name] = self.risk_classifier.get_risk_distribution(y_proba)

        if self.config.get('run_cross_validation', True):
            print("[CV] Stratified k-fold evaluation...")
            self.cv_results = self.evaluator.evaluate_with_stratified_kfold(
                self.data['X_full'],
                self.data['y_full'],
                self.trainer,
                n_splits=self.config.get('cv_splits', 5),
                random_state=self.config.get('random_state', 42)
            )
        
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return {
            'models': self.models,
            'evaluation': self.evaluation_results,
            'risk_distribution': risk_results,
            'scaler': self.scaler,
            'cv_results': self.cv_results
        }
    
    def save_models(self, output_dir='models'):
        """Save all trained models and configuration"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = f"{output_dir}/{model_name.replace(' ', '_')}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"[OK] Saved: {filepath}")
        
        # Save scaler
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save evaluation results
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump({k: v for k, v in self.evaluation_results.items()}, f, indent=2, default=str)

        with open(f"{output_dir}/threshold_metadata.json", 'w') as f:
            json.dump(self.threshold_metadata, f, indent=2, default=str)

        if self.cv_results:
            with open(f"{output_dir}/cross_validation_results.json", 'w') as f:
                json.dump(self.cv_results, f, indent=2, default=str)
        
        self.logger.info(f"[OK] All models saved to {output_dir}/")
    
    def load_model(self, model_name: str, model_dir='models'):
        """Load a specific trained model"""
        filepath = f"{model_dir}/{model_name.replace(' ', '_')}.pkl"
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # Also load scaler if available
        try:
            with open(f"{model_dir}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
        except:
            pass

        try:
            with open(f"{model_dir}/threshold_metadata.json", 'r') as f:
                self.threshold_metadata = json.load(f)
        except:
            pass
        
        return model
    
    def get_prediction_engine(self, model_name: str) -> PredictionEngine:
        """Get prediction engine for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return PredictionEngine(
            model=self.models[model_name],
            scaler=self.scaler,
            risk_classifier=self.risk_classifier,
            decision_threshold=self.threshold_metadata.get(model_name, {}).get('decision_threshold', 0.5),
            model_name=model_name,
            feature_names=self.data.get('feature_names', ['LOC', 'CBO', 'RFC', 'WMC']),
            feature_ranges=self.data.get('feature_ranges', {})
        )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize system
    system = BugPredictionSystem()
    
    # Run complete pipeline
    results = system.run_complete_pipeline('data/processed/cleaned_dataset.csv')
    
    # Save models
    system.save_models()
    
    # Get prediction engine
    predictor = system.get_prediction_engine('Improved RF')
    
    # Example: Predict on new data
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    print("\nSingle prediction example:")
    sample = np.array([100, 5, 15, 10])  # [LOC, CBO, RFC, WMC]
    result = predictor.predict_single(sample)
    print(f"  Features: LOC=100, CBO=5, RFC=15, WMC=10")
    print(f"  Prediction: {result['label']}")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    
    print("\n✓ PHASE 9 COMPLETED: System Design")
