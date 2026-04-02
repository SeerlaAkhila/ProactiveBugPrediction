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
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
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
    - Train-test split
    - Feature scaling
    """
    
    REQUIRED_FEATURES = ['LOC', 'CBO', 'RFC', 'WMC', 'defect']
    
    def __init__(self, test_size=0.2, random_state=42, scale=True):
        self.test_size = test_size
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
        """Extract features and target variable"""
        feature_cols = ['LOC', 'CBO', 'RFC', 'WMC']
        X = data[feature_cols]
        y = data['defect']
        
        self.logger.info(f"[OK] Features prepared: {X.shape}")
        self.logger.info(f"  Class distribution - Clean: {(y == 0).sum()}, Buggy: {(y == 1).sum()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"[OK] Train-test split completed:")
        self.logger.info(f"  Training: {X_train.shape[0]} samples")
        self.logger.info(f"  Testing: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """Scale features using StandardScaler"""
        if not self.scale:
            return X_train.values, X_test.values
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info("[OK] Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, filepath: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        # Load and validate
        data = self.load_data(filepath)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Prepare features
        X, y = self.prepare_features_and_target(data)
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': ['LOC', 'CBO', 'RFC', 'WMC']
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
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.logger = setup_logger('ModelTrainer')
        self.smote = None
    
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
        model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        self.logger.info("[OK] Baseline Random Forest trained")
        return model
    
    def train_improved_rf(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Improved Random Forest with SMOTE"""
        X_balanced, y_balanced = self.apply_smote(X_train, y_train)
        
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
        """Train Logistic Regression with SMOTE"""
        X_balanced, y_balanced = self.apply_smote(X_train, y_train)
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state, n_jobs=-1)
        model.fit(X_balanced, y_balanced)
        self.logger.info("[OK] Logistic Regression trained")
        return model
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train Gaussian Naive Bayes with SMOTE"""
        X_balanced, y_balanced = self.apply_smote(X_train, y_train)
        
        model = GaussianNB()
        model.fit(X_balanced, y_balanced)
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
        
        return metrics
    
    def evaluate_all_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all trained models"""
        self.logger.info("Evaluating all models...")
        
        results = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = self.evaluate(y_test, y_pred, y_proba)
            results[model_name] = metrics
            
            self.logger.info(f"[OK] {model_name}: Accuracy={metrics['accuracy']:.4f}, Recall={metrics['recall']:.4f}")
        
        return results


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
    
    def __init__(self, model, scaler, risk_classifier=None):
        self.model = model
        self.scaler = scaler
        self.risk_classifier = risk_classifier or RiskClassifier()
        self.logger = setup_logger('PredictionEngine')
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict for single sample"""
        # Reshape for single sample
        X = features.reshape(1, -1)
        
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1] if hasattr(self.model, 'predict_proba') else None
        risk = self.risk_classifier.classify(probability) if probability else None
        
        return {
            'prediction': int(prediction),
            'probability': float(probability) if probability else None,
            'risk_level': risk,
            'label': 'Buggy' if prediction == 1 else 'Clean'
        }
    
    def predict_batch(self, X: np.ndarray) -> pd.DataFrame:
        """Predict for batch of samples"""
        # Scale if scaler available
        if self.scaler:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        risks = self.risk_classifier.classify_batch(probabilities) if probabilities is not None else None
        
        return pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': risks,
            'label': ['Buggy' if p == 1 else 'Clean' for p in predictions]
        })


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
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.risk_classifier = RiskClassifier()
        
        # Storage
        self.models = {}
        self.evaluation_results = {}
        self.data = {}
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
            self.models, self.data['X_test'], self.data['y_test']
        )
        
        # Step 4: Risk classification
        print("[4/4] Risk Classification...")
        risk_results = {}
        for model_name, model in self.models.items():
            y_proba = model.predict_proba(self.data['X_test'])[:, 1]
            risk_results[model_name] = self.risk_classifier.get_risk_distribution(y_proba)
        
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return {
            'models': self.models,
            'evaluation': self.evaluation_results,
            'risk_distribution': risk_results,
            'scaler': self.scaler
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
        
        return model
    
    def get_prediction_engine(self, model_name: str) -> PredictionEngine:
        """Get prediction engine for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return PredictionEngine(
            model=self.models[model_name],
            scaler=self.scaler,
            risk_classifier=self.risk_classifier
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
