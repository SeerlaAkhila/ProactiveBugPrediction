# Proactive Software Bug Prediction System

## 1. Title Page

### Project Name
Proactive Software Bug Prediction System

### Tagline
A machine learning system for proactive defect risk prediction using static code metrics.

### Document Purpose
This document presents a complete technical overview of the repository for academic submission, hackathon presentation, portfolio use, and recruiter review.

---

## 2. Abstract

This project predicts whether software modules are likely to be defective by learning from object-oriented static code metrics. The repository implements a full ML workflow including data preparation, model training, model comparison, probability-based risk classification, visualization, and an interactive Streamlit application. The system operates on a processed dataset with features LOC, CBO, RFC, and WMC, and predicts a binary defect label with probability and risk tier. Multiple models are trained and compared, including Random Forest variants, Logistic Regression, and Gaussian Naive Bayes. The project includes artifact persistence, threshold metadata for inference-time decisions, and reliability-focused tests around thresholding behavior. The resulting system is intended to support risk-based testing and better QA prioritization.

---

## 3. Problem Statement

Software teams typically have limited time for code review and testing, but defect-prone modules are not uniformly distributed. Without predictive prioritization, high-risk areas may be under-tested while low-risk areas consume disproportionate effort.

This project addresses that problem by:
- Predicting defect likelihood at module level from static code metrics.
- Translating probability outputs into practical risk tiers.
- Enabling proactive testing decisions before release.

Why this matters:
- Missed defects increase production risk and maintenance cost.
- Early risk identification improves test allocation efficiency.
- Defect-risk visibility supports quality engineering decisions.

---

## 4. Objectives

- Build an end-to-end defect prediction pipeline from prepared dataset to inference.
- Compare multiple ML models for binary bug prediction.
- Provide interpretable risk classes (LOW, MEDIUM, HIGH) from model probabilities.
- Deliver an interactive UI for single-module and batch predictions.
- Persist trained models and metadata for reproducible usage.
- Include testing to improve reliability of decision-threshold behavior.

---

## 5. System Overview

The repository combines training scripts, a modular core system, and a web application layer.

High-level behavior:
- Input: module metrics (LOC, CBO, RFC, WMC).
- Processing: scaling, model inference, threshold-based decisioning.
- Output: binary defect prediction, defect probability, and risk level.

Main layers:
- Application layer: Streamlit UI in app.py.
- Core system layer: modular architecture in src/system.py.
- Analysis/reporting layer: comparison, visualization, risk scripts in src.
- Artifact layer: serialized models and JSON metrics in models.

---

## 6. Architecture / Workflow

### End-to-End Workflow

1. Data ingestion
- The processed dataset is loaded from data/processed/cleaned_dataset.csv.
- Required columns: LOC, CBO, RFC, WMC, defect.

2. Validation and cleaning
- Numeric coercion is applied to feature and target fields.
- Invalid rows (missing/non-numeric entries, negative metrics, non-binary targets) are dropped in DataPreprocessor.

3. Data split and scaling
- Split strategy in the core system: train, validation, and test.
- StandardScaler is fitted on training data and applied to validation/test data.

4. Model training
- Four classifiers are trained:
  - Baseline RF
  - Improved RF
  - Logistic Regression
  - Naive Bayes

5. Evaluation and thresholding
- Metrics computed include accuracy, precision, recall, F1, confusion matrix, and optionally AUC-ROC/Brier.
- The core system supports validation-based threshold tuning and stores threshold metadata.

6. Risk classification
- Predicted probabilities are mapped to:
  - LOW: p < 0.3
  - MEDIUM: 0.3 <= p < 0.7
  - HIGH: p >= 0.7

7. Artifact persistence
- Models and scaler are stored in models directory.
- Metrics and threshold metadata are stored as JSON.

8. User interaction
- Streamlit UI loads existing artifacts and serves predictions, model comparison, risk views, and system info.

---

## 7. Tech Stack

### Language and Runtime
- Python
- Runtime target: python-3.10.13 (runtime.txt)

### Libraries and Tools
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn
- plotly
- streamlit
- joblib
- shap

### Engineering Utilities
- logging module for subsystem logs
- json/pickle for metadata and artifact serialization
- pytest for automated reliability checks

---

## 8. Dataset Details

### Dataset Files in Repository
- data/raw/single-version-ck-oo.csv
- data/raw/bug-metrics.csv
- data/processed/cleaned_dataset.csv

### Processed Training Dataset
- File: data/processed/cleaned_dataset.csv
- Rows: 997
- Columns: LOC, CBO, RFC, WMC, defect
- Class distribution:
  - defect = 0: 791
  - defect = 1: 206

### Raw Data Characteristics
- Raw files are semicolon-separated and contain CK-style metrics and bug counts.
- Feature engineering script check.py converts bug counts into binary defect.

### Preprocessing Steps Implemented
- Column selection and renaming to standardized feature names.
- bugs to defect conversion: defect = 1 if bugs > 0 else 0.
- Feature scaling with StandardScaler in model training pipelines.
- Validation of required features in core preprocessing module.

---

## 9. Methodology

### Approach Type
Supervised binary classification on tabular software metrics.

### Design Strategy
- Train multiple baseline and improved classical ML models.
- Compare performance across multiple metrics.
- Emphasize defect-detection capability (recall-aware usage).
- Convert model probabilities into operational risk categories.

### Key Methodological Decisions Observed in Code
- Class imbalance handling:
  - SMOTE used in improved Random Forest pathway.
  - Class-weighted Logistic Regression.
  - Baseline RF and Naive Bayes trained without SMOTE in current system implementation.
- Probability-aware output handling:
  - Threshold metadata available in models/threshold_metadata.json.
- Modular architecture:
  - Distinct responsibilities for preprocessing, training, evaluation, risk, and prediction.

---

## 10. Model Details

### Models Implemented
1. Baseline Random Forest
- Purpose: reference benchmark.
- Script support: src/trainbaseline.py and src/model_comparison.py.

2. Improved Random Forest
- Purpose: imbalance-aware and tuned tree ensemble.
- Uses SMOTE in the improvement path.
- Script support: src/train_improved.py and src/model_comparison.py.

3. Logistic Regression
- Purpose: linear probabilistic baseline.
- Uses class_weight balanced.

4. Gaussian Naive Bayes
- Purpose: lightweight probabilistic baseline.

### Training Process
- Data is split with stratification.
- Features are standardized.
- Models are trained and then evaluated on held-out test data.
- The modular system also includes validation split support for threshold tuning.

### Persisted Model Artifacts Present
- models/Baseline_RF.pkl
- models/Improved_RF.pkl
- models/Logistic_Regression.pkl
- models/Naive_Bayes.pkl
- models/scaler.pkl

---

## 11. Evaluation Metrics

Metrics used across repository scripts and reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- AUC-ROC
- Brier score (implemented in src/system.py evaluation path)

Threshold metadata fields present:
- decision_threshold
- default_threshold
- threshold_optimized_for

Automated reliability tests validate threshold behavior in tests/test_thresholding_reliability.py.

---

## 12. Results and Analysis

### Hold-Out Model Results (models/comparison_metrics.json)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Baseline RF | 0.8450 | 0.6923 | 0.4390 | 0.5373 | 0.7896 |
| Improved RF | 0.8050 | 0.5263 | 0.4878 | 0.5063 | 0.7671 |
| Logistic Regression | 0.8150 | 0.5500 | 0.5366 | 0.5432 | 0.7964 |
| Naive Bayes | 0.8550 | 0.7727 | 0.4146 | 0.5397 | 0.7574 |

### Observations
- Logistic Regression has the highest recall among saved comparison metrics.
- Naive Bayes has the highest hold-out accuracy and precision in current saved metrics.
- Improved RF increases recall vs Baseline RF but with lower precision and accuracy.

### Risk Distribution Snapshot (risk_classification_report.json)

| Model | Low Risk | Medium Risk | High Risk |
|---|---:|---:|---:|
| Baseline RF | 158 | 29 | 13 |
| Improved RF | 130 | 49 | 21 |
| Logistic Regression | 62 | 116 | 22 |
| Naive Bayes | 175 | 3 | 22 |

### Strengths
- Complete end-to-end implementation from training to UI deployment.
- Multi-model comparison and multiple visualization outputs.
- Risk-centric output design for practical testing prioritization.

### Limitations
- No cross_validation_metrics.json or cross_validation_results.json artifact is currently present in models directory.
- Saved evaluation_results.json appears to contain hold-out metrics only (without richer threshold gain fields expected from the current system class).
- Test coverage is focused on threshold reliability; broader unit/integration coverage can be expanded.

---

## 13. Explainability

Explainability support is included via SHAP in the Streamlit application:
- app.py attempts to import shap and conditionally enables explainability components.
- UI documentation/comments indicate per-module contribution analysis for single prediction flows.

Current implementation note:
- SHAP is dependency-optional at runtime (graceful fallback if import fails).

---

## 14. Implementation Details

### Core Modules and Roles

1. DataPreprocessor (src/system.py)
- Loads CSV.
- Validates required columns.
- Cleans invalid rows.
- Splits into train/validation/test.
- Scales features.

2. ModelTrainer (src/system.py)
- Trains all supported models.
- Applies SMOTE in configured model pathways.

3. ModelEvaluator (src/system.py)
- Computes classification and probability metrics.
- Supports threshold tuning and optional probability calibration.

4. RiskClassifier (src/system.py, src/risk_classification.py)
- Maps probabilities to LOW, MEDIUM, HIGH risk levels.
- Produces risk-distribution summaries.

5. PredictionEngine (src/system.py)
- Single and batch prediction API.
- Includes threshold-aware labeling.
- Includes out-of-distribution style warnings using feature ranges and z-score checks.

6. BugPredictionSystem (src/system.py)
- Orchestrates full pipeline and model persistence.

7. Streamlit Frontend (app.py)
- Dashboard, single prediction, upload analysis, risk analysis, model comparison, and system information views.
- Loads saved artifacts and avoids runtime retraining when artifacts are unavailable.

### Additional Scripts
- src/model_comparison.py: phase-style all-model training and reporting.
- src/visualization.py: generates 7 model-performance visualizations.
- src/risk_classification.py: generates risk report and 3 risk visualizations.
- src/trainbaseline.py and src/train_improved.py: baseline and improved experiments.
- check.py: raw-to-processed dataset preparation utility.

---

## 15. Challenges Faced

Based on implemented logic and repository structure, key technical challenges addressed include:
- Class imbalance in defect labels (handled with SMOTE and class weighting strategies).
- Trade-offs between recall, precision, and accuracy across models.
- Need for operationally meaningful outputs beyond binary labels (risk tiers).
- Artifact consistency between training scripts and UI loading requirements.
- Reliability of threshold usage (covered by dedicated threshold tests).

---

## 16. Future Enhancements

- Add and persist cross-validation result artifacts in the main training flow used for deployment.
- Align all training/evaluation scripts to one authoritative pipeline output format.
- Expand automated testing to include preprocessing validation, artifact loading, and batch prediction contracts.
- Add calibration diagnostics and visual calibration reports when thresholding is used.
- Introduce model/version registry and experiment tracking for reproducibility at scale.
- Expose prediction API endpoints for service-based integration.

---

## 17. Conclusion

The project delivers a practical machine learning solution for proactive software defect risk prediction using static OO code metrics. It includes a modular backend, multi-model experimentation, risk classification, persisted artifacts, and an interactive UI suitable for demonstration and applied usage. The repository already reflects a strong implementation foundation for academic and hackathon contexts, with clear opportunities for maturity through unified artifact management, broader automated tests, and stronger validation/reporting standardization.

---

## Appendix A: Repository Outputs Confirmed

### Model and Metric Artifacts
- models/Baseline_RF.pkl
- models/Improved_RF.pkl
- models/Logistic_Regression.pkl
- models/Naive_Bayes.pkl
- models/scaler.pkl
- models/comparison_metrics.json
- models/evaluation_results.json
- models/threshold_metadata.json

### Visualization Outputs
- visualizations/01_metric_comparison.png
- visualizations/02_recall_improvement.png
- visualizations/03_confusion_matrices.png
- visualizations/04_roc_curves.png
- visualizations/05_feature_importance.png
- visualizations/06_model_rankings.png
- visualizations/07_baseline_vs_improved.png
- visualizations/08a_risk_distribution_counts.png
- visualizations/08b_risk_distribution_pct.png
- visualizations/08c_high_risk_modules.png

### Test Artifact
- tests/test_thresholding_reliability.py
