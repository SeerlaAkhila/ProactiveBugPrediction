# Proactive Software Bug Prediction System

## Overview
This project predicts whether a software module is likely to be buggy using static code metrics:
- `LOC` (Lines of Code)
- `CBO` (Coupling Between Objects)
- `RFC` (Response For Class)
- `WMC` (Weighted Methods per Class)

The goal is to support proactive testing by identifying high-risk modules before release.

## Core Use Case
- Input: per-module software metrics
- Output: defect probability, binary label (`Buggy` / `Clean`), and risk tier (`LOW` / `MEDIUM` / `HIGH`)
- Primary users: QA leads, SDETs, and engineering teams doing risk-based test planning

## Current Architecture

### 1) Application Layer
- `app.py` (Streamlit UI)
  - Dashboard
  - Single Prediction (default model: Logistic Regression)
  - Upload Analysis (prediction-focused batch inference)
  - Risk Analysis (risk distribution and prioritization views)
  - Model Comparison
  - System Information

### 2) Core ML System
- `src/system.py` (modular orchestration)
  - `DataPreprocessor`
  - `ModelTrainer`
  - `ModelEvaluator`
  - `RiskClassifier`
  - `PredictionEngine`
  - `BugPredictionSystem`

### 3) Supporting Scripts
- `src/model_comparison.py` (phase script for training/comparison)
- `src/visualization.py` (phase script for charts)
- `src/risk_classification.py` (phase script for report + plots)
- `src/trainbaseline.py`, `src/train_improved.py` (quick standalone checks)
- `src/evaluate.py` (simple plotting utility)

## Reliability-Critical Design (Latest)

### Validation-only threshold tuning
`src/system.py` now uses a train/validation/test split:
1. Train models on `train`
2. Tune decision thresholds on `validation`
3. Report final metrics on `test`

This prevents optimistic threshold leakage from the test set.

### Startup behavior in app
`app.py` now:
1. Loads saved artifacts from `models/` first
2. Does **not** retrain at runtime (fails safely if artifacts are missing/broken)
3. Uses default threshold `0.5` when `threshold_metadata.json` is unavailable

## Data and Artifacts

### Input dataset
- `data/processed/cleaned_dataset.csv`
- Required columns: `LOC, CBO, RFC, WMC, defect`

### Saved artifacts
- `models/Baseline_RF.pkl`
- `models/Improved_RF.pkl`
- `models/Logistic_Regression.pkl`
- `models/Naive_Bayes.pkl`
- `models/scaler.pkl`
- `models/evaluation_results.json`
- `models/threshold_metadata.json` (if generated/saved from `system.py`)
- `models/cross_validation_results.json` (saved by `system.py` when CV runs)
- `models/cross_validation_metrics.json` (saved by `src/model_comparison.py`)

## Modeling Details
- Class imbalance handling (current):
  - Baseline RF: class-weighted (`balanced_subsample`)
  - Improved RF: SMOTE on train split
  - Logistic Regression: class-weighted (`balanced`)
  - Naive Bayes: no synthetic oversampling
- Evaluated models:
  - Baseline Random Forest
  - Improved Random Forest
  - Logistic Regression
  - Gaussian Naive Bayes
- Metrics tracked:
  - Accuracy
  - Precision
  - Recall (primary for bug detection)
  - F1-score
  - AUC-ROC
  - Brier score (probability quality)
- Thresholding and calibration:
  - Thresholds tuned on validation probabilities
  - Probability calibration supported (Platt scaling in evaluator path)
- Cross-validation:
  - Stratified K-fold reporting with mean and standard deviation metrics

## UI Behavior Notes
- Single Prediction supports model selection; SHAP panel is shown for RF models.
- Single Prediction uses dataset-driven recommended input ranges for `LOC/CBO/RFC/WMC`.
- Inputs outside training distribution are flagged as potentially unreliable.
- Upload Analysis is prediction-only (no risk summary overlap).
- Risk Analysis is dedicated to risk distribution and high-risk module prioritization.

## Installation

```bash
pip install -r requirements.txt
```

## Run

### Streamlit app
```bash
streamlit run app.py
```

### Core pipeline (train + evaluate + save)
```bash
python src/system.py
```

### Optional phase scripts
```bash
python src/model_comparison.py
python src/visualization.py
python src/risk_classification.py
```

## Programmatic Usage

```python
from src.system import BugPredictionSystem
import numpy as np

system = BugPredictionSystem({
    "run_cross_validation": True,
    "cv_splits": 5,
    "calibrate_probabilities": True
})
system.run_complete_pipeline("data/processed/cleaned_dataset.csv")
system.save_models()

predictor = system.get_prediction_engine("Logistic Regression")
result = predictor.predict_single(np.array([150, 8, 20, 10]))
print(result)
# keys include probability, decision_threshold, risk_level, and OOD flags
# e.g., result["out_of_distribution"], result["ood_warnings"]
```

## Tests
Reliability tests were added to guard thresholding behavior:
- `tests/test_thresholding_reliability.py`

Run:
```bash
pytest -q
```

## Project Strengths
- Practical end-to-end workflow from training to interactive inference
- Modular core system (`src/system.py`)
- Clear focus on recall-aware bug detection
- Threshold-aware inference supported by metadata

## Known Gaps / Next Improvements
1. Add calibration curve plots and expected calibration error (ECE) reporting
2. Consolidate duplicate phase scripts over time to reduce maintenance drift
3. Expand automated tests beyond thresholding (I/O contracts, artifact loading, UI smoke)
4. Add project-wise generalization checks (for multi-project datasets)

## Repository Layout

```text
Proactivebug/
  app.py
  requirements.txt
  README.md
  data/
    raw/
    processed/
      cleaned_dataset.csv
  src/
    system.py
    model_comparison.py
    visualization.py
    risk_classification.py
    trainbaseline.py
    train_improved.py
    evaluate.py
  models/
  visualizations/
  logs/
  tests/
    test_thresholding_reliability.py
```

## Attribution
This project uses CK-style object-oriented design metrics for defect prediction experiments.
