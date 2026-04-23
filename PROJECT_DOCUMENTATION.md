# Proactive Software Bug Prediction System - Project Documentation

## 1. Project Overview

### Project Name
Proactive Software Bug Prediction System

### Purpose
This project predicts whether a software module is likely to be buggy using static code metrics. It is designed to help teams prioritize testing, code review, and maintenance effort before defects reach production.

### Problem It Solves
Traditional testing often treats modules uniformly, even though some modules are much more defect-prone than others. This project addresses that problem by assigning risk to modules based on machine learning predictions over software metrics such as LOC, CBO, RFC, and WMC.

### Target Users
- Software quality engineers
- QA and test automation teams
- Developers doing risk-based code review
- Engineering managers planning limited testing resources

### Real-World Use Cases
- Prioritizing regression tests for risky modules
- Focusing code review on components with high defect probability
- Identifying modules that need refactoring or extra validation
- Helping teams triage large codebases with limited QA capacity

## 2. System Architecture

### High-Level Architecture
The system is built as a pipeline-based machine learning application with a Streamlit front end and a modular backend.

1. Data is loaded from a processed CSV dataset.
2. Features are validated and split into train, validation, and test sets.
3. Multiple models are trained and evaluated.
4. Decision thresholds are tuned on validation data.
5. Predictions are converted into risk levels.
6. The Streamlit UI presents predictions, metrics, charts, and explanations.

### Component Breakdown
- `app.py`: Streamlit user interface and interactive analysis pages
- `src/system.py`: Core orchestration layer with preprocessing, training, evaluation, thresholding, and prediction
- `src/model_comparison.py`: Standalone model training and comparison script
- `src/risk_classification.py`: Risk analysis and visualization logic
- `src/trainbaseline.py`: Baseline Random Forest training script
- `src/train_improved.py`: SMOTE-based improved Random Forest script
- `src/evaluate.py`: Simple plotting helper for comparison metrics
- `check.py`: Dataset cleaning and transformation script

### How the Parts Interact
- `app.py` loads a `BugPredictionSystem` instance from `src/system.py`.
- The system loads saved models and scaler from the `models/` directory when available.
- If saved artifacts are missing, the system can retrain from `data/processed/cleaned_dataset.csv`.
- Predictions are made through `PredictionEngine`, which applies the stored threshold and risk classification.
- The UI displays model comparison charts, batch upload analysis, single-sample predictions, and system information.

### Text Architecture Diagram
```text
CSV Dataset -> DataPreprocessor -> ModelTrainer -> ModelEvaluator -> Threshold Metadata
        |                                                             |
        |                                                             v
        +-------------------------------> PredictionEngine -> RiskClassifier
                                                      |
                                                      v
                                                Streamlit UI
```

## 3. Tech Stack

### Core Technologies
- Python: Main implementation language
- Pandas: Data loading, transformation, and tabular handling
- NumPy: Numerical operations and array processing
- scikit-learn: Model training, scaling, splitting, and evaluation
- imbalanced-learn: SMOTE class balancing
- Plotly: Interactive charts in the UI
- Matplotlib and Seaborn: Static visualization generation
- Streamlit: Web application interface
- Joblib / pickle: Model persistence
- SHAP: Explainability for feature contribution analysis

### Why These Were Chosen
- Python provides a mature ecosystem for ML and data apps.
- scikit-learn and imbalanced-learn are well suited for classical defect prediction on structured metrics.
- Streamlit minimizes UI complexity while still allowing interactive analytics.
- SHAP provides model explanation without building a separate interpretability service.

### Possible Alternatives
- XGBoost or LightGBM instead of Random Forest / Logistic Regression
- FastAPI instead of Streamlit if this were exposed as an API-first service
- Great Expectations or Pandera for stronger data validation
- MLflow for experiment tracking and model registry
- React or Next.js for a richer custom UI

## 4. Setup and Installation

### Prerequisites
- Python 3.8 or later
- pip or a virtual environment manager
- Access to the cleaned dataset at `data/processed/cleaned_dataset.csv`

### Installation Steps
1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Confirm the dataset exists in `data/processed/`.
4. Run the training pipeline if models are not already present.
5. Launch the Streamlit application.

### Local Run Instructions
```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python src/model_comparison.py
streamlit run app.py
```

### Notes
- The application is designed to prefer saved artifacts in `models/`.
- If models are missing, the system can retrain automatically from the processed dataset.

## 5. Folder and Code Structure

### Top-Level Files
- `app.py`: Main Streamlit entry point
- `README.md`: Short project summary and usage notes
- `requirements.txt`: Python dependencies
- `check.py`: Data preparation script
- `COMPLETE_IMPLEMENTATION_DOCUMENTATION.md`: Currently empty placeholder
- `IMPLEMENTATION_SUMMARY.md`: High-level completion summary
- `risk_classification_report.json`: Saved risk distribution output

### `data/`
- `data/raw/bug-metrics.csv`: Raw metrics dataset
- `data/raw/single-version-ck-oo.csv`: Source CK metrics file
- `data/processed/cleaned_dataset.csv`: Cleaned binary-classification dataset used for training

### `src/`
- `system.py`: Core modular ML system
- `model_comparison.py`: Multi-model training and evaluation
- `risk_classification.py`: Risk analysis and charts
- `trainbaseline.py`: Baseline RF experiment script
- `train_improved.py`: SMOTE-enhanced RF experiment script
- `evaluate.py`: Simple comparison plot script

### `models/`
- Serialized trained classifiers
- Saved scaler
- Evaluation and threshold metadata JSON files

### `visualizations/`
- Generated charts for model comparison and risk analysis

### `logs/`
- Runtime logs from preprocessing, training, evaluation, and prediction modules

## 6. Core Features and Functionality

### 1. Dataset Loading and Validation
The system loads the processed dataset and checks that the required columns exist: `LOC`, `CBO`, `RFC`, `WMC`, and `defect`.

### 2. Feature Scaling
Numeric features are standardized using `StandardScaler` so that algorithms sensitive to feature scale behave consistently.

### 3. Model Training
Four classifiers are trained:
- Baseline Random Forest
- Improved Random Forest with SMOTE and tuned hyperparameters
- Logistic Regression with SMOTE
- Gaussian Naive Bayes with SMOTE

### 4. Model Evaluation
Each model is scored using:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrix

### 5. Threshold Tuning
Instead of using a default 0.5 threshold blindly, the system chooses a decision threshold based on validation data to optimize a selected metric, usually F1.

### 6. Risk Classification
Predicted probabilities are mapped to risk levels:
- LOW: below 0.3
- MEDIUM: 0.3 to 0.7
- HIGH: above 0.7

### 7. Single Prediction UI
Users can enter LOC, CBO, RFC, and WMC for one module and get a defect prediction, probability, and risk level.

### 8. Batch Upload Analysis
Users can upload CSV, TSV, Excel, JSON, Parquet, XML, or HTML tables for batch prediction and risk triage.

### 9. Model Comparison Dashboard
The UI shows model performance side by side with interactive charts.

### 10. SHAP Explainability
For supported models, the app computes SHAP values to show which features most influenced an individual prediction.

### End-to-End Data Flow
1. Raw metric data is cleaned into a binary defect dataset.
2. The system splits the data into train, validation, and test partitions.
3. Models are trained on the training split.
4. Thresholds are tuned on validation data.
5. Final metrics are computed on the test split.
6. Saved models and scaler are loaded into the UI.
7. User input is transformed, scored, and displayed with risk labels and explanations.

## 7. Deep Code Explanation

### `BugPredictionSystem`
This is the main orchestrator in `src/system.py`. It coordinates preprocessing, training, evaluation, persistence, and prediction serving.

Key responsibilities:
- Manage end-to-end pipeline execution
- Store trained models and metadata
- Expose prediction engines for each model
- Persist artifacts to disk

### `DataPreprocessor`
This class:
- Loads the dataset
- Validates required columns
- Extracts features and target
- Splits data into train/validation/test
- Applies feature scaling

The move to a validation split is important because it makes threshold tuning less biased.

### `ModelTrainer`
This class trains the supported classifiers.

Important design points:
- SMOTE is applied only to training data for balanced learning.
- The improved Random Forest uses stronger regularization than the baseline model.
- Logistic Regression is used as a linear baseline with probabilistic output.

### `ModelEvaluator`
This module calculates standard classification metrics and tunes thresholds.

Non-obvious logic:
- Candidate thresholds are generated from a grid plus unique predicted probabilities.
- The best threshold is selected by a score tuple that prioritizes the target metric, then recall, precision, and closeness to 0.5.
- This produces stable threshold selection while avoiding arbitrary boundary choices.

### `PredictionEngine`
This is the runtime inference wrapper.

It handles:
- Single-sample prediction
- Batch prediction
- Probability estimation
- Risk label assignment

It uses the stored threshold rather than a hardcoded 0.5 cutoff.

### `RiskClassifier`
This module converts numeric probabilities into LOW, MEDIUM, or HIGH risk labels.

This layer matters because users usually want a triage decision, not just a probability.

### `app.py` Helper Logic
The app contains supporting code for:
- Flexible file parsing
- Metric column alias resolution
- SHAP explanation rendering
- Risk report display

The upload parser is intentionally broad so users can provide data in different tabular formats without manual conversion.

## 8. APIs and Database

### APIs
This project does not expose a REST or GraphQL API. The interface is a Streamlit application.

### Data Interface
The main input is a tabular dataset with these columns:
- `LOC`
- `CBO`
- `RFC`
- `WMC`
- `defect`

### Outputs
- Model predictions
- Probability scores
- Risk labels
- Evaluation summaries
- SHAP explanation tables
- JSON reports and generated charts

### Database
There is no database layer in the current project.
Model artifacts and reports are stored in files under `models/`, `visualizations/`, and the project root.

## 9. Design Decisions

### 1. Classical ML Instead of Deep Learning
The problem is tabular and small-scale, so classical models are more practical, faster to train, and easier to explain.

### 2. SMOTE for Imbalance Handling
Buggy modules are usually the minority class. SMOTE improves recall by providing more balanced training data.

### 3. Validation-Based Thresholding
Bug detection is recall-sensitive, so threshold selection is tuned separately from final evaluation to reduce optimistic bias.

### 4. Model Comparison Instead of Single Model Reliance
Multiple models are retained because different teams may prioritize different trade-offs between recall and precision.

### 5. Streamlit for Delivery
Streamlit keeps the project lightweight and easy to run without a separate frontend stack.

### Trade-Offs
- Better recall often lowers precision.
- SHAP adds interpretability but increases computation time.
- File-based persistence is simple but not as robust as a dedicated model registry.

## 10. Performance and Scalability

### Current Limitations
- The dataset is relatively small, so metric variance can be significant.
- The system uses a single train/validation/test split rather than repeated cross-validation.
- SHAP explanation may be slow for larger batches.

### How It Scales
- Training remains manageable because the feature set is small.
- Inference is lightweight and suitable for interactive use.
- The system can scale to more modules or more rows in batch upload as long as memory remains sufficient.

### Optimization Opportunities
- Add repeated stratified cross-validation
- Add probability calibration
- Cache model artifacts more aggressively in the UI
- Move from file-based storage to an experiment tracking system
- Add incremental retraining for new datasets

## 11. Testing and Debugging

### How to Test
- Run `python src/model_comparison.py` to verify training and metrics generation.
- Run `python src/risk_classification.py` to verify risk reporting and visualizations.
- Launch `streamlit run app.py` to test the UI.
- Execute reliability tests in `tests/test_thresholding_reliability.py` if `pytest` is installed.

### Common Issues
- Missing dependencies: install from `requirements.txt`.
- Missing dataset: confirm `data/processed/cleaned_dataset.csv` exists.
- Missing model artifacts: rerun the training pipeline.
- Encoding issues on Windows: avoid non-ASCII console output in terminal scripts.

### Debugging Tips
- Check logs under `logs/` for training or preprocessing failures.
- Verify that `models/threshold_metadata.json` exists when using persisted predictions.
- Confirm the upload file has compatible metric columns or aliases.

## 12. Improvements and Future Scope

### Practical Improvements
- Add cross-validation and confidence intervals
- Add model calibration plots
- Add an exportable prediction report for uploaded files
- Add more feature importances and error analysis views
- Add test coverage for file parsing and SHAP fallback behavior

### Features That Could Be Added
- REST API endpoint for automated scoring
- Authentication for internal teams
- Model registry and versioning
- Dataset drift monitoring
- Explainability for batch predictions
- Threshold customization in the UI

## 13. Interview Explanation Section

### How to Explain This Project in an Interview
This is a machine learning-based bug prediction system that uses static software metrics to estimate the probability that a module contains defects. I built it as a modular pipeline with preprocessing, model training, threshold tuning, risk classification, and an interactive Streamlit dashboard. The most important design choice was to optimize the decision threshold on validation data rather than using a fixed 0.5 cutoff, because bug prediction is recall-sensitive and false negatives are costly.

### Key Talking Points
- The project solves a real testing prioritization problem.
- It balances recall, precision, and operational usability.
- It includes explainability using SHAP.
- It supports both single prediction and batch triage.
- It persists models and metadata for reproducible inference.

### Possible Interview Questions
1. Why did you choose Random Forest, Logistic Regression, and Naive Bayes?
2. Why is recall so important in bug prediction?
3. What does SMOTE do, and when should it be used?
4. Why tune threshold on validation data instead of test data?
5. How would you improve reliability if you had more time?
6. How does SHAP help in this project?
7. What are the limitations of a single train/test split?
8. How would you turn this into a production service?

## 14. Final Summary (TL;DR)

This project is a Streamlit-based software bug prediction system that uses static code metrics to classify modules as clean or buggy and to assign them a risk level. It trains multiple ML models, tunes thresholds for better defect detection, supports batch analysis, and provides explainability through SHAP. The codebase is modular, reproducible, and already strong for a prototype or internal decision-support tool, with clear next steps around cross-validation, calibration, and production hardening.
