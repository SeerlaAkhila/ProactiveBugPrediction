# 🐛 Proactive Software Bug Prediction System

## Project Overview

A **machine learning-based defect prediction system** that identifies bug-prone software modules using static code metrics. The system employs advanced techniques like SMOTE for class balancing and multi-model comparison to achieve reliable bug detection.

**Business Value:**
- 🎯 Proactive bug detection (11.1% improvement in recall)
- 💰 Reduce maintenance costs
- 🧪 Optimize testing effort
- ⚠️ Risk-based testing prioritization
- 🎚️ Threshold-tuned predictions for better recall/precision balance
- 🔍 SHAP-based module explainability for root-cause insight

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT WEB APPLICATION (app.py)             │
│  Interactive UI for predictions, visualizations, analysis   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────┐
│          MODULAR BUG PREDICTION SYSTEM (system.py)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ DataPreprocessor │  │  ModelTrainer    │               │
│  │  - Load data     │  │  - Train models  │               │
│  │  - Clean data    │  │  - Apply SMOTE   │               │
│  │  - Scale features│  │  - Hyperparams   │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ ModelEvaluator   │  │ RiskClassifier   │               │
│  │  - Accuracy      │  │  - LOW (0-0.3)   │               │
│  │  - Recall        │  │  - MEDIUM (0.3-0.7)│             │
│  │  - F1-Score      │  │  - HIGH (0.7-1)  │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ PredictionEngine │  │ Visualizer       │               │
│  │  - Single pred   │  │ - Model compare  │               │
│  │  - Batch pred    │  │ - Risk distrib.  │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────┐
│                    DATA & MODELS                            │
├─────────────────────────────────────────────────────────────┤
│  • cleaned_dataset.csv (997 samples)                        │
│  • Trained Models:                                          │
│    - Baseline Random Forest                                 │
│    - Improved Random Forest (SMOTE)                         │
│    - Logistic Regression                                    │
│    - Gaussian Naive Bayes                                   │
│  • Scaler (StandardScaler)                                  │
│  • Evaluation Metrics                                       │
│  • Risk Classification Results                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Summary

### Trained Models (`src/model_comparison.py`)
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Baseline RF | 0.8450 | 0.6923 | **0.4390** | 0.5373 | 0.7896 |
| Improved RF | 0.8050 | 0.5263 | **0.4878** | 0.5063 | 0.7671 |
| Logistic Regression | 0.8150 | 0.5500 | **0.5366** | 0.5432 | 0.7964 |
| Naive Bayes | 0.8550 | 0.7727 | **0.4146** | 0.5397 | 0.7574 |

### Visualization Outputs (`src/visualization.py`)
1. `01_metric_comparison.png` - Accuracy, Precision, Recall, F1 comparison
2. `02_recall_improvement.png` - Bug detection improvement highlight
3. `03_confusion_matrices.png` - Prediction accuracy for each model
4. `04_roc_curves.png` - ROC curves and AUC comparison
5. `05_feature_importance.png` - Feature impact analysis
6. `06_model_rankings.png` - Model ranked by different metrics
7. `07_baseline_vs_improved.png` - Direct baseline vs improved comparison

### Risk Classification (`src/risk_classification.py`)
- 🟢 **LOW RISK (0.0 - 0.3):** Unlikely to have bugs
- 🟡 **MEDIUM RISK (0.3 - 0.7):** Moderate chance of bugs
- 🔴 **HIGH RISK (0.7 - 1.0):** Likely to have bugs

### Application Features (`app.py`)
- 📊 **Dashboard:** Key metrics and charts
- 🎯 **Single Prediction:** One-module prediction with probability + risk level
- 📁 **Upload Analysis:** Batch CSV prediction and triage
- 📈 **Model Comparison:** Performance table + charts
- 🎚️ **Threshold-Aware Inference:** Uses tuned model thresholds
- 🔍 **SHAP Explainability Panel:** Per-module feature contribution view
- ℹ️ **System Information:** Project capabilities and dataset details

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Step 1: Clone/Download Project
```bash
cd Proactivebug
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Dataset
Ensure `data/processed/cleaned_dataset.csv` exists with columns:
- LOC, CBO, RFC, WMC, defect

### Step 5: Train Models (if needed)
```bash
python src/model_comparison.py
```

### Step 6: Generate Visualizations
```bash
python src/visualization.py
```

### Step 7: Generate Risk Classification
```bash
python src/risk_classification.py
```

---

## Usage Guide

### Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command Line
```bash
# Train all models
python src/model_comparison.py

# Generate visualizations
python src/visualization.py

# Risk classification
python src/risk_classification.py
```

### Option 3: Python API
```python
from src.system import BugPredictionSystem, PredictionEngine

# Load system
system = BugPredictionSystem()
system.run_complete_pipeline('data/processed/cleaned_dataset.csv')

# Get predictor
predictor = system.get_prediction_engine('Improved RF')

# Single prediction
result = predictor.predict_single([100, 5, 15, 10])
# Output:
# {
#   'prediction': 0,  # 0=Clean, 1=Buggy
#   'probability': 0.1356,  # Defect probability
#   'risk_level': 'LOW',  # Risk category
#   'label': 'Clean',
#   'decision_threshold': 0.432  # Tuned threshold used for classification
# }

# Batch prediction
import numpy as np
X = np.array([
    [100, 5, 15, 10],
    [200, 10, 25, 15],
    [150, 8, 20, 12]
])
results_df = predictor.predict_batch(X)
```

---

## File Structure

```
Proactivebug/
├── app.py                          # Streamlit UI application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── check.py                        # Data preprocessing script
├── data/
│   ├── raw/
│   │   └── bug-metrics.csv        # Raw dataset
│   └── processed/
│       └── cleaned_dataset.csv    # Cleaned dataset (997 samples)
├── src/
│   ├── model_comparison.py        # Phase 6: Train & compare models
│   ├── visualization.py           # Phase 7: Generate visualizations
│   ├── risk_classification.py     # Phase 8: Risk categorization
│   ├── system.py                  # Phase 9: Modular architecture
│   ├── train_improved.py          # Improved RF training (baseline)
│   ├── trainbaseline.py           # Baseline RF training
│   └── evaluate.py                # Basic evaluation script
├── models/                        # Trained models (generated)
│   ├── Baseline_RF.pkl
│   ├── Improved_RF.pkl
│   ├── Logistic_Regression.pkl
│   ├── Naive_Bayes.pkl
│   ├── scaler.pkl
│   ├── comparison_metrics.json
│   ├── evaluation_results.json
│   └── threshold_metadata.json
├── visualizations/                # Generated charts (generated)
│   ├── 01_metric_comparison.png
│   ├── 02_recall_improvement.png
│   ├── 03_confusion_matrices.png
│   ├── 04_roc_curves.png
│   ├── 05_feature_importance.png
│   ├── 06_model_rankings.png
│   └── 07_baseline_vs_improved.png
├── logs/                          # Application logs (generated)
│   └── *.log
└── risk_classification_report.json # Risk analysis results (generated)
```

---

## Software Metrics Explanation

### 1. LOC (Lines of Code)
- **Definition:** Total number of lines in a class
- **Range:** 1 - 10,000+
- **Interpretation:** Higher values indicate more complex modules

### 2. CBO (Coupling Between Objects)
- **Definition:** Number of other classes this class couples to
- **Range:** 0 - 100+
- **Interpretation:** High coupling increases maintenance difficulty

### 3. RFC (Response For Class)
- **Definition:** Number of different methods a class uses
- **Range:** 0 - 100+
- **Interpretation:** Higher RFC = more dependencies

### 4. WMC (Weighted Methods per Class)
- **Definition:** Sum of method complexities in a class
- **Range:** 1 - 100+
- **Interpretation:** Higher WMC = more complex methods

---

## Model Performance Summary

### Baseline Random Forest
- **Accuracy:** 84.5%
- **Recall:** 43.9% (Misses ~56% of bugs)
- **Limitation:** Conservative predictions, misses many bugs

### Improved Random Forest (RECOMMENDED)
- **Accuracy:** 80.5%
- **Recall:** 48.78% (+11.1% improvement)
- **Advantage:** Better bug detection with SMOTE balancing

### Logistic Regression
- **Accuracy:** 81.5%
- **Recall:** 53.66% (Best recall)
- **Advantage:** Highest recall for bug detection

### Naive Bayes
- **Accuracy:** 85.5%
- **Recall:** 41.46%
- **Limitation:** Lower recall despite high accuracy

---

## Key Findings & Recommendations

### ✅ Class Imbalance Solution
- **Problem:** 80% clean modules, 20% buggy modules
- **Solution:** SMOTE (Synthetic Minority Over-sampling)
- **Result:** Balanced training data (50-50), improved bug detection

### ✅ Model Selection
- **For Bug Detection:** Use Improved RF or Logistic Regression
- **For Overall Performance:** Use Naive Bayes (highest accuracy)
- **Recommendation:** Improved RF (good balance of accuracy & recall)

### ✅ Risk-Based Testing Strategy
1. **High Risk Modules:** Intensive testing + code review
2. **Medium Risk Modules:** Standard testing + peer review
3. **Low Risk Modules:** Basic testing + documentation

---

## Running the Complete Pipeline

### Option A: Interactive Streamlit App (Easiest)
```bash
streamlit run app.py
```

### Option B: Command Line (Step-by-Step)
```bash
# Step 1: Train all models
python src/model_comparison.py

# Step 2: Generate visualizations
python src/visualization.py

# Step 3: Risk classification
python src/risk_classification.py

# Step 4: Check models directory
ls models/
```

### Option C: Python Script
```python
from src.system import BugPredictionSystem

system = BugPredictionSystem()
results = system.run_complete_pipeline('data/processed/cleaned_dataset.csv')
system.save_models()
print("Pipeline complete!")
```

---

## Troubleshooting

### Issue: "Module not found" error
**Solution:** Ensure you're in the Proactivebug directory and have installed all requirements
```bash
pip install -r requirements.txt
```

### Issue: "cleaned_dataset.csv not found"
**Solution:** Run data preprocessing first:
```bash
python check.py
```

### Issue: Models not found in app.py
**Solution:** Train models first:
```bash
python src/model_comparison.py
```

### Issue: Streamlit not installed
**Solution:**
```bash
pip install streamlit plotly
```

### Issue: SHAP panel shows "library is not installed"
**Solution:** Install SHAP in the same environment used to run Streamlit
```bash
pip install shap
python -m streamlit run app.py
```

---

## Performance Metrics Explanation

### Accuracy
- **Definition:** (TP + TN) / (TP + TN + FP + FN)
- **Use Case:** Overall correctness
- **Note:** Can be misleading with imbalanced data

### Recall (MOST IMPORTANT FOR BUG DETECTION)
- **Definition:** TP / (TP + FN)
- **Use Case:** Finding all bugs (minimize false negatives)
- **Target:** As high as possible (we want to catch bugs)

### Precision
- **Definition:** TP / (TP + FP)
- **Use Case:** Minimizing false alarms
- **Note:** Trade-off with recall

### F1-Score
- **Definition:** 2 * (Precision * Recall) / (Precision + Recall)
- **Use Case:** Balanced metric when both matter

### AUC-ROC
- **Definition:** Area under the receiver operating characteristic curve
- **Use Case:** Performance across different thresholds
- **Range:** 0.5 (random) to 1.0 (perfect)

---

## Future Enhancements

### Short-term
- [ ] Cross-validation for more robust evaluation
- [ ] Hyperparameter grid search
- [ ] Feature engineering (interaction terms, polynomials)
- [ ] ROC threshold optimization

### Medium-term
- [ ] Deep learning models (Neural Networks)
- [ ] Ensemble methods (Voting, Stacking)
- [ ] Imbalanced learning alternatives (EasyEnsemble, BalancedRF)
- [ ] API for integration with CI/CD pipelines

### Long-term
- [ ] Multi-version defect prediction
- [ ] Temporal analysis (bugs over versions)
- [ ] Code smell integration
- [ ] Real-time static analysis integration

---

## References & Resources

### Software Metrics (CK Metrics)
- CBO (Coupling Between Objects)
- RFC (Response For Class)
- WMC (Weighted Methods per Class)
- LOC (Lines of Code)

**Source:** Chidamber & Kemerer, 1994

### Class Imbalance Solutions
- SMOTE: Chawla et al., 2002
- Random Over-sampling
- Threshold adjustment
- Class weights

### Model Comparison
- Random Forest (Breiman, 2001)
- Logistic Regression (Logit Model)
- Naive Bayes (Probabilistic Classification)

### Tools & Libraries
- scikit-learn: ML algorithms
- imbalanced-learn: SMOTE implementation
- Streamlit: Web UI
- Plotly: Interactive visualizations
- SHAP: Local prediction explainability

---

## Contact & Support

For issues, questions, or improvements:
1. Check existing documentation
2. Review troubleshooting section
3. Examine log files in `logs/` directory
4. Inspect model performance in `models/evaluation_results.json`
5. Inspect threshold metadata in `models/threshold_metadata.json`

---

## License & Attribution

This project implements defect prediction using the CK metrics dataset.

**Citation:** Chidamber, S. R., & Kemerer, C. F. (1994). A metrics suite for object-oriented design. Transactions on Software Engineering, 20(6), 476-493.

---

**Last Updated:** April 8, 2026  
**Version:** 2.1 (Threshold + Explainability Update)  
**Status:** ✅ Production Ready
