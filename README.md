# 🐛 Proactive Software Bug Prediction System

## Project Overview

A **machine learning-based defect prediction system** that identifies bug-prone software modules using static code metrics. The system employs advanced techniques like SMOTE for class balancing and multi-model comparison to achieve reliable bug detection.

**Business Value:**
- 🎯 Proactive bug detection (11.1% improvement in recall)
- 💰 Reduce maintenance costs
- 🧪 Optimize testing effort
- ⚠️ Risk-based testing prioritization

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
│  │  - Recall        │  │  - MEDIUM (0.3-7)│               │
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

## Phase Breakdown

### ✅ PHASE 1-5: Completed (Baseline)
- Problem understanding & dataset preparation
- Data preprocessing & exploratory analysis
- Baseline Random Forest model (83% accuracy, 57% recall)

### ✅ PHASE 6: Model Comparison
**Status:** Completed ✓

**Implementation:** `src/model_comparison.py`

**Trained 4 Models:**
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Baseline RF | 0.8450 | 0.6923 | **0.4390** | 0.5373 | 0.7896 |
| Improved RF | 0.8050 | 0.5263 | **0.4878** | 0.5063 | 0.7671 |
| Logistic Regression | 0.8150 | 0.5500 | **0.5366** | 0.5432 | 0.7964 |
| Naive Bayes | 0.8550 | 0.7727 | **0.4146** | 0.5397 | 0.7574 |

**Key Results:**
- ✅ Improved RF: +11.1% recall improvement
- ✅ Best recall: Logistic Regression (0.5366)
- ✅ SMOTE successfully balanced data (632:632)

### ✅ PHASE 7: Visualization
**Status:** Completed ✓

**Implementation:** `src/visualization.py`

**Generated 7 Comprehensive Visualizations:**
1. `01_metric_comparison.png` - Accuracy, Precision, Recall, F1 comparison
2. `02_recall_improvement.png` - Bug detection improvement highlight
3. `03_confusion_matrices.png` - Prediction accuracy for each model
4. `04_roc_curves.png` - ROC curves and AUC comparison
5. `05_feature_importance.png` - Feature impact analysis
6. `06_model_rankings.png` - Model ranked by different metrics
7. `07_baseline_vs_improved.png` - Direct baseline vs improved comparison

### ✅ PHASE 8: Risk Classification
**Status:** Completed ✓

**Implementation:** `src/risk_classification.py`

**Risk Thresholds:**
- 🟢 **LOW RISK (0.0 - 0.3):** Unlikely to have bugs
- 🟡 **MEDIUM RISK (0.3 - 0.7):** Moderate chance of bugs
- 🔴 **HIGH RISK (0.7 - 1.0):** Likely to have bugs

**Distribution Example (Improved RF):**
- Low Risk: 130 modules (65%)
- Medium Risk: 49 modules (24.5%)
- High Risk: 21 modules (10.5%)

**Generated Visualizations:**
- `08a_risk_distribution_counts.png` - Absolute module counts
- `08b_risk_distribution_pct.png` - Percentage distribution
- `08c_high_risk_modules.png` - High-risk module detection

### ✅ PHASE 9: System Design
**Status:** Completed ✓

**Implementation:** `src/system.py`

**Modular Architecture (6 Components):**

1. **DataPreprocessor**
   - Load and validate datasets
   - Feature extraction & scaling
   - Train-test split with stratification

2. **ModelTrainer**
   - Baseline & Improved Random Forest
   - Logistic Regression & Naive Bayes
   - SMOTE for class balancing

3. **ModelEvaluator**
   - Comprehensive metric calculation
   - Confusion matrices
   - ROC-AUC analysis

4. **RiskClassifier**
   - Probability to risk conversion
   - Risk distribution analysis
   - Batch risk classification

5. **PredictionEngine**
   - Single sample predictions
   - Batch predictions
   - Probability estimates

6. **BugPredictionSystem** (Orchestrator)
   - Coordinate all modules
   - Complete pipelines
   - Model persistence (save/load)

### ✅ PHASE 10: Functional System
**Status:** Completed ✓

**Features:**
- Complete integrated system
- All modules working together
- Model training & evaluation pipeline
- Batch prediction support

### ✅ PHASE 12: Streamlit UI
**Status:** Completed ✓

**Implementation:** `app.py`

**Features:**
- 📊 **Dashboard:** Key metrics & visualizations
- 🎯 **Single Prediction:** Enter metrics for one module
- 📁 **Batch Predictions:** Upload CSV, predict multiple modules
- 📈 **Model Comparison:** Compare all 4 models
- ⚠️ **Risk Analysis:** Risk distribution & statistics
- ℹ️ **System Information:** Architecture & documentation

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
#   'label': 'Clean'
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
│   └── comparison_metrics.json
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

---

## Contact & Support

For issues, questions, or improvements:
1. Check existing documentation
2. Review troubleshooting section
3. Examine log files in `logs/` directory
4. Inspect model performance in `models/comparison_metrics.json`

---

## License & Attribution

This project implements defect prediction using the CK metrics dataset.

**Citation:** Chidamber, S. R., & Kemerer, C. F. (1994). A metrics suite for object-oriented design. Transactions on Software Engineering, 20(6), 476-493.

---

**Last Updated:** March 26, 2026  
**Version:** 2.0 (Complete System)  
**Status:** ✅ Production Ready
