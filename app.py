"""
Software Bug Prediction System - Interactive Streamlit UI
=========================================================

Web interface for:
- Input: Software metrics (LOC, CBO, RFC, WMC)
- Output: Defect prediction, probability, risk level
- Visualization: Model performance and risk distribution
- Explainability: SHAP per-module contribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os
import io
from pathlib import Path
try:
    import shap
except ImportError:
    shap = None

# Add src to path
import sys
sys.path.insert(0, 'src')

from system import BugPredictionSystem, PredictionEngine, RiskClassifier

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bug Prediction System",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def load_system():
    """Load Bug Prediction System"""
    system = BugPredictionSystem()
    try:
        system.run_complete_pipeline('data/processed/cleaned_dataset.csv')
        return system
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None


METRIC_ALIASES = {
    'LOC': ['loc', 'linesofcode', 'numberoflinesofcode', 'number_of_lines_of_code', 'linecount'],
    'CBO': ['cbo', 'couplingbetweenobjects', 'coupling_between_objects'],
    'RFC': ['rfc', 'responseforclass', 'response_for_class'],
    'WMC': ['wmc', 'weightedmethodsperclass', 'weighted_methods_per_class'],
}


def _normalize_name(name):
    """Normalize column/key names for flexible matching."""
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def extract_metric_columns(df):
    """
    Extract required model features from flexible CSV schemas.

    Returns:
        extracted_df: DataFrame with standard columns LOC, CBO, RFC, WMC
        mapping: dict showing detected source columns
        missing: list of still-missing standard columns
    """
    normalized_columns = {_normalize_name(col): col for col in df.columns}
    mapping = {}

    for target_col, aliases in METRIC_ALIASES.items():
        candidates = [_normalize_name(target_col), *aliases]
        for candidate in candidates:
            if candidate in normalized_columns:
                mapping[target_col] = normalized_columns[candidate]
                break

    missing = [col for col in METRIC_ALIASES if col not in mapping]
    if missing:
        return None, mapping, missing

    extracted_df = pd.DataFrame({
        target_col: pd.to_numeric(df[source_col], errors='coerce')
        for target_col, source_col in mapping.items()
    })

    invalid_cols = [col for col in extracted_df.columns if extracted_df[col].isna().all()]
    if invalid_cols:
        return None, mapping, invalid_cols

    return extracted_df, mapping, []


def prepare_prediction_input(df):
    """Prepare uploaded data for prediction with flexible schema handling."""
    feature_df, mapping, missing = extract_metric_columns(df)
    if feature_df is None:
        return None, mapping, missing, None, None

    invalid_rows = feature_df.isna().any(axis=1)
    valid_rows = ~invalid_rows

    if int(valid_rows.sum()) == 0:
        return None, mapping, ['No valid numeric rows found for prediction'], None, None

    return (
        feature_df.reset_index(drop=True),
        mapping,
        [],
        df.reset_index(drop=True).copy(),
        valid_rows.reset_index(drop=True)
    )


def read_uploaded_dataframe(uploaded_file):
    """
    Read an uploaded file into a DataFrame with flexible format detection.

    Supports common table formats directly and falls back to parser probing.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    filename = getattr(uploaded_file, "name", "uploaded_file")
    suffix = Path(filename).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    attempts = []

    def _run_parser(parser_label, parser_fn):
        try:
            parsed_df = parser_fn(io.BytesIO(raw_bytes))
            if isinstance(parsed_df, list):
                if not parsed_df:
                    raise ValueError("No tables found in uploaded file.")
                parsed_df = parsed_df[0]
            if not isinstance(parsed_df, pd.DataFrame):
                raise ValueError("Unsupported parsed data structure.")
            if parsed_df.empty:
                raise ValueError("Parsed file contains no rows.")
            return parsed_df, parser_label
        except Exception as exc:
            attempts.append(f"{parser_label}: {exc}")
            return None, None

    preferred_parsers = {
        ".csv": [("CSV", lambda b: pd.read_csv(b))],
        ".tsv": [("TSV", lambda b: pd.read_csv(b, sep="\t"))],
        ".txt": [("Delimited text", lambda b: pd.read_csv(b, sep=None, engine="python"))],
        ".json": [("JSON", lambda b: pd.read_json(b))],
        ".xls": [("Excel", lambda b: pd.read_excel(b))],
        ".xlsx": [("Excel", lambda b: pd.read_excel(b))],
        ".ods": [("Excel/ODS", lambda b: pd.read_excel(b))],
        ".parquet": [("Parquet", lambda b: pd.read_parquet(b))],
        ".xml": [("XML", lambda b: pd.read_xml(b))],
        ".html": [("HTML table", lambda b: pd.read_html(b))],
        ".htm": [("HTML table", lambda b: pd.read_html(b))],
    }

    fallback_parsers = [
        ("CSV", lambda b: pd.read_csv(b)),
        ("Delimited text", lambda b: pd.read_csv(b, sep=None, engine="python")),
        ("JSON", lambda b: pd.read_json(b)),
        ("Excel", lambda b: pd.read_excel(b)),
        ("Parquet", lambda b: pd.read_parquet(b)),
        ("XML", lambda b: pd.read_xml(b)),
        ("HTML table", lambda b: pd.read_html(b)),
    ]

    for label, parser in preferred_parsers.get(suffix, []):
        df, parser_used = _run_parser(label, parser)
        if df is not None:
            return df, parser_used

    for label, parser in fallback_parsers:
        df, parser_used = _run_parser(label, parser)
        if df is not None:
            return df, parser_used

    raise ValueError(
        "Could not parse this file as tabular data. "
        "Accepted formats include CSV, TSV/TXT, Excel, JSON, Parquet, XML, and HTML tables."
    )


def get_risk_value(metrics, base_name):
    """Read risk metrics from either legacy or current report schemas."""
    key_variants = {
        'low': ['low', 'low_risk'],
        'medium': ['medium', 'medium_risk'],
        'high': ['high', 'high_risk'],
        'low_pct': ['low_pct', 'low_risk_pct'],
        'medium_pct': ['medium_pct', 'medium_risk_pct'],
        'high_pct': ['high_pct', 'high_risk_pct'],
    }

    for key in key_variants.get(base_name, [base_name]):
        if key in metrics:
            return metrics[key]
    return 0


def format_pct(value):
    """Format a score in 0-1 range as a percentage string."""
    return f"{value * 100:.1f}%"


def format_threshold(value):
    """Format decision threshold values for display."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def get_best_model(eval_results, metric_name):
    """Return the best model name and metrics for a given metric."""
    best_name = max(eval_results, key=lambda model_name: eval_results[model_name].get(metric_name, 0))
    return best_name, eval_results[best_name]


def get_single_sample_shap(system, model_name, feature_values):
    """
    Compute SHAP values for a single module prediction.

    Returns a dict with ranked feature contributions and model output baseline.
    """
    if shap is None:
        return None, "SHAP library is not installed. Install it with `pip install shap`."

    try:
        model = system.models[model_name]
        feature_names = system.data.get('feature_names', ['LOC', 'CBO', 'RFC', 'WMC'])
        input_df = pd.DataFrame([feature_values], columns=feature_names)

        if system.scaler is not None:
            scaled_values = system.scaler.transform(input_df)
            model_input = pd.DataFrame(scaled_values, columns=feature_names)
        else:
            model_input = input_df.copy()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            contribution_values = shap_values[1][0]
            base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            contribution_values = shap_values[0, :, 1]
            base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        elif isinstance(shap_values, np.ndarray):
            contribution_values = shap_values[0]
            base_value = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            return None, "Unable to parse SHAP output for this model."

        explain_df = pd.DataFrame({
            'feature': feature_names,
            'input_value': feature_values,
            'shap_value': contribution_values
        })
        explain_df['abs_shap'] = explain_df['shap_value'].abs()
        explain_df = explain_df.sort_values('abs_shap', ascending=False).reset_index(drop=True)

        raw_output = float(base_value + explain_df['shap_value'].sum())
        return {
            'explain_df': explain_df,
            'base_value': float(base_value),
            'raw_output': raw_output
        }, None
    except Exception as exc:
        return None, f"SHAP explanation failed: {exc}"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("## 🐛 Software Bug Prediction System")
    st.markdown("Proactive defect detection using Machine Learning")
    
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        [
            "Dashboard",
            "Single Prediction",
            "Upload Analysis",
            "Model Comparison",
            "System Information"
        ]
    )
    
    # Load system
    system = load_system()
    
    if system is None:
        st.error("Failed to load the Bug Prediction System")
        return
    
    # Page routing
    if page == "Dashboard":
        show_dashboard(system)
    elif page == "Single Prediction":
        show_single_prediction(system)
    elif page == "Upload Analysis":
        show_upload_analysis(system)
    elif page == "Model Comparison":
        show_model_comparison(system)
    elif page == "System Information":
        show_system_info(system)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

def show_dashboard(system):
    """Main dashboard with key metrics and visualizations"""
    st.header("System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get metrics
    eval_results = system.evaluation_results
    best_recall_name, best_recall_metrics = get_best_model(eval_results, 'recall')
    best_accuracy_name, best_accuracy_metrics = get_best_model(eval_results, 'accuracy')
    best_f1_name, best_f1_metrics = get_best_model(eval_results, 'f1')
    baseline_recall = eval_results.get('Baseline RF', {}).get('recall', 0)
    recall_gain = best_recall_metrics.get('recall', 0) - baseline_recall
    
    with col1:
        st.metric(
            "Best Recall",
            format_pct(best_recall_metrics.get('recall', 0)),
            f"{best_recall_name} | {recall_gain * 100:+.1f} pts vs Baseline RF",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Best Accuracy",
            format_pct(best_accuracy_metrics.get('accuracy', 0)),
            best_accuracy_name,
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Best F1-Score",
            format_pct(best_f1_metrics.get('f1', 0)),
            best_f1_name,
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Models Trained",
            "4",
            "RF, LR, NB, Baseline"
        )
    
    st.markdown("---")
    
    # Create comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Recall Comparison")
        recall_data = {
            'Model': list(eval_results.keys()),
            'Recall': [eval_results[m].get('recall', 0) for m in eval_results.keys()]
        }
        df_recall = pd.DataFrame(recall_data)
        
        fig = px.bar(
            df_recall,
            x='Model',
            y='Recall',
            title='Recall Score by Model',
            color='Recall',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Accuracy Comparison")
        acc_data = {
            'Model': list(eval_results.keys()),
            'Accuracy': [eval_results[m].get('accuracy', 0) for m in eval_results.keys()]
        }
        df_acc = pd.DataFrame(acc_data)
        
        fig = px.bar(
            df_acc,
            x='Model',
            y='Accuracy',
            title='Accuracy Score by Model',
            color='Accuracy',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
        <strong>Best for Bug Detection:</strong><br>
        {best_recall_name}<br>
        Recall: {format_pct(best_recall_metrics.get('recall', 0))}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>Class Imbalance Handled:</strong><br>
        SMOTE Applied to Balance Dataset<br>
        632 samples per class
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <strong>Test Set Size:</strong><br>
        200 modules evaluated<br>
        Stratified split (80-20)
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================

def show_single_prediction(system):
    """Single module prediction interface"""
    st.header("Single Module Prediction")
    st.write("Enter software metrics for a single module to predict bug risk")
    
    # Get predictor for Improved RF
    predictor = system.get_prediction_engine('Improved RF')
    st.caption(f"Saved decision threshold for Improved RF: {predictor.decision_threshold:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loc = st.number_input(
            "Lines of Code (LOC)",
            min_value=0,
            max_value=10000,
            value=150,
            step=10
        )
    
    with col2:
        cbo = st.number_input(
            "Coupling Between Objects (CBO)",
            min_value=0,
            max_value=100,
            value=8,
            step=1
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        rfc = st.number_input(
            "Response for Class (RFC)",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )
    
    with col4:
        wmc = st.number_input(
            "Weighted Methods per Class (WMC)",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )
    
    if st.button("Predict", key="predict_single", use_container_width=True):
        # Create feature array
        features = np.array([loc, cbo, rfc, wmc])
        
        # Get prediction
        result = predictor.predict_single(features)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result['prediction'] == 1:
                st.markdown("""
                <div class="danger-box">
                <strong style="font-size: 1.5rem;">BUGGY</strong><br>
                Likely to have defects
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <strong style="font-size: 1.5rem;">CLEAN</strong><br>
                Unlikely to have defects
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Defect Probability",
                f"{result['probability']:.4f}",
                f"{result['probability']*100:.1f}%"
            )
        
        with col3:
            risk_color = {
                'LOW': '🟢 LOW',
                'MEDIUM': '🟡 MEDIUM',
                'HIGH': '🔴 HIGH'
            }
            st.metric(
                "Risk Level",
                risk_color.get(result['risk_level'], 'UNKNOWN')
            )
        
        # Risk description
        st.markdown("---")
        st.subheader("Risk Assessment")
        
        risk_descriptions = {
            'LOW': 'This module has low risk of containing bugs. Standard testing is recommended.',
            'MEDIUM': 'This module has moderate risk. Additional code review and testing recommended.',
            'HIGH': 'This module has high risk of bugs. Prioritize testing and code review efforts.'
        }
        
        st.info(risk_descriptions.get(result['risk_level'], ''))
        st.caption(f"Classification label uses threshold {result['decision_threshold']:.3f}")

        st.markdown("---")
        st.subheader("SHAP Explainability Panel")
        st.write("This panel explains how each metric pushed the prediction toward Buggy or Clean.")

        shap_result, shap_error = get_single_sample_shap(system, 'Improved RF', features)
        if shap_error:
            st.warning(shap_error)
        elif shap_result is not None:
            explain_df = shap_result['explain_df']

            fig_shap = px.bar(
                explain_df.sort_values('shap_value'),
                x='shap_value',
                y='feature',
                orientation='h',
                color='shap_value',
                color_continuous_scale='RdBu_r',
                title='Feature Contribution to This Prediction (SHAP)',
                labels={'shap_value': 'SHAP Value (impact on model output)', 'feature': 'Metric'}
            )
            fig_shap.add_vline(x=0, line_width=1, line_dash='dash', line_color='gray')
            st.plotly_chart(fig_shap, use_container_width=True)

            top_positive = explain_df.sort_values('shap_value', ascending=False).head(1).iloc[0]
            top_negative = explain_df.sort_values('shap_value', ascending=True).head(1).iloc[0]

            col_pos, col_neg, col_base = st.columns(3)
            with col_pos:
                st.metric(
                    "Strongest Push to Buggy",
                    top_positive['feature'],
                    f"{top_positive['shap_value']:+.4f}"
                )
            with col_neg:
                st.metric(
                    "Strongest Push to Clean",
                    top_negative['feature'],
                    f"{top_negative['shap_value']:+.4f}"
                )
            with col_base:
                st.metric(
                    "Explainer Base Value",
                    f"{shap_result['base_value']:.4f}",
                    f"Output {shap_result['raw_output']:.4f}"
                )

            # Static rendering avoids jitter/shaking from interactive dataframe scrollbars.
            st.table(
                explain_df[['feature', 'input_value', 'shap_value']]
                .assign(
                    input_value=lambda df: df['input_value'].map(lambda x: f"{x:.4f}"),
                    shap_value=lambda df: df['shap_value'].map(lambda x: f"{x:+.4f}")
                )
                .set_index('feature')
            )


# ============================================================================
# PAGE: BATCH PREDICTIONS
# ============================================================================

def show_upload_analysis(system):
    """Unified upload analysis page for prediction and risk review."""
    st.header("Upload Analysis")
    st.write("Upload a data file to analyze modules, predict bug risk, and review risk distribution.")
    
    # Template
    st.subheader("Required Metric Columns")
    template_df = pd.DataFrame({
        'LOC': [150, 200, 100],
        'CBO': [8, 12, 5],
        'RFC': [20, 25, 15],
        'WMC': [10, 15, 8]
    })
    st.dataframe(template_df)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose input file")
    with col2:
        st.markdown("**Model Used**")
        model_choice = st.selectbox(
            "Choose prediction model",
            ['Improved RF', 'Logistic Regression', 'Baseline RF', 'Naive Bayes'],
            index=0,
            key="upload_model_select"
        )

    st.info(f"Selected model for upload analysis: {model_choice}")
    
    if uploaded_file is not None:
        # Read file
        try:
            df, parser_used = read_uploaded_dataframe(uploaded_file)
            st.caption(f"Loaded file using: {parser_used}")
        except ValueError as exc:
            st.error(str(exc))
            return

        # Extract columns from flexible input schema
        feature_df, mapping, missing, source_df, valid_rows = prepare_prediction_input(df)
        if feature_df is None:
            st.error(
                "Could not extract all required metrics. "
                "Please include columns matching LOC, CBO, RFC, and WMC."
            )
            if mapping:
                st.info(f"Detected columns: {mapping}")
            if missing:
                st.warning(f"Missing or non-numeric metrics: {missing}")
            return

        st.success(f"Detected metric columns: {mapping}")
        invalid_count = int((~valid_rows).sum())
        if invalid_count:
            st.warning(f"Found {invalid_count} row(s) with missing or non-numeric metric values. They will remain visible and be marked as not analyzed.")

        # Get predictor
        predictor = system.get_prediction_engine(model_choice)
        st.caption(f"Saved decision threshold for {model_choice}: {predictor.decision_threshold:.3f}")
        
        # Predict
        if st.button("Predict All Modules"):
            valid_feature_df = feature_df.loc[valid_rows].reset_index(drop=True)
            predictions_df = predictor.predict_batch(valid_feature_df.values)

            results_df = source_df.copy()
            for col in ['LOC', 'CBO', 'RFC', 'WMC']:
                if col not in results_df.columns:
                    results_df[col] = feature_df[col]

            results_df['analysis_status'] = np.where(valid_rows, 'Analyzed', 'Invalid input')
            results_df['prediction'] = np.nan
            results_df['probability'] = np.nan
            results_df['risk_level'] = np.nan
            results_df['label'] = np.nan
            results_df['decision_threshold'] = np.nan

            valid_indices = results_df.index[valid_rows]
            results_df.loc[valid_indices, ['prediction', 'probability', 'risk_level', 'label', 'decision_threshold']] = predictions_df[['prediction', 'probability', 'risk_level', 'label', 'decision_threshold']].values
            
            # Display results
            st.subheader(f"Module-by-Module Analysis - {model_choice}")
            st.dataframe(results_df, use_container_width=True)
            
            # Risk distribution
            st.subheader("Risk Distribution Summary")
            analyzed_results = results_df[results_df['analysis_status'] == 'Analyzed']
            risk_counts = analyzed_results['risk_level'].value_counts()
            total = len(analyzed_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                low_count = risk_counts.get('LOW', 0)
                low_pct = (low_count / total * 100) if total > 0 else 0
                st.metric("Low Risk", low_count, f"{low_pct:.1f}%")
            with col2:
                medium_count = risk_counts.get('MEDIUM', 0)
                medium_pct = (medium_count / total * 100) if total > 0 else 0
                st.metric("Medium Risk", medium_count, f"{medium_pct:.1f}%")
            with col3:
                high_count = risk_counts.get('HIGH', 0)
                high_pct = (high_count / total * 100) if total > 0 else 0
                st.metric("High Risk", high_count, f"{high_pct:.1f}%")

            st.subheader("High-Risk Modules")
            high_risk_modules = analyzed_results[analyzed_results['risk_level'] == 'HIGH']
            if len(high_risk_modules) > 0:
                st.dataframe(
                    high_risk_modules[['LOC', 'CBO', 'RFC', 'WMC', 'probability', 'risk_level']],
                    use_container_width=True
                )
            else:
                st.info("No high-risk modules were found in the analyzed rows.")
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download predictions as CSV",
                csv,
                f"predictions_{model_choice.replace(' ', '_')}.csv",
                "text/csv"
            )


# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================

def show_model_comparison(system):
    """Compare all trained models"""
    st.header("Model Comparison")
    
    eval_results = system.evaluation_results
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in eval_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': format_pct(metrics.get('accuracy', 0)),
            'Precision': format_pct(metrics.get('precision', 0)),
            'Recall': format_pct(metrics.get('recall', 0)),
            'F1-Score': format_pct(metrics.get('f1', 0)),
            'AUC-ROC': format_pct(metrics.get('auc_roc', 0)),
            'Threshold': format_threshold(metrics.get('decision_threshold')),
            'Optimized For': metrics.get('threshold_optimized_for', 'N/A')
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    chart_df = pd.DataFrame([
        {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0) * 100,
            'Precision': metrics.get('precision', 0) * 100,
            'Recall': metrics.get('recall', 0) * 100,
            'F1-Score': metrics.get('f1', 0) * 100,
            'AUC-ROC': metrics.get('auc_roc', 0) * 100
        }
        for model_name, metrics in eval_results.items()
    ])
    
    # Display table
    st.subheader("Performance Metrics")
    # Static table avoids UI jitter from dynamic dataframe scroll rendering.
    st.table(df_comparison.set_index('Model'))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            chart_df,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title='Metric Comparison by Model (%)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            chart_df,
            x='Model',
            y='Recall',
            title='Recall Comparison (Bug Detection %)',
            color='Recall',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("Recommendations")
    best_recall_name, best_recall_metrics = get_best_model(eval_results, 'recall')
    best_balanced_name, best_balanced_metrics = get_best_model(eval_results, 'f1')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
        <strong>Best for Bug Detection:</strong><br>
        {best_recall_name}<br>
        Recall: {format_pct(best_recall_metrics.get('recall', 0))}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <strong>Most Balanced:</strong><br>
        {best_balanced_name}<br>
        F1-Score: {format_pct(best_balanced_metrics.get('f1', 0))}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: RISK ANALYSIS
# ============================================================================

def show_risk_analysis(system):
    """Risk distribution and analysis with user input"""
    st.header("Risk Analysis")
    
    # Display risk thresholds
    st.subheader("Risk Classification Thresholds")
    st.markdown("""
    - **LOW RISK (0.0 - 0.3):** Unlikely to have bugs - Standard testing
    - **MEDIUM RISK (0.3 - 0.7):** Moderate chance - Code review + Testing
    - **HIGH RISK (0.7 - 1.0):** Likely to have bugs - Priority testing & review
    """)
    
    st.markdown("---")
    
    # TAB 1: DATASET-LEVEL REPORT
    st.subheader("📊 Dataset-Level Risk Distribution (Test Set)")
    
    # Load risk classification report
    try:
        with open('risk_classification_report.json', 'r') as f:
            risk_data = json.load(f)
        
        # Risk distribution by model
        risk_comparison = []
        for model, metrics in risk_data.items():
            risk_comparison.append({
                'Model': model,
                'Low': get_risk_value(metrics, 'low'),
                'Medium': get_risk_value(metrics, 'medium'),
                'High': get_risk_value(metrics, 'high')
            })
        
        df_risk = pd.DataFrame(risk_comparison)
        
        fig = px.bar(
            df_risk,
            x='Model',
            y=['Low', 'Medium', 'High'],
            title='Module Risk Distribution (200 Test Samples)',
            barmode='stack',
            color_discrete_map={'Low': '#51CF66', 'Medium': '#FFD93D', 'High': '#FF6B6B'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        st.subheader("Detailed Statistics by Model")
        for model, metrics in risk_data.items():
            with st.expander(f"📊 {model}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    low_value = get_risk_value(metrics, 'low')
                    low_pct = get_risk_value(metrics, 'low_pct')
                    st.metric("Low Risk", low_value, f"{low_pct:.1f}%")
                with col2:
                    medium_value = get_risk_value(metrics, 'medium')
                    medium_pct = get_risk_value(metrics, 'medium_pct')
                    st.metric("Medium Risk", medium_value, f"{medium_pct:.1f}%")
                with col3:
                    high_value = get_risk_value(metrics, 'high')
                    high_pct = get_risk_value(metrics, 'high_pct')
                    st.metric("High Risk", high_value, f"{high_pct:.1f}%")
    except:
        st.warning("Dataset-level risk classification report not found")
    
    st.markdown("---")
    
    # TAB 2: USER UPLOAD - CUSTOM RISK ANALYSIS
    st.subheader("📁 Custom Risk Analysis (Upload Your Data)")
    st.write("Upload a data file with software metrics to analyze risk distribution for your modules")
    
    # Template
    template_cols = st.columns(2)
    with template_cols[0]:
        st.markdown("**Required Metric Columns:**")
        template_df = pd.DataFrame({
            'LOC': [150, 200, 100],
            'CBO': [8, 12, 5],
            'RFC': [20, 25, 15],
            'WMC': [10, 15, 8]
        })
        st.dataframe(template_df, use_container_width=True)
    
    with template_cols[1]:
        st.markdown("**Select Model for Analysis:**")
        model_choice = st.radio(
            "Choose prediction model",
            ['Improved RF', 'Logistic Regression', 'Baseline RF', 'Naive Bayes'],
            key="risk_model_select"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Upload input file", key="risk_upload")
    
    if uploaded_file is not None:
        st.markdown("---")
        
        # Read file
        try:
            df, parser_used = read_uploaded_dataframe(uploaded_file)
            st.caption(f"Loaded file using: {parser_used}")
        except ValueError as exc:
            st.error(str(exc))
            return

        # Extract columns from flexible input schema
        feature_df, mapping, missing, source_df, valid_rows = prepare_prediction_input(df)
        if feature_df is None:
            st.error(
                "Could not extract all required metrics. "
                "Please include columns matching LOC, CBO, RFC, and WMC."
            )
            if mapping:
                st.info(f"Detected columns: {mapping}")
            if missing:
                st.warning(f"Missing or non-numeric metrics: {missing}")
            return

        st.success(f"Detected metric columns: {mapping}")
        invalid_count = int((~valid_rows).sum())
        if invalid_count:
            st.warning(f"Found {invalid_count} row(s) with missing or non-numeric metric values. They will remain visible and be marked as not analyzed.")
        
        # Get predictor for selected model
        predictor = system.get_prediction_engine(model_choice)
        
        # Predict
        if st.button("Analyze Risk Distribution", key="analyze_risk"):
            with st.spinner("Analyzing risk distribution..."):
                valid_feature_df = feature_df.loc[valid_rows].reset_index(drop=True)
                predictions_df = predictor.predict_batch(valid_feature_df.values)

                results_df = source_df.copy()
                for col in ['LOC', 'CBO', 'RFC', 'WMC']:
                    if col not in results_df.columns:
                        results_df[col] = feature_df[col]

                results_df['analysis_status'] = np.where(valid_rows, 'Analyzed', 'Invalid input')
                results_df['prediction'] = np.nan
                results_df['probability'] = np.nan
                results_df['risk_level'] = np.nan
                results_df['label'] = np.nan

                valid_indices = results_df.index[valid_rows]
                results_df.loc[valid_indices, ['prediction', 'probability', 'risk_level', 'label']] = predictions_df[['prediction', 'probability', 'risk_level', 'label']].values
                
                # Display predictions
                st.subheader(f"Module-by-Module Analysis - {model_choice}")
                st.dataframe(results_df, use_container_width=True)
                
                # Risk distribution for uploaded file
                st.subheader("Risk Distribution Summary")
                
                analyzed_results = results_df[results_df['analysis_status'] == 'Analyzed']
                risk_counts = analyzed_results['risk_level'].value_counts()
                total = len(analyzed_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    low_count = risk_counts.get('LOW', 0)
                    low_pct = (low_count / total * 100) if total > 0 else 0
                    st.metric("🟢 Low Risk", low_count, f"{low_pct:.1f}%")
                
                with col2:
                    med_count = risk_counts.get('MEDIUM', 0)
                    med_pct = (med_count / total * 100) if total > 0 else 0
                    st.metric("🟡 Medium Risk", med_count, f"{med_pct:.1f}%")
                
                with col3:
                    high_count = risk_counts.get('HIGH', 0)
                    high_pct = (high_count / total * 100) if total > 0 else 0
                    st.metric("🔴 High Risk", high_count, f"{high_pct:.1f}%")
                
                # Risk distribution pie chart
                st.subheader("Risk Distribution Breakdown")
                
                risk_summary = pd.DataFrame({
                    'Risk Level': ['Low', 'Medium', 'High'],
                    'Count': [
                        risk_counts.get('LOW', 0),
                        risk_counts.get('MEDIUM', 0),
                        risk_counts.get('HIGH', 0)
                    ]
                })
                
                fig_pie = px.pie(
                    risk_summary,
                    values='Count',
                    names='Risk Level',
                    color='Risk Level',
                    color_discrete_map={'Low': '#51CF66', 'Medium': '#FFD93D', 'High': '#FF6B6B'},
                    title=f"Risk Distribution for Your Modules ({total} total)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # High-risk modules highlight
                high_risk_modules = analyzed_results[analyzed_results['risk_level'] == 'HIGH']
                if len(high_risk_modules) > 0:
                    st.subheader("⚠️ High-Risk Modules (Priority Testing)")
                    st.dataframe(
                        high_risk_modules[['LOC', 'CBO', 'RFC', 'WMC', 'probability', 'risk_level']],
                        use_container_width=True
                    )
                    st.warning(f"Found {len(high_risk_modules)} high-risk module(s) requiring priority testing")
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Risk Analysis Results",
                    csv,
                    f"risk_analysis_{model_choice.replace(' ', '_')}.csv",
                    "text/csv"
                )


# ============================================================================
# PAGE: SYSTEM INFORMATION
# ============================================================================

def show_system_info(system):
    """System architecture and information"""
    st.header("System Information")
    eval_results = system.evaluation_results
    best_recall_name, best_recall_metrics = get_best_model(eval_results, 'recall')
    best_accuracy_name, best_accuracy_metrics = get_best_model(eval_results, 'accuracy')
    
    st.subheader("Project Overview")
    st.markdown("""
    **Proactive Bug Prediction System** - Identifies bug-prone software modules using ML
    
    ### Core Capabilities
    - Multi-model bug prediction (Baseline RF, Improved RF, Logistic Regression, Naive Bayes)
    - Threshold-tuned classification for better recall/precision control
    - Single-module prediction with SHAP explainability panel
    - Batch CSV upload analysis with risk-level triage
    - Interactive model-comparison and risk-distribution views
    """)
    
    st.subheader("Trained Models")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### Random Forest Models
        - **Baseline RF:** No optimization
        - **Improved RF:** SMOTE + Hyperparameter tuning
        """)
    
    with col2:
        st.info("""
        ### Linear Models
        - **Logistic Regression:** Linear classifier
        - **Naive Bayes:** Probabilistic classifier
        """)
    
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", "997")
    with col2:
        st.metric("Features", "4 (LOC, CBO, RFC, WMC)")
    with col3:
        st.metric("Classes", "2 (Clean/Buggy)")
    
    st.subheader("Software Metrics Explained")
    metrics_info = {
        'LOC': 'Lines of Code - Total lines in the class',
        'CBO': 'Coupling Between Objects - Number of classes coupled',
        'RFC': 'Response For Class - Number of methods called',
        'WMC': 'Weighted Methods per Class - Complexity measure'
    }
    
    for metric, desc in metrics_info.items():
        st.markdown(f"**{metric}:** {desc}")
    
    st.subheader("Key Results")
    st.success(f"""
    ✓ **Best recall model:** {best_recall_name} at {format_pct(best_recall_metrics.get('recall', 0))}
    ✓ **Best accuracy model:** {best_accuracy_name} at {format_pct(best_accuracy_metrics.get('accuracy', 0))}
    ✓ **4 models trained:** comparison framework is available
    ✓ **Risk categorization:** actionable low / medium / high prioritization
    """)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
