"""
PHASE 10 & 12: BUG PREDICTION SYSTEM - INTERACTIVE STREAMLIT UI
================================================================

Complete functional system with web interface for:
- Input: Software metrics (LOC, CBO, RFC, WMC)
- Output: Defect prediction, probability, risk level
- Visualization: Model performance, risk distribution
- Comparison: Baseline vs Improved model

FEATURES:
1. Single/Batch Predictions
2. Model Comparison Dashboard
3. Risk Analysis & Visualization
4. Performance Metrics Display
5. Downloadable Reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os
from pathlib import Path

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

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🐛 Software Bug Prediction System")
        st.markdown("Proactive defect detection using Machine Learning")
    with col2:
        st.info("📊 Phase 10 & 12 Complete")
    
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        [
            "Dashboard",
            "Single Prediction",
            "Batch Predictions",
            "Model Comparison",
            "Risk Analysis",
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
    elif page == "Batch Predictions":
        show_batch_predictions(system)
    elif page == "Model Comparison":
        show_model_comparison(system)
    elif page == "Risk Analysis":
        show_risk_analysis(system)
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
    improved_rf = eval_results.get('Improved RF', {})
    baseline_rf = eval_results.get('Baseline RF', {})
    
    with col1:
        st.metric(
            "Best Recall",
            f"{improved_rf.get('recall', 0):.4f}",
            f"+{(improved_rf.get('recall', 0) - baseline_rf.get('recall', 0)):.4f}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Best Accuracy",
            f"{baseline_rf.get('accuracy', 0):.4f}",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Best F1-Score",
            f"{improved_rf.get('f1', 0):.4f}",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Models Trained",
            "4️⃣",
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
        st.markdown("""
        <div class="success-box">
        <strong>Best for Bug Detection:</strong><br>
        Improved Random Forest<br>
        Recall: 0.4878 (+11.1%)
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


# ============================================================================
# PAGE: BATCH PREDICTIONS
# ============================================================================

def show_batch_predictions(system):
    """Batch prediction from CSV upload"""
    st.header("Batch Module Predictions")
    st.write("Upload a CSV file with multiple modules to predict bugs")
    
    # Template
    st.subheader("CSV Format Template")
    template_df = pd.DataFrame({
        'LOC': [150, 200, 100],
        'CBO': [8, 12, 5],
        'RFC': [20, 25, 15],
        'WMC': [10, 15, 8]
    })
    st.dataframe(template_df)
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read file
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['LOC', 'CBO', 'RFC', 'WMC']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return
        
        # Get predictor
        predictor = system.get_prediction_engine('Improved RF')
        
        # Predict
        if st.button("Predict All Modules"):
            predictions_df = predictor.predict_batch(df[required_cols].values)
            results_df = pd.concat([df, predictions_df], axis=1)
            
            # Display results
            st.subheader("Predictions")
            st.dataframe(results_df, use_container_width=True)
            
            # Risk distribution
            st.subheader("Risk Distribution Summary")
            risk_counts = results_df['risk_level'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Risk", risk_counts.get('LOW', 0))
            with col2:
                st.metric("Medium Risk", risk_counts.get('MEDIUM', 0))
            with col3:
                st.metric("High Risk", risk_counts.get('HIGH', 0))
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download predictions as CSV",
                csv,
                "predictions.csv",
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
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1', 0),
            'AUC-ROC': metrics.get('auc_roc', 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display table
    st.subheader("Performance Metrics")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_comparison,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title='Metric Comparison by Model',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_comparison,
            x='Model',
            y='Recall',
            title='Recall Comparison (Bug Detection)',
            color='Recall',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("Recommendations")
    best_recall = df_comparison.loc[df_comparison['Recall'].idxmax()]
    best_balanced = df_comparison.loc[df_comparison['F1-Score'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
        <strong>Best for Bug Detection:</strong><br>
        {best_recall['Model']}<br>
        Recall: {best_recall['Recall']:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <strong>Most Balanced:</strong><br>
        {best_balanced['Model']}<br>
        F1-Score: {best_balanced['F1-Score']:.4f}
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
                'Low': metrics.get('low', 0),
                'Medium': metrics.get('medium', 0),
                'High': metrics.get('high', 0)
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
                    st.metric("Low Risk", metrics.get('low', 0), f"{metrics.get('low_pct', 0):.1f}%")
                with col2:
                    st.metric("Medium Risk", metrics.get('medium', 0), f"{metrics.get('medium_pct', 0):.1f}%")
                with col3:
                    st.metric("High Risk", metrics.get('high', 0), f"{metrics.get('high_pct', 0):.1f}%")
    except:
        st.warning("Dataset-level risk classification report not found")
    
    st.markdown("---")
    
    # TAB 2: USER UPLOAD - CUSTOM RISK ANALYSIS
    st.subheader("📁 Custom Risk Analysis (Upload Your Data)")
    st.write("Upload a CSV file with software metrics to analyze risk distribution for your modules")
    
    # Template
    template_cols = st.columns(2)
    with template_cols[0]:
        st.markdown("**CSV Format Required:**")
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
    uploaded_file = st.file_uploader("Upload CSV file", type="csv", key="risk_upload")
    
    if uploaded_file is not None:
        st.markdown("---")
        
        # Read file
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['LOC', 'CBO', 'RFC', 'WMC']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return
        
        # Get predictor for selected model
        predictor = system.get_prediction_engine(model_choice)
        
        # Predict
        if st.button("Analyze Risk Distribution", key="analyze_risk"):
            with st.spinner("Analyzing risk distribution..."):
                predictions_df = predictor.predict_batch(df[required_cols].values)
                results_df = pd.concat([df, predictions_df], axis=1)
                
                # Display predictions
                st.subheader(f"Risk Predictions - {model_choice}")
                st.dataframe(results_df, use_container_width=True)
                
                # Risk distribution for uploaded file
                st.subheader("Risk Distribution Summary")
                
                risk_counts = results_df['risk_level'].value_counts()
                total = len(results_df)
                
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
                high_risk_modules = results_df[results_df['risk_level'] == 'HIGH']
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
    
    st.subheader("Project Overview")
    st.markdown("""
    **Proactive Bug Prediction System** - Identifies bug-prone software modules using ML
    
    ### Architecture
    - **Phase 6:** Model Comparison (4 models trained & evaluated)
    - **Phase 7:** Visualization (7 comprehensive charts)
    - **Phase 8:** Risk Classification (3-level risk system)
    - **Phase 9:** Modular System Design (Production-ready architecture)
    - **Phase 10:** Functional System (Complete integrated solution)
    - **Phase 12:** Streamlit UI (This interactive web application)
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
    st.success("""
    ✓ **Improved Recall:** +11.1% improvement in bug detection
    ✓ **SMOTE Applied:** Successfully balanced class distribution
    ✓ **4 Models Trained:** Comprehensive comparison framework
    ✓ **Risk Categorization:** Actionable risk levels for testing prioritization
    """)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
