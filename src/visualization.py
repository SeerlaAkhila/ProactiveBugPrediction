"""
PHASE 7: VISUALIZATION MODULE
==============================

Creates comprehensive visualizations for model comparison:
- Accuracy comparison
- Recall comparison (most important for bug detection)
- Precision/F1-Score comparison
- Confusion matrices
- ROC curves
- Feature importance
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import joblib
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class BugPredictionVisualizer:
    """Handles all visualizations for bug prediction system"""
    
    def __init__(self, models_dir='models', output_dir='visualizations'):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics and models
        with open(f'{models_dir}/comparison_metrics.json', 'r') as f:
            self.metrics = json.load(f)
        
        self.scaler = joblib.load(f'{models_dir}/scaler.pkl')
        self.models = {}
        for metric in self.metrics:
            model_name = metric['model_name'].replace(' ', '_')
            self.models[metric['model_name']] = joblib.load(
                f'{models_dir}/{model_name}.pkl'
            )
    
    # =========================================================================
    # VISUALIZATION 1: METRIC COMPARISON BARS
    # =========================================================================
    
    def plot_metric_comparison(self):
        """Create comparison bar plots for all metrics"""
        print("📊 Generating metric comparison plots...")
        
        # Prepare data
        model_names = [m['model_name'] for m in self.metrics]
        accuracy = [m['accuracy'] for m in self.metrics]
        precision = [m['precision'] for m in self.metrics]
        recall = [m['recall'] for m in self.metrics]
        f1_scores = [m['f1'] for m in self.metrics]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison - Performance Metrics', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].bar(model_names, accuracy, color='steelblue', alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Accuracy', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('Score', fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].axhline(y=0.8, color='red', linestyle='--', label='80% threshold', alpha=0.5)
        for i, v in enumerate(accuracy):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        # Precision
        axes[0, 1].bar(model_names, precision, color='coral', alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Precision (Low False Positives)', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(precision):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        # Recall ⭐ MOST IMPORTANT FOR BUG DETECTION
        axes[1, 0].bar(model_names, recall, color='limegreen', alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('Recall ⭐ (Bug Detection - MOST IMPORTANT)', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('Score', fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        for i, v in enumerate(recall):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        # F1-Score
        axes[1, 1].bar(model_names, f1_scores, color='mediumpurple', alpha=0.8, edgecolor='black')
        axes[1, 1].set_title('F1-Score (Balanced Metric)', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_metric_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_metric_comparison.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 2: RECALL IMPROVEMENT HIGHLIGHT
    # =========================================================================
    
    def plot_recall_improvement(self):
        """Highlight recall improvement (critical for bug detection)"""
        print("📊 Generating recall improvement plot...")
        
        model_names = [m['model_name'] for m in self.metrics]
        recall = [m['recall'] for m in self.metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#ff6b6b' if i == 0 else '#51cf66' for i in range(len(model_names))]
        bars = ax.bar(model_names, recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_title('Recall Comparison: Bug Detection Performance ⭐', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(recall) + 0.15])
        ax.axhline(y=0.5, color='gray', linestyle='--', label='50% Threshold', alpha=0.7)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, recall)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.4f}\n({val*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Highlight improvement
        if len(recall) > 1:
            improvement = recall[1] - recall[0]
            improvement_pct = (improvement / recall[0]) * 100
            ax.text(0.5, max(recall) * 0.95, 
                   f'Improvement: +{improvement:.4f} (+{improvement_pct:.1f}%)',
                   ha='center', fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontweight='bold')
        
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_recall_improvement.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_recall_improvement.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 3: CONFUSION MATRICES
    # =========================================================================
    
    def plot_confusion_matrices(self, y_test, predictions_dict):
        """Plot confusion matrices for all models"""
        print("📊 Generating confusion matrices...")
        
        n_models = len(self.metrics)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Confusion Matrices - Model Predictions', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for idx, metric in enumerate(self.metrics):
            y_pred = predictions_dict[metric['model_name']]['pred']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
            axes[idx].set_title(f"{metric['model_name']}\n(Accuracy: {metric['accuracy']:.3f}, Recall: {metric['recall']:.3f})",
                               fontweight='bold')
            axes[idx].set_ylabel('True Label', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontweight='bold')
            axes[idx].set_xticklabels(['Clean', 'Bug'], rotation=0)
            axes[idx].set_yticklabels(['Clean', 'Bug'], rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 03_confusion_matrices.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 4: ROC CURVES
    # =========================================================================
    
    def plot_roc_curves(self, y_test, predictions_dict):
        """Plot ROC curves for all models"""
        print("📊 Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, metric in enumerate(self.metrics):
            y_proba = predictions_dict[metric['model_name']]['proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2.5,
                   label=f"{metric['model_name']} (AUC = {roc_auc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
        ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 04_roc_curves.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 5: FEATURE IMPORTANCE
    # =========================================================================
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest models"""
        print("📊 Generating feature importance plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Feature Importance - Random Forest Models', fontsize=13, fontweight='bold')
        
        feature_names = ['LOC', 'CBO', 'RFC', 'WMC']
        models_to_plot = ['Baseline RF', 'Improved RF']
        
        for ax_idx, model_name in enumerate(models_to_plot):
            if model_name in self.models:
                model = self.models[model_name]
                importance = model.feature_importances_
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                sorted_features = [feature_names[i] for i in sorted_idx]
                sorted_importance = importance[sorted_idx]
                
                colors_importance = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
                axes[ax_idx].barh(sorted_features, sorted_importance, color=colors_importance, edgecolor='black')
                axes[ax_idx].set_title(f'{model_name}', fontweight='bold')
                axes[ax_idx].set_xlabel('Importance Score', fontweight='bold')
                
                for i, v in enumerate(sorted_importance):
                    axes[ax_idx].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 05_feature_importance.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 6: MODEL RANKINGS
    # =========================================================================
    
    def plot_model_rankings(self):
        """Rank models by different metrics"""
        print("📊 Generating model rankings...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Rankings by Different Metrics', fontsize=13, fontweight='bold')
        
        metrics_to_plot = [
            ('accuracy', 'Accuracy', axes[0, 0]),
            ('precision', 'Precision', axes[0, 1]),
            ('recall', 'Recall', axes[1, 0]),
            ('f1', 'F1-Score', axes[1, 1])
        ]
        
        for metric_key, metric_label, ax in metrics_to_plot:
            data = sorted(self.metrics, key=lambda x: x[metric_key], reverse=True)
            model_names = [m['model_name'] for m in data]
            scores = [m[metric_key] for m in data]
            
            colors_ranking = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_names)))
            bars = ax.barh(model_names, scores, color=colors_ranking, edgecolor='black')
            ax.set_xlim([0, 1])
            ax.set_title(f'Ranked by {metric_label}', fontweight='bold')
            ax.set_xlabel('Score', fontweight='bold')
            
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_model_rankings.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 06_model_rankings.png")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 7: BASELINE VS IMPROVED COMPARISON
    # =========================================================================
    
    def plot_baseline_vs_improved(self):
        """Side-by-side comparison of Baseline vs Improved model"""
        print("📊 Generating baseline vs improved comparison...")
        
        baseline = [m for m in self.metrics if m['model_name'] == 'Baseline RF'][0]
        improved = [m for m in self.metrics if m['model_name'] == 'Improved RF'][0]
        
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        baseline_scores = [baseline[m] for m in metrics_list]
        improved_scores = [improved[m] for m in metrics_list]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline RF',
                      color='#FF6B6B', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, improved_scores, width, label='Improved RF',
                      color='#4ECDC4', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_title('Baseline vs Improved Random Forest', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_baseline_vs_improved.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07_baseline_vs_improved.png")
        plt.close()
    
    # =========================================================================
    # GENERATE ALL VISUALIZATIONS
    # =========================================================================
    
    def generate_all(self, y_test, predictions_dict):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("PHASE 7: VISUALIZATION")
        print("=" * 80)
        
        self.plot_metric_comparison()
        self.plot_recall_improvement()
        self.plot_confusion_matrices(y_test, predictions_dict)
        self.plot_roc_curves(y_test, predictions_dict)
        self.plot_feature_importance()
        self.plot_model_rankings()
        self.plot_baseline_vs_improved()
        
        print("\n" + "=" * 80)
        print("✓ PHASE 7 COMPLETED: All visualizations saved!")
        print("=" * 80)
        print(f"\n📁 Visualizations saved to: {self.output_dir}/")
        print("\nGenerated files:")
        for i, filename in enumerate([
            "01_metric_comparison.png",
            "02_recall_improvement.png",
            "03_confusion_matrices.png",
            "04_roc_curves.png",
            "05_feature_importance.png",
            "06_model_rankings.png",
            "07_baseline_vs_improved.png"
        ], 1):
            print(f"  {i}. {filename}")


# ============================================================================
# HELPER: Load predictions for visualization
# ============================================================================

def load_predictions_for_visualization(X_test, y_test, models_dict):
    """Load model predictions for visualization"""
    predictions_dict = {}
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        predictions_dict[model_name] = {
            'pred': y_pred,
            'proba': y_pred_proba
        }
    
    return predictions_dict


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = BugPredictionVisualizer()
    
    # Load test data and models
    import joblib
    scaler = joblib.load('models/scaler.pkl')
    data = pd.read_csv('data/processed/cleaned_dataset.csv')
    X = data[['LOC', 'CBO', 'RFC', 'WMC']]
    y = data['defect']
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test = scaler.transform(X_test)
    
    # Load predictions
    predictions_dict = load_predictions_for_visualization(X_test, y_test, visualizer.models)
    
    # Generate all visualizations
    visualizer.generate_all(y_test, predictions_dict)
