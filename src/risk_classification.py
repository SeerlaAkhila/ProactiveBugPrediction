"""
PHASE 8: RISK CLASSIFICATION MODULE
====================================

Converts prediction probabilities into interpretable risk levels:
- Low Risk (0.0 - 0.3): Unlikely to have bugs
- Medium Risk (0.3 - 0.7): Moderate chance of bugs
- High Risk (0.7 - 1.0): Likely to have bugs

This helps with proactive testing and resource allocation.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RiskClassifier:
    """Classify modules into risk categories based on prediction probabilities"""
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,      # 0.0 - 0.3
        'medium': 0.7,   # 0.3 - 0.7
        'high': 1.0      # 0.7 - 1.0
    }
    
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler
        self.model = model
    
    def classify_risk(self, probability):
        """
        Classify risk level based on probability
        
        Args:
            probability: Defect probability (0-1)
            
        Returns:
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'
            risk_score: Risk score (0-1)
        """
        if probability < self.RISK_THRESHOLDS['low']:
            return 'LOW', probability
        elif probability < self.RISK_THRESHOLDS['medium']:
            return 'MEDIUM', probability
        else:
            return 'HIGH', probability
    
    def classify_batch(self, probabilities):
        """Classify multiple probabilities"""
        risks = []
        for prob in probabilities:
            risk_level, risk_score = self.classify_risk(prob)
            risks.append({
                'probability': prob,
                'risk_level': risk_level,
                'risk_score': risk_score
            })
        return risks
    
    @staticmethod
    def get_risk_metrics(risks):
        """Calculate statistics for risk distribution"""
        df = pd.DataFrame(risks)
        
        return {
            'total_modules': len(risks),
            'low_risk': (df['risk_level'] == 'LOW').sum(),
            'medium_risk': (df['risk_level'] == 'MEDIUM').sum(),
            'high_risk': (df['risk_level'] == 'HIGH').sum(),
            'low_risk_pct': (df['risk_level'] == 'LOW').sum() / len(risks) * 100,
            'medium_risk_pct': (df['risk_level'] == 'MEDIUM').sum() / len(risks) * 100,
            'high_risk_pct': (df['risk_level'] == 'HIGH').sum() / len(risks) * 100,
            'avg_probability': df['probability'].mean(),
            'max_probability': df['probability'].max(),
            'min_probability': df['probability'].min(),
        }


def apply_risk_classification_to_test_set(y_test, predictions_dict):
    """Apply risk classification to test predictions"""
    print("\n" + "=" * 80)
    print("PHASE 8: RISK CLASSIFICATION")
    print("=" * 80)
    
    risk_classifier = RiskClassifier()
    
    all_risk_results = {}
    all_risk_metrics = {}
    
    for model_name, predictions in predictions_dict.items():
        y_proba = predictions['proba']
        
        # Classify risks
        risks = risk_classifier.classify_batch(y_proba)
        risk_metrics = risk_classifier.get_risk_metrics(risks)
        
        all_risk_results[model_name] = risks
        all_risk_metrics[model_name] = risk_metrics
        
        # Print metrics
        print(f"\n{'─' * 80}")
        print(f"Risk Classification - {model_name}")
        print(f"{'─' * 80}")
        print(f"  Total Modules:   {risk_metrics['total_modules']}")
        print(f"  Low Risk:        {risk_metrics['low_risk']:3d} ({risk_metrics['low_risk_pct']:5.1f}%)")
        print(f"  Medium Risk:     {risk_metrics['medium_risk']:3d} ({risk_metrics['medium_risk_pct']:5.1f}%)")
        print(f"  High Risk:       {risk_metrics['high_risk']:3d} ({risk_metrics['high_risk_pct']:5.1f}%)")
        print(f"\n  Probability Stats:")
        print(f"    Average: {risk_metrics['avg_probability']:.4f}")
        print(f"    Min:     {risk_metrics['min_probability']:.4f}")
        print(f"    Max:     {risk_metrics['max_probability']:.4f}")
    
    return all_risk_results, all_risk_metrics


def generate_risk_report(all_risk_metrics, output_file='risk_classification_report.json'):
    """Generate comprehensive risk classification report"""
    # Convert numpy types to Python native types for JSON serialization
    json_safe_metrics = {}
    for model_name, metrics in all_risk_metrics.items():
        json_safe_metrics[model_name] = {
            k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v
            for k, v in metrics.items()
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_safe_metrics, f, indent=2)
    
    print(f"\n✓ Risk classification report saved: {output_file}")


def visualize_risk_distribution(all_risk_metrics, output_dir='visualizations'):
    """Visualize risk distribution across models"""
    print("\n📊 Generating risk distribution visualizations...")
    
    model_names = list(all_risk_metrics.keys())
    
    # Prepare data
    low_risk_counts = [all_risk_metrics[m]['low_risk'] for m in model_names]
    medium_risk_counts = [all_risk_metrics[m]['medium_risk'] for m in model_names]
    high_risk_counts = [all_risk_metrics[m]['high_risk'] for m in model_names]
    
    low_risk_pcts = [all_risk_metrics[m]['low_risk_pct'] for m in model_names]
    medium_risk_pcts = [all_risk_metrics[m]['medium_risk_pct'] for m in model_names]
    high_risk_pcts = [all_risk_metrics[m]['high_risk_pct'] for m in model_names]
    
    # Visualization 1: Stacked bar chart (absolute counts)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.6
    
    bars1 = ax.bar(x, low_risk_counts, width, label='Low Risk', color='#51CF66', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, medium_risk_counts, width, bottom=low_risk_counts,
                  label='Medium Risk', color='#FFD93D', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x, high_risk_counts, width,
                  bottom=[low_risk_counts[i] + medium_risk_counts[i] for i in range(len(model_names))],
                  label='High Risk', color='#FF6B6B', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Number of Modules', fontweight='bold', fontsize=11)
    ax.set_title('Module Risk Distribution by Model', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08a_risk_distribution_counts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08a_risk_distribution_counts.png")
    plt.close()
    
    # Visualization 2: Stacked percentage chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x, low_risk_pcts, width, label='Low Risk', color='#51CF66', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, medium_risk_pcts, width, bottom=low_risk_pcts,
                  label='Medium Risk', color='#FFD93D', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x, high_risk_pcts, width,
                  bottom=[low_risk_pcts[i] + medium_risk_pcts[i] for i in range(len(model_names))],
                  label='High Risk', color='#FF6B6B', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
    ax.set_title('Module Risk Distribution (%)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, model in enumerate(model_names):
        y_offset = 0
        for pct, color in [(low_risk_pcts[i], '#51CF66'),
                          (medium_risk_pcts[i], '#FFD93D'),
                          (high_risk_pcts[i], '#FF6B6B')]:
            if pct > 5:  # Only show if percentage is significant
                ax.text(i, y_offset + pct/2, f'{pct:.1f}%', ha='center', va='center',
                       fontweight='bold', fontsize=9)
            y_offset += pct
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08b_risk_distribution_pct.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08b_risk_distribution_pct.png")
    plt.close()
    
    # Visualization 3: High Risk Module Count Comparison
    fig, ax = plt.subplots(figsize=(11, 6))
    
    high_risk_data = [all_risk_metrics[m]['high_risk'] for m in model_names]
    colors = ['#FF6B6B' if count > 50 else '#FFB347' for count in high_risk_data]
    
    bars = ax.bar(model_names, high_risk_data, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of High-Risk Modules', fontweight='bold', fontsize=11)
    ax.set_title('High Risk Module Detection (Important for Testing Prioritization)',
                fontweight='bold', fontsize=13)
    ax.set_ylim([0, max(high_risk_data) * 1.2])
    
    for bar, count in zip(bars, high_risk_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08c_high_risk_modules.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08c_high_risk_modules.png")
    plt.close()


def create_risk_report_summary(all_risk_metrics):
    """Create a summary report of risk classification"""
    print("\n" + "=" * 80)
    print("RISK CLASSIFICATION SUMMARY")
    print("=" * 80)
    
    print("\nRisk Thresholds:")
    print("  • LOW RISK:    Probability 0.0 - 0.3  (Unlikely to have bugs)")
    print("  • MEDIUM RISK: Probability 0.3 - 0.7  (Moderate chance of bugs)")
    print("  • HIGH RISK:   Probability 0.7 - 1.0  (Likely to have bugs)")
    
    print("\nUsage:")
    print("  • Focus testing effort on HIGH RISK modules")
    print("  • Provide code review for MEDIUM RISK modules")
    print("  • Standard testing for LOW RISK modules")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/processed/cleaned_dataset.csv')
    X = data[['LOC', 'CBO', 'RFC', 'WMC']]
    y = data['defect']
    
    # Split and scale
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X[X.index.isin(X.index.difference(X_test.index))])
    X_test_scaled = scaler.transform(X_test)
    
    # Load models
    import json
    with open('models/comparison_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    models = {}
    for metric in metrics:
        model_name = metric['model_name'].replace(' ', '_')
        models[metric['model_name']] = joblib.load(f'models/{model_name}.pkl')
    
    # Generate predictions
    predictions_dict = {}
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        predictions_dict[model_name] = {
            'proba': y_pred_proba
        }
    
    # Apply risk classification
    all_risk_results, all_risk_metrics = apply_risk_classification_to_test_set(y_test, predictions_dict)
    
    # Generate report
    generate_risk_report(all_risk_metrics)
    
    # Visualize
    visualize_risk_distribution(all_risk_metrics)
    
    # Summary
    create_risk_report_summary(all_risk_metrics)
    
    print("\n" + "=" * 80)
    print("✓ PHASE 8 COMPLETED: Risk Classification")
    print("=" * 80)
