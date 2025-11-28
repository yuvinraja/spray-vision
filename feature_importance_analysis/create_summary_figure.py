"""
Create a publication-quality summary figure showing key findings
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Key results from analysis
targets = ['Angle\n(Shadow)', 'Length\n(Shadow)', 'Angle\n(Mie)', 'Length\n(Mie)']
features = ['Pc_bar', 'Tc_K', 'Pinj_bar', 'rho_kgm3', 'mu_Pas']

# Permutation importance (Test Set) - the most reliable measure
importance_matrix = np.array([
    [6.94, 14.10, 30.32, 25.23, 23.42],  # Angle (Shadow)
    [9.17, 5.09, 80.91, 2.75, 2.09],     # Length (Shadow)
    [3.77, 71.50, 17.30, 4.67, 2.77],    # Angle (Mie)
    [9.51, 4.88, 78.92, 3.16, 3.53]      # Length (Mie)
])

# Model performance
test_r2 = [0.874, 0.645, 0.920, 0.680]
test_rmse = [0.406, 28.82, 0.371, 26.10]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Main title
fig.suptitle('Feature Importance Analysis: Gradient Boosting Without Time\n' + 
             'Predicting Spray Characteristics from 5 Thermodynamic Features',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================================
# Panel 1: Heatmap of feature importance across targets
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

im = ax1.imshow(importance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax1.set_xticks(np.arange(len(features)))
ax1.set_yticks(np.arange(len(targets)))
ax1.set_xticklabels(features, fontsize=11)
ax1.set_yticklabels(targets, fontsize=11)

# Rotate x labels
plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")

# Add text annotations
for i in range(len(targets)):
    for j in range(len(features)):
        text = ax1.text(j, i, f'{importance_matrix[i, j]:.1f}%',
                       ha="center", va="center", color="black" if importance_matrix[i, j] < 50 else "white",
                       fontsize=10, fontweight='bold')

ax1.set_title('Permutation Feature Importance (Test Set) by Target Variable', 
             fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Input Features', fontsize=11, fontweight='bold')
ax1.set_ylabel('Target Variables', fontsize=11, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, aspect=30)
cbar.set_label('Importance (%)', fontsize=10, fontweight='bold')

# ============================================================================
# Panel 2: Overall feature importance (averaged)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

overall_importance = importance_matrix.mean(axis=0)
overall_std = importance_matrix.std(axis=0)

# Sort by importance
sorted_indices = np.argsort(overall_importance)
sorted_features = [features[i] for i in sorted_indices]
sorted_importance = overall_importance[sorted_indices]
sorted_std = overall_std[sorted_indices]

colors = plt.cm.Set2(np.linspace(0, 1, len(features)))
bars = ax2.barh(sorted_features, sorted_importance, xerr=sorted_std, 
                color=colors, capsize=5, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Mean Importance Across Targets (%)', fontsize=11, fontweight='bold')
ax2.set_title('Overall Feature Rankings', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax2.text(val + sorted_std[i] + 2, i, f'{val:.1f}%', 
            va='center', fontsize=10, fontweight='bold')

ax2.set_xlim(0, 60)

# ============================================================================
# Panel 3: Target-specific dominance
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# Find dominant feature for each target
dominant_features = []
dominant_values = []
for i, target in enumerate(targets):
    max_idx = np.argmax(importance_matrix[i])
    dominant_features.append(features[max_idx])
    dominant_values.append(importance_matrix[i, max_idx])

# Create bar chart
colors_targets = ['steelblue', 'coral', 'mediumseagreen', 'gold']
bars = ax3.bar(targets, dominant_values, color=colors_targets, 
               edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Dominant Feature Importance (%)', fontsize=11, fontweight='bold')
ax3.set_title('Most Important Feature per Target', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 100)

# Add feature labels on bars
for i, (bar, feat, val) in enumerate(zip(bars, dominant_features, dominant_values)):
    ax3.text(i, val + 2, feat, ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color='black')
    ax3.text(i, val/2, f'{val:.1f}%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

# ============================================================================
# Panel 4: Model performance
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

x_pos = np.arange(len(targets))
bars = ax4.bar(x_pos, test_r2, color=colors_targets, 
               edgecolor='black', linewidth=1.5)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(targets, fontsize=10)
ax4.set_ylabel('R² Score (Test Set)', fontsize=11, fontweight='bold')
ax4.set_title('Model Prediction Performance', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 1.0)
ax4.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.5, label='R²=0.8 threshold')
ax4.grid(axis='y', alpha=0.3)
ax4.legend()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, test_r2)):
    ax4.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

# ============================================================================
# Panel 5: Key insights text box
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

insights_text = """
KEY FINDINGS:

1. INJECTION PRESSURE dominates overall
   • 50.6% mean importance across targets
   • 79-81% importance for penetration length
   • Critical predictor for spray momentum

2. CHAMBER TEMPERATURE controls Mie angle
   • 71.5% importance for Mie scattering angle
   • Evaporation effects shape liquid phase
   • Low importance for other targets (5-14%)

3. FUEL PROPERTIES affect shadowgraphy
   • Density: 30.3% for shadowgraphy angle
   • Viscosity: 23.4% for shadowgraphy angle
   • Control atomization and spreading

4. CHAMBER PRESSURE surprisingly low
   • Only 6.9% mean importance
   • Expected to dominate angle (theory)
   • May reflect limited dataset variation
"""

ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes,
        fontsize=9.5, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

# Add footer
fig.text(0.5, 0.01, 
         'Analysis: 726 observations, 5 features (Pc_bar, Tc_K, Pinj_bar, rho_kgm3, mu_Pas), 4 targets | ' +
         'Model: Gradient Boosting (300 trees, depth=5) | Method: Permutation Importance (30 repeats)',
         ha='center', fontsize=8, style='italic', color='gray')

plt.savefig('../plots/feature_importance_summary.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/feature_importance_summary.png")

plt.show()
