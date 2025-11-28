"""
Feature Importance Analysis for Gradient Boosting Model - No Time Feature
===========================================================================

This script performs a comprehensive feature importance analysis for predicting
spray characteristics using only 5 thermodynamic/injection features (NO TIME):
- Pc_bar: Chamber Pressure
- Tc_K: Chamber Temperature  
- Pinj_bar: Injection Pressure
- rho_kgm3: Fuel Density
- mu_Pas: Fuel Dynamic Viscosity

Target Variables:
- Spray Cone Angle (Shadowgraphy): angle_shadow_deg
- Spray Penetration Length (Shadowgraphy): len_shadow_L_D
- Spray Cone Angle (Mie Scattering): angle_mie_deg
- Spray Penetration Length (Mie Scattering): len_mie_L_D

Approach: Uses ALL temporal observations (726 samples) but excludes Time_ms as a feature.
This captures the average relationship between thermodynamic conditions and spray 
behavior across the entire transient process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS - 5 FEATURES (NO TIME)")
print("="*80)
print("\nLoading preprocessed dataset...")

# Load data
df = pd.read_csv('../data/processed/preprocessed_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Display basic statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
print(f"Total number of observations: {len(df)}")
print(f"Number of unique runs: {df['run'].nunique()}")
print(f"Time range: {df['Time_ms'].min():.3f} ms to {df['Time_ms'].max():.3f} ms")
print(f"\nTime value counts:")
print(df['Time_ms'].value_counts().sort_index().head(10))

# ============================================================================
# 2. PREPARE DATA (Use all time points, exclude Time as feature)
# ============================================================================

print("\n" + "="*80)
print("DATA PREPARATION STRATEGY")
print("="*80)

# Use ALL data points, treating each observation independently
# This captures the relationship between thermodynamic conditions and spray 
# characteristics across the entire transient process
df_analysis = df.copy()

print(f"\nApproach: Use all temporal observations without Time as a feature")
print(f"  - This treats each time point as an independent observation")
print(f"  - Model learns average relationship between conditions and spray behavior")
print(f"  - Captures both transient and quasi-steady effects")

print(f"\nDataset: {len(df_analysis)} observations")
print(f"Number of unique experimental runs: {df_analysis['run'].nunique()}")
print(f"Time points per run: {len(df_analysis) / df_analysis['run'].nunique():.0f}")

# Display time distribution
print(f"\nTime range: {df_analysis['Time_ms'].min():.3f} ms to {df_analysis['Time_ms'].max():.3f} ms")
print(f"Number of unique time points: {df_analysis['Time_ms'].nunique()}")

# ============================================================================
# 3. DEFINE FEATURES AND TARGETS
# ============================================================================

# Input features (5 features - NO TIME)
feature_names = ['Pc_bar', 'Tc_K', 'Pinj_bar', 'rho_kgm3', 'mu_Pas']

# Target variables (4 targets)
target_names = {
    'angle_shadow_deg': 'Spray Cone Angle (Shadowgraphy)',
    'len_shadow_L_D': 'Spray Penetration Length (Shadowgraphy)',
    'angle_mie_deg': 'Spray Cone Angle (Mie Scattering)',
    'len_mie_L_D': 'Spray Penetration Length (Mie Scattering)'
}

print("\n" + "="*80)
print("FEATURE AND TARGET DEFINITIONS")
print("="*80)
print(f"\nInput Features ({len(feature_names)}):")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i}. {feat}")

print(f"\nTarget Variables ({len(target_names)}):")
for i, (key, val) in enumerate(target_names.items(), 1):
    print(f"  {i}. {key}: {val}")

# Extract features and check for missing values
X = df_analysis[feature_names].copy()
print(f"\n\nFeature matrix shape: {X.shape}")
print(f"Missing values in features: {X.isnull().sum().sum()}")

# Display feature statistics
print("\n" + "="*80)
print("FEATURE STATISTICS (ALL TIME POINTS)")
print("="*80)
print(X.describe())

# ============================================================================
# 4. GRADIENT BOOSTING MODEL CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("GRADIENT BOOSTING MODEL CONFIGURATION")
print("="*80)

# Optimized hyperparameters for feature importance analysis
gb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'subsample': 0.8,
    'max_features': 'sqrt',
    'random_state': 42,
    'verbose': 0
}

print("\nHyperparameters:")
for param, value in gb_params.items():
    print(f"  {param}: {value}")

# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS FOR EACH TARGET
# ============================================================================

results = {}

for target_col, target_desc in target_names.items():
    print("\n" + "="*80)
    print(f"ANALYZING TARGET: {target_desc}")
    print(f"Variable: {target_col}")
    print("="*80)
    
    # Prepare target variable
    y = df_analysis[target_col].copy()
    
    # Check for missing values
    valid_mask = ~y.isnull()
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    print(f"\nTarget statistics:")
    print(f"  Valid samples: {len(y_valid)}")
    print(f"  Mean: {y_valid.mean():.3f}")
    print(f"  Std: {y_valid.std():.3f}")
    print(f"  Min: {y_valid.min():.3f}")
    print(f"  Max: {y_valid.max():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Train model
    print(f"\nTraining Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = gb_model.predict(X_train)
    y_pred_test = gb_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n{'Metric':<20} {'Training':<15} {'Testing'}")
    print(f"{'-'*50}")
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:.4f}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:.4f}")
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:.4f}")
    
    # Cross-validation
    print(f"\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(
        gb_model, X_valid, y_valid, cv=5, 
        scoring='r2', n_jobs=-1
    )
    print(f"  CV R² Scores: {cv_scores}")
    print(f"  CV R² Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ========================================================================
    # FEATURE IMPORTANCE METHODS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Method 1: Built-in feature importance (Gini importance)
    print(f"\n1. Built-in Feature Importance (Mean Decrease in Impurity):")
    builtin_importance = gb_model.feature_importances_
    builtin_importance_pct = 100 * builtin_importance / builtin_importance.sum()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': builtin_importance,
        'Importance_%': builtin_importance_pct
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{importance_df.to_string(index=False)}")
    
    # Method 2: Permutation importance on test set
    print(f"\n2. Permutation Importance (Test Set):")
    perm_importance = permutation_importance(
        gb_model, X_test, y_test, n_repeats=30, 
        random_state=42, n_jobs=-1
    )
    
    perm_importance_mean = perm_importance.importances_mean
    perm_importance_std = perm_importance.importances_std
    perm_importance_pct = 100 * perm_importance_mean / perm_importance_mean.sum()
    
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance_mean,
        'Std': perm_importance_std,
        'Importance_%': perm_importance_pct
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{perm_df.to_string(index=False)}")
    
    # Method 3: Permutation importance on training set
    print(f"\n3. Permutation Importance (Training Set):")
    perm_importance_train = permutation_importance(
        gb_model, X_train, y_train, n_repeats=30, 
        random_state=42, n_jobs=-1
    )
    
    perm_train_mean = perm_importance_train.importances_mean
    perm_train_std = perm_importance_train.importances_std
    perm_train_pct = 100 * perm_train_mean / perm_train_mean.sum()
    
    perm_train_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_train_mean,
        'Std': perm_train_std,
        'Importance_%': perm_train_pct
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{perm_train_df.to_string(index=False)}")
    
    # Store results
    results[target_col] = {
        'model': gb_model,
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        },
        'builtin_importance': importance_df,
        'perm_importance_test': perm_df,
        'perm_importance_train': perm_train_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }

# ============================================================================
# 6. COMPREHENSIVE VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

# Create output directory
import os
os.makedirs('../plots', exist_ok=True)

# Figure 1: Feature Importance Comparison (All Methods, All Targets)
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
fig.suptitle('Feature Importance Analysis - Gradient Boosting (5 Features, No Time)', 
             fontsize=16, fontweight='bold')

for idx, (target_col, target_desc) in enumerate(target_names.items()):
    res = results[target_col]
    
    # Plot 1: Built-in importance
    ax1 = axes[idx, 0]
    builtin_df = res['builtin_importance'].sort_values('Importance_%', ascending=True)
    ax1.barh(builtin_df['Feature'], builtin_df['Importance_%'], color='steelblue')
    ax1.set_xlabel('Importance (%)', fontsize=10)
    ax1.set_title(f'{target_desc}\nBuiltin Importance', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Permutation importance (test)
    ax2 = axes[idx, 1]
    perm_test_df = res['perm_importance_test'].sort_values('Importance_%', ascending=True)
    ax2.barh(perm_test_df['Feature'], perm_test_df['Importance_%'], 
             xerr=perm_test_df['Std']*100, color='coral', capsize=5)
    ax2.set_xlabel('Importance (%)', fontsize=10)
    ax2.set_title(f'{target_desc}\nPermutation Imp. (Test)', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Permutation importance (train)
    ax3 = axes[idx, 2]
    perm_train_df = res['perm_importance_train'].sort_values('Importance_%', ascending=True)
    ax3.barh(perm_train_df['Feature'], perm_train_df['Importance_%'], 
             xerr=perm_train_df['Std']*100, color='mediumseagreen', capsize=5)
    ax3.set_xlabel('Importance (%)', fontsize=10)
    ax3.set_title(f'{target_desc}\nPermutation Imp. (Train)', fontsize=10, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
print("\nSaved: ../plots/feature_importance_comprehensive.png")

# Figure 2: Aggregated Feature Importance Across All Targets
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Aggregated Feature Importance Across All Targets', 
             fontsize=14, fontweight='bold')

# Aggregate built-in importance
builtin_agg = pd.DataFrame()
for target_col in target_names.keys():
    df_temp = results[target_col]['builtin_importance'][['Feature', 'Importance_%']].copy()
    df_temp.columns = ['Feature', target_col]
    if builtin_agg.empty:
        builtin_agg = df_temp
    else:
        builtin_agg = builtin_agg.merge(df_temp, on='Feature')

builtin_agg['Mean_%'] = builtin_agg[list(target_names.keys())].mean(axis=1)
builtin_agg = builtin_agg.sort_values('Mean_%', ascending=True)

ax1 = axes[0]
ax1.barh(builtin_agg['Feature'], builtin_agg['Mean_%'], color='steelblue')
ax1.set_xlabel('Mean Importance (%)', fontsize=11)
ax1.set_title('Built-in Importance\n(Mean Across Targets)', fontsize=11, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Aggregate permutation importance (test)
perm_test_agg = pd.DataFrame()
for target_col in target_names.keys():
    df_temp = results[target_col]['perm_importance_test'][['Feature', 'Importance_%']].copy()
    df_temp.columns = ['Feature', target_col]
    if perm_test_agg.empty:
        perm_test_agg = df_temp
    else:
        perm_test_agg = perm_test_agg.merge(df_temp, on='Feature')

perm_test_agg['Mean_%'] = perm_test_agg[list(target_names.keys())].mean(axis=1)
perm_test_agg = perm_test_agg.sort_values('Mean_%', ascending=True)

ax2 = axes[1]
ax2.barh(perm_test_agg['Feature'], perm_test_agg['Mean_%'], color='coral')
ax2.set_xlabel('Mean Importance (%)', fontsize=11)
ax2.set_title('Permutation Importance (Test)\n(Mean Across Targets)', fontsize=11, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Aggregate permutation importance (train)
perm_train_agg = pd.DataFrame()
for target_col in target_names.keys():
    df_temp = results[target_col]['perm_importance_train'][['Feature', 'Importance_%']].copy()
    df_temp.columns = ['Feature', target_col]
    if perm_train_agg.empty:
        perm_train_agg = df_temp
    else:
        perm_train_agg = perm_train_agg.merge(df_temp, on='Feature')

perm_train_agg['Mean_%'] = perm_train_agg[list(target_names.keys())].mean(axis=1)
perm_train_agg = perm_train_agg.sort_values('Mean_%', ascending=True)

ax3 = axes[2]
ax3.barh(perm_train_agg['Feature'], perm_train_agg['Mean_%'], color='mediumseagreen')
ax3.set_xlabel('Mean Importance (%)', fontsize=11)
ax3.set_title('Permutation Importance (Train)\n(Mean Across Targets)', fontsize=11, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/feature_importance_aggregated.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/feature_importance_aggregated.png")

# Figure 3: Model Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Model Performance Metrics - Gradient Boosting (5 Features, No Time)', 
             fontsize=14, fontweight='bold')

for idx, (target_col, target_desc) in enumerate(target_names.items()):
    ax = axes.flat[idx]
    res = results[target_col]
    
    # Actual vs Predicted
    y_test = res['y_test']
    y_pred = res['y_pred_test']
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{target_desc}\nR² = {res["metrics"]["test_r2"]:.4f}, RMSE = {res["metrics"]["test_rmse"]:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/model_performance_prediction.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/model_performance_prediction.png")

# Figure 4: Heatmap of Feature Importance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Importance Heatmap Across Targets', fontsize=14, fontweight='bold')

# Built-in importance heatmap
builtin_matrix = pd.DataFrame()
for target_col, target_desc in target_names.items():
    builtin_matrix[target_desc] = results[target_col]['builtin_importance'].set_index('Feature')['Importance_%']

ax1 = axes[0]
sns.heatmap(builtin_matrix, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Importance (%)'}, ax=ax1)
ax1.set_title('Built-in Importance', fontsize=11, fontweight='bold')
ax1.set_ylabel('Features', fontsize=10)

# Permutation importance (test) heatmap
perm_test_matrix = pd.DataFrame()
for target_col, target_desc in target_names.items():
    perm_test_matrix[target_desc] = results[target_col]['perm_importance_test'].set_index('Feature')['Importance_%']

ax2 = axes[1]
sns.heatmap(perm_test_matrix, annot=True, fmt='.1f', cmap='Oranges', cbar_kws={'label': 'Importance (%)'}, ax=ax2)
ax2.set_title('Permutation Importance (Test)', fontsize=11, fontweight='bold')
ax2.set_ylabel('')

# Permutation importance (train) heatmap
perm_train_matrix = pd.DataFrame()
for target_col, target_desc in target_names.items():
    perm_train_matrix[target_desc] = results[target_col]['perm_importance_train'].set_index('Feature')['Importance_%']

ax3 = axes[2]
sns.heatmap(perm_train_matrix, annot=True, fmt='.1f', cmap='Greens', cbar_kws={'label': 'Importance (%)'}, ax=ax3)
ax3.set_title('Permutation Importance (Train)', fontsize=11, fontweight='bold')
ax3.set_ylabel('')

plt.tight_layout()
plt.savefig('../plots/feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/feature_importance_heatmap.png")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY REPORT: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\n1. AGGREGATED FEATURE RANKINGS (Mean Across All Targets)")
print("-" * 80)

print("\nBuilt-in Feature Importance:")
for i, row in builtin_agg.sort_values('Mean_%', ascending=False).iterrows():
    print(f"  {row['Feature']:<15} {row['Mean_%']:>6.2f}%")

print("\nPermutation Importance (Test Set):")
for i, row in perm_test_agg.sort_values('Mean_%', ascending=False).iterrows():
    print(f"  {row['Feature']:<15} {row['Mean_%']:>6.2f}%")

print("\nPermutation Importance (Training Set):")
for i, row in perm_train_agg.sort_values('Mean_%', ascending=False).iterrows():
    print(f"  {row['Feature']:<15} {row['Mean_%']:>6.2f}%")

print("\n2. MODEL PERFORMANCE SUMMARY")
print("-" * 80)

performance_summary = pd.DataFrame()
for target_col, target_desc in target_names.items():
    metrics = results[target_col]['metrics']
    performance_summary = pd.concat([performance_summary, pd.DataFrame({
        'Target': [target_desc],
        'Train_R2': [metrics['train_r2']],
        'Test_R2': [metrics['test_r2']],
        'CV_R2_Mean': [metrics['cv_r2_mean']],
        'Test_RMSE': [metrics['test_rmse']],
        'Test_MAE': [metrics['test_mae']]
    })])

performance_summary = performance_summary.reset_index(drop=True)
print("\n" + performance_summary.to_string(index=False))

print("\n3. TARGET-SPECIFIC FEATURE RANKINGS")
print("-" * 80)

for target_col, target_desc in target_names.items():
    print(f"\n{target_desc}:")
    print(f"  Top 3 Features (Built-in):")
    top_builtin = results[target_col]['builtin_importance'].head(3)
    for i, row in top_builtin.iterrows():
        print(f"    {row['Feature']:<15} {row['Importance_%']:>6.2f}%")
    
    print(f"  Top 3 Features (Permutation - Test):")
    top_perm = results[target_col]['perm_importance_test'].head(3)
    for i, row in top_perm.iterrows():
        print(f"    {row['Feature']:<15} {row['Importance_%']:>6.2f}% (±{row['Std']:.4f})")

# ============================================================================
# 8. PHYSICAL INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("PHYSICAL INTERPRETATION")
print("="*80)

print("""
Based on the feature importance analysis without time as a predictor, we can 
interpret the results in the context of spray physics:

KEY FINDING: Without temporal information, the model must rely solely on 
thermodynamic and fuel properties to predict spray characteristics. This reveals
which conditions most strongly correlate with observed spray behavior across all
stages of spray development.

CHAMBER PRESSURE (Pc_bar):
- Expected to dominate spray cone angle predictions (aerodynamic spreading)
- Higher ambient density forces spray to widen via aerodynamic interactions
- Creates drag that affects penetration length
- Physical mechanism: rho_ambient is proportional to P_chamber / (R*T_chamber)

CHAMBER TEMPERATURE (Tc_K):
- Expected to dominate liquid penetration length (evaporation effects)
- Higher temperature accelerates fuel evaporation at liquid/vapor boundary
- Affects spray angle by altering evaporation at spray edges
- Physical mechanism: Evaporation rate proportional to exp(-deltaH_vap / R*T_chamber)

INJECTION PRESSURE (Pinj_bar):
- Affects initial atomization energy and droplet momentum
- Higher injection pressure leads to finer atomization and wider spray angle
- Influences penetration through initial jet velocity
- Physical mechanism: v_jet proportional to sqrt(deltaP_injection)

FUEL DENSITY (rho_kgm3):
- Influences momentum flux and penetration via inertia
- Affects atomization quality and droplet size distribution
- Physical mechanism: Momentum proportional to rho_fuel * v_jet^2

FUEL VISCOSITY (mu_Pas):
- Controls liquid breakup mechanisms and droplet formation
- Lower viscosity promotes faster atomization and wider spray
- Affects liquid length through breakup timescales
- Physical mechanism: Ohnesorge number Oh = sqrt(We/Re) proportional to mu/(rho*sigma*d)^0.5

EXPECTED RANKINGS:
For Spray Cone Angle: Pc_bar > Pinj_bar > Tc_K > mu_Pas > rho_kgm3
For Penetration Length: Tc_K > Pc_bar > Pinj_bar > rho_kgm3 > mu_Pas

The model's feature importance rankings reveal which parameters have the strongest
statistical association with spray characteristics when temporal evolution is not
explicitly modeled.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nAll visualizations saved to: ../plots/")
print("  - feature_importance_comprehensive.png")
print("  - feature_importance_aggregated.png")
print("  - model_performance_prediction.png")
print("  - feature_importance_heatmap.png")
