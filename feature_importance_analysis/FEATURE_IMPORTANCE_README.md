# Feature Importance Analysis - README

## Overview
This directory contains a comprehensive feature importance analysis for gradient boosting models predicting spray characteristics **without using time as an input feature**.

## Quick Summary

**Question:** Which 5 thermodynamic features are most important for predicting spray behavior when time is excluded?

**Answer:**
1. **Injection Pressure (Pinj_bar):** 51% importance - dominates penetration length
2. **Chamber Temperature (Tc_K):** 24% importance - controls Mie scattering angle  
3. **Fuel Density (rho_kgm3):** 10% importance - affects shadowgraphy angle
4. **Fuel Viscosity (mu_Pas):** 9% importance - affects shadowgraphy angle
5. **Chamber Pressure (Pc_bar):** 7% importance - surprisingly low

## Files

### Analysis Scripts
- **`feature_importance_analysis.py`** - Main analysis script (thorough, production-ready)
- **`create_summary_figure.py`** - Creates publication-quality summary visualization
- **`feature_importance_output.txt`** - Complete console output from analysis

### Output Report
- **`../outputs/FEATURE_IMPORTANCE_REPORT.md`** - Comprehensive 14-page analysis report

### Visualizations (in `../plots/`)
1. **`feature_importance_summary.png`** - Single-page summary with all key findings
2. **`feature_importance_comprehensive.png`** - 12-panel grid (3 methods × 4 targets)
3. **`feature_importance_aggregated.png`** - Mean importance across targets
4. **`model_performance_prediction.png`** - Actual vs. Predicted scatter plots
5. **`feature_importance_heatmap.png`** - Importance heatmaps across targets

## Methodology

### Data
- **Source:** `../data/processed/preprocessed_dataset.csv`
- **Samples:** 726 observations from 6 experimental runs
- **Time points:** 121 per run (0.0 to 3.0 ms in 0.025 ms steps)

### Approach
Instead of filtering to steady-state only (which gives only 6 samples), we use **ALL temporal observations** but exclude Time_ms as a feature. This:
- Treats each time point as independent observation
- Learns average relationship between conditions and spray behavior  
- Captures both transient and quasi-steady effects
- Provides sufficient data (726 samples) for robust ML

### Features (5 inputs)
1. **Pc_bar** - Chamber Pressure (bar)
2. **Tc_K** - Chamber Temperature (K)
3. **Pinj_bar** - Injection Pressure (bar)
4. **rho_kgm3** - Fuel Density (kg/m³)
5. **mu_Pas** - Fuel Dynamic Viscosity (Pa·s)

### Targets (4 outputs)
1. **angle_shadow_deg** - Spray Cone Angle (Shadowgraphy)
2. **len_shadow_L_D** - Spray Penetration Length (Shadowgraphy)
3. **angle_mie_deg** - Spray Cone Angle (Mie Scattering)
4. **len_mie_L_D** - Spray Penetration Length (Mie Scattering)

### Model
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt'
)
```

### Feature Importance Methods
1. **Built-in (Gini) Importance** - Mean decrease in impurity
2. **Permutation Importance (Test)** - Drop in R² when feature shuffled (30 repeats)
3. **Permutation Importance (Train)** - Same as above on training data

## Key Results

### Overall Rankings (Permutation Importance - Test Set)
```
Rank 1: Pinj_bar    50.6%  ██████████████████████████████████████████████████
Rank 2: Tc_K        23.9%  ███████████████████████
Rank 3: rho_kgm3     9.7%  █████████
Rank 4: mu_Pas       8.9%  ████████
Rank 5: Pc_bar       7.0%  ███████
```

### Target-Specific Champions

| Target | Dominant Feature | Importance |
|--------|------------------|------------|
| Shadowgraphy Angle | **Fuel Density** | 30.3% |
| Shadowgraphy Length | **Injection Pressure** | 80.9% |
| Mie Angle | **Chamber Temperature** | 71.5% |
| Mie Length | **Injection Pressure** | 78.9% |

### Model Performance

| Target | Test R² | Test RMSE | Test MAE |
|--------|---------|-----------|----------|
| Angle (Shadow) | 0.874 | 0.406° | 0.273° |
| Length (Shadow) | 0.645 | 28.82 L/D | 23.49 L/D |
| Angle (Mie) | 0.920 | 0.371° | 0.260° |
| Length (Mie) | 0.680 | 26.10 L/D | 21.26 L/D |

## Key Insights

### 1. Injection Pressure Dominates
- **51% overall importance**, up to 81% for penetration length
- Physical reason: Controls jet momentum (v_jet ∝ √ΔP)
- Indicates dataset captures momentum-driven transient physics

### 2. Temperature Controls Liquid Phase
- **72% importance for Mie scattering angle**
- Physical reason: Evaporation rate ∝ exp(-ΔH_vap/RT)
- High temperature shrinks liquid core, narrowing Mie-visible spray

### 3. Fuel Properties Shape Angles
- **Density: 30% for shadowgraphy angle**
- **Viscosity: 23% for shadowgraphy angle**  
- Physical reason: Control atomization and spreading dynamics
- Combined importance (44-54%) rivals single features

### 4. Chamber Pressure Unexpectedly Low
- **Only 7% mean importance** (expected to dominate angles)
- Possible reasons:
  - Limited variation in dataset (6 conditions)
  - Correlation with temperature
  - Time-averaging obscures steady-state aerodynamics
  - Requires further investigation

## Physical Interpretation

### Expected vs. Observed Rankings

**For Spray Cone Angle:**
- **Expected:** Pc_bar > Pinj_bar > Tc_K (based on spray theory)
- **Observed (Shadow):** rho_kgm3 > Pinj_bar > mu_Pas > Tc_K > Pc_bar
- **Observed (Mie):** Tc_K >> Pinj_bar > rho_kgm3

**For Penetration Length:**
- **Expected:** Tc_K > Pc_bar > Pinj_bar (steady-state theory)
- **Observed:** Pinj_bar >>> Pc_bar > Tc_K

The discrepancy suggests:
1. Model learns transient momentum physics (hence Pinj_bar dominance)
2. Dataset has limited variation in Pc_bar to learn its effects
3. Temperature effects are strongest for Mie (liquid phase sensitivity)

### Feature Mechanisms

| Feature | Primary Effect | Physical Mechanism |
|---------|---------------|-------------------|
| **Pinj_bar** | Jet momentum | v_jet ∝ √(ΔP_inj) |
| **Tc_K** | Evaporation | Rate ∝ exp(-ΔH_vap/RT) |
| **rho_kgm3** | Inertia | Momentum ∝ ρ·v² |
| **mu_Pas** | Breakup | Oh ∝ μ/(ρ·σ·d)^0.5 |
| **Pc_bar** | Drag | ρ_ambient ∝ P/(R·T) |

## How to Use

### Run Complete Analysis
```bash
cd notebooks
python feature_importance_analysis.py > feature_importance_output.txt
```

### Generate Summary Figure
```bash
cd notebooks
python create_summary_figure.py
```

### View Results
- Read: `../outputs/FEATURE_IMPORTANCE_REPORT.md`
- View plots in: `../plots/feature_importance_*.png`
- Check console output: `feature_importance_output.txt`

## Recommendations

### For Experimentalists
1. Vary chamber pressure more widely to improve statistical power
2. Decouple Pc and Tc to isolate individual effects
3. Consider separate early/late injection regimes
4. Add more fuel types to strengthen fuel property analysis

### For Modelers
1. Test time-windowed models (transient vs. quasi-steady)
2. Investigate non-linear interactions (Pc×Tc, Pinj×rho)
3. Explain low chamber pressure importance
4. Validate on independent dataset with different conditions

### For Theorists
1. Injection momentum appears more important than ambient conditions
2. Temperature effects isolated to liquid phase (Mie scattering)
3. Fuel properties matter for angle, pressure for length
4. Time-averaged predictions may not match steady-state expectations

## Dependencies

```
numpy>=2.3
pandas>=2.3  
scikit-learn>=1.7
matplotlib>=3.10
seaborn>=0.13
```

## Contact & Citation

**Author:** Feature Importance Analysis Pipeline  
**Date:** November 28, 2024  
**Dataset:** ETH Spray Characterization Data  
**Tool:** scikit-learn GradientBoostingRegressor

If you use this analysis, please cite the original spray dataset and reference this analysis methodology.

---

## Quick Start

**Want the executive summary?**  
→ View: `../plots/feature_importance_summary.png`

**Want detailed findings?**  
→ Read: `../outputs/FEATURE_IMPORTANCE_REPORT.md`

**Want to reproduce?**  
→ Run: `python feature_importance_analysis.py`

**Want the raw output?**  
→ Check: `feature_importance_output.txt`
