# Feature Importance Analysis Report
## Gradient Boosting Model - 5 Features (No Time)

**Date:** November 28, 2024  
**Model:** Gradient Boosting Regressor  
**Features:** 5 thermodynamic/injection parameters (Time_ms excluded)  
**Samples:** 726 observations across 6 experimental runs

---

## Executive Summary

This analysis investigates feature importance for predicting spray characteristics using only thermodynamic and fuel properties, **without temporal information**. By excluding Time as a predictor, we reveal which experimental conditions most strongly correlate with observed spray behavior across all stages of spray development.

### Key Findings:

1. **INJECTION PRESSURE (Pinj_bar)** dominates overall importance (37-51% across methods)
   - Strongest predictor for penetration length (54-81% importance)
   - Critical for both shadowgraphy and Mie scattering measurements

2. **CHAMBER TEMPERATURE (Tc_K)** is the second most important feature (22-24%)
   - Dominates Mie scattering cone angle predictions (50-72% importance)
   - Controls evaporation and liquid/vapor phase transitions

3. **FUEL PROPERTIES (rho_kgm3, mu_Pas)** show significant importance for spray angles
   - Fuel density: 10-13% overall, up to 30% for shadowgraphy angle
   - Viscosity: 9-14% overall, up to 25% for shadowgraphy angle

4. **CHAMBER PRESSURE (Pc_bar)** shows lower than expected importance (7-12%)
   - This is surprising given its theoretical role in aerodynamic spreading
   - May indicate correlation with other features or limited variation in dataset

---

## 1. Aggregated Feature Rankings (Mean Across All Targets)

### Built-in Feature Importance (Gini/Impurity-based):
```
Rank 1: Pinj_bar    37.21%  ████████████████████████████████████
Rank 2: Tc_K        23.86%  ███████████████████████
Rank 3: mu_Pas      14.34%  ██████████████
Rank 4: rho_kgm3    12.54%  ████████████
Rank 5: Pc_bar      12.04%  ████████████
```

### Permutation Importance (Test Set):
```
Rank 1: Pinj_bar    50.59%  ██████████████████████████████████████████████████
Rank 2: Tc_K        23.89%  ███████████████████████
Rank 3: rho_kgm3     9.65%  █████████
Rank 4: mu_Pas       8.93%  ████████
Rank 5: Pc_bar       6.95%  ██████
```

### Permutation Importance (Training Set):
```
Rank 1: Pinj_bar    45.42%  █████████████████████████████████████████████
Rank 2: Tc_K        22.31%  ██████████████████████
Rank 3: mu_Pas      11.29%  ███████████
Rank 4: Pc_bar      10.75%  ██████████
Rank 5: rho_kgm3    10.22%  ██████████
```

---

## 2. Target-Specific Feature Importance

### 2.1 Spray Cone Angle (Shadowgraphy)

**Model Performance:**
- Training R²: 0.991
- Test R²: 0.874
- Test RMSE: 0.406°
- Test MAE: 0.273°

**Top 3 Features (Permutation - Test Set):**
```
1. rho_kgm3 (Fuel Density)      30.32% ± 3.01%
2. Pinj_bar (Injection Pressure) 25.23% ± 3.17%
3. mu_Pas (Fuel Viscosity)       23.42% ± 2.63%
```

**Physical Interpretation:**  
Fuel properties dominate shadowgraphy angle predictions. This makes physical sense as:
- Fuel density affects atomization momentum and droplet trajectories
- Viscosity controls breakup mechanisms and spray spreading
- Injection pressure provides the initial atomization energy
- Chamber pressure has surprisingly low importance (7%)

---

### 2.2 Spray Penetration Length (Shadowgraphy)

**Model Performance:**
- Training R²: 0.956
- Test R²: 0.645
- Test RMSE: 28.82 L/D
- Test MAE: 23.49 L/D

**Top 3 Features (Permutation - Test Set):**
```
1. Pinj_bar (Injection Pressure)  80.91% ± 15.73%
2. Pc_bar (Chamber Pressure)       9.17% ± 3.68%
3. Tc_K (Chamber Temperature)      5.09% ± 1.94%
```

**Physical Interpretation:**  
Injection pressure absolutely dominates penetration length (81% importance). This aligns with jet physics:
- Higher injection pressure → higher jet velocity → deeper penetration
- Chamber pressure creates drag but is secondary effect
- Temperature affects evaporation but less critical for liquid shadowgraphy

---

### 2.3 Spray Cone Angle (Mie Scattering)

**Model Performance:**
- Training R²: 0.993
- Test R²: 0.920
- Test RMSE: 0.371°
- Test MAE: 0.260°

**Top 3 Features (Permutation - Test Set):**
```
1. Tc_K (Chamber Temperature)     71.50% ± 10.06%
2. Pinj_bar (Injection Pressure)  17.30% ± 3.19%
3. rho_kgm3 (Fuel Density)         4.67% ± 1.23%
```

**Physical Interpretation:**  
Temperature dominates Mie scattering angle (72% importance). This is expected because:
- Mie scattering measures liquid phase droplets
- High temperature evaporates spray edges, narrowing the visible liquid cone
- This is the strongest temperature effect observed across all targets
- Chamber pressure again shows low importance (~4%)

---

### 2.4 Spray Penetration Length (Mie Scattering)

**Model Performance:**
- Training R²: 0.959
- Test R²: 0.680
- Test RMSE: 26.10 L/D
- Test MAE: 21.26 L/D

**Top 3 Features (Permutation - Test Set):**
```
1. Pinj_bar (Injection Pressure)  78.92% ± 15.40%
2. Pc_bar (Chamber Pressure)       9.51% ± 3.49%
3. Tc_K (Chamber Temperature)      4.88% ± 2.06%
```

**Physical Interpretation:**  
Similar to shadowgraphy, injection pressure dominates (79%). The liquid length measured by Mie scattering is primarily controlled by jet momentum, with temperature and pressure as secondary effects.

---

## 3. Model Performance Summary

| Target | Train R² | Test R² | CV R² Mean | Test RMSE | Test MAE |
|--------|----------|---------|------------|-----------|----------|
| Angle (Shadow) | 0.991 | 0.874 | -0.371 | 0.406° | 0.273° |
| Length (Shadow) | 0.956 | 0.645 | -0.659 | 28.82 L/D | 23.49 L/D |
| Angle (Mie) | 0.993 | 0.920 | -0.114 | 0.371° | 0.260° |
| Length (Mie) | 0.959 | 0.680 | -0.728 | 26.10 L/D | 21.26 L/D |

**Notes:**
- Models achieve excellent training performance (R² > 0.95)
- Test performance is good for angles (R² > 0.87) but moderate for lengths (R² ~ 0.64-0.68)
- Negative cross-validation scores indicate high variance between folds (only 6 experimental conditions)
- The dataset has limited diversity in experimental conditions, affecting generalization

---

## 4. Physical Interpretation & Discussion

### 4.1 Comparison to Physical Expectations

Based on spray physics literature, we expected:

**For Spray Cone Angle:**
- Expected: Pc_bar > Pinj_bar > Tc_K > mu_Pas > rho_kgm3
- Observed (Shadow): rho_kgm3 > Pinj_bar > mu_Pas > Tc_K > Pc_bar
- Observed (Mie): Tc_K >> Pinj_bar > rho_kgm3 > mu_Pas > Pc_bar

**For Penetration Length:**
- Expected: Tc_K > Pc_bar > Pinj_bar > rho_kgm3 > mu_Pas
- Observed (Shadow): Pinj_bar >>> Pc_bar > Tc_K > mu_Pas > rho_kgm3
- Observed (Mie): Pinj_bar >>> Pc_bar > Tc_K > rho_kgm3 > mu_Pas

### 4.2 Key Discrepancies

**Chamber Pressure (Pc_bar) Lower Than Expected:**
- Theory suggests chamber pressure should dominate angle via aerodynamic spreading
- Observed importance is only 6-12% across targets
- Possible explanations:
  1. Limited variation in Pc_bar in dataset (mean=65±10 bar, only 6 unique conditions)
  2. Correlation with chamber temperature (higher P often paired with higher T)
  3. Aerodynamic effects may be captured indirectly through fuel properties
  4. Model is averaging across transient process, diluting steady-state pressure effects

**Injection Pressure (Pinj_bar) Dominates:**
- Injection pressure shows 37-51% importance overall, up to 81% for penetration
- This exceeds theoretical expectations for steady-state conditions
- Explanation: Dataset includes entire transient process where injection momentum is critical
- The model learns that Pinj_bar correlates with spray development rate

**Temperature Effects Isolated to Mie Scattering Angle:**
- Tc_K shows 72% importance for Mie angle but only 5-14% for other targets
- This makes sense: Mie scattering is most sensitive to evaporation
- Temperature controls liquid/vapor boundary which defines Mie-visible cone

### 4.3 Feature Mechanisms

**INJECTION PRESSURE (Pinj_bar):**  
*Physical mechanism:* v_jet ∝ √(ΔP_injection)
- Controls initial jet velocity and momentum
- Directly determines penetration depth
- Affects atomization energy and droplet size
- **Why it dominates:** Dataset captures transient spray development where momentum is king

**CHAMBER TEMPERATURE (Tc_K):**  
*Physical mechanism:* Evaporation rate ∝ exp(-ΔH_vap / R·T_chamber)
- Accelerates fuel evaporation at high temperatures
- Shrinks liquid core and narrows Mie-visible spray
- Less important for shadowgraphy (measures all phases)
- **Why Mie angle responds:** Direct control of liquid/vapor interface

**FUEL DENSITY (rho_kgm3):**  
*Physical mechanism:* Momentum ∝ ρ_fuel · v_jet²
- Affects jet penetration through inertia
- Influences atomization quality and droplet trajectories
- **Why important for shadowgraphy angle:** Controls spray spreading dynamics

**FUEL VISCOSITY (mu_Pas):**  
*Physical mechanism:* Ohnesorge number Oh = √(We/Re) ∝ μ/(ρ·σ·d)^0.5
- Controls breakup timescales and mechanisms
- Lower viscosity → faster atomization → wider spray
- **Why important for shadowgraphy angle:** Dictates spray expansion rate

**CHAMBER PRESSURE (Pc_bar):**  
*Physical mechanism:* ρ_ambient ∝ P_chamber / (R·T_chamber)
- Creates ambient density for aerodynamic drag
- Should widen spray via gas-liquid momentum exchange
- **Why lower than expected:** Limited dataset variation, correlation with temperature

---

## 5. Methodology

### 5.1 Data Preparation
- **Dataset:** 726 observations from 6 experimental runs (121 time points each)
- **Approach:** All temporal observations used, Time_ms excluded as feature
- **Rationale:** Treats each time point as independent observation of condition→spray relationship
- **Train/Test Split:** 80/20 (580 train, 146 test)

### 5.2 Model Configuration
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
```

### 5.3 Feature Importance Methods

1. **Built-in (Gini) Importance:**
   - Measures mean decrease in impurity at splits
   - Fast to compute, but biased toward high-cardinality features
   - Calculated from tree structure

2. **Permutation Importance (Test Set):**
   - Measures drop in R² when feature is randomly shuffled
   - More reliable indicator of true predictive power
   - 30 permutations per feature

3. **Permutation Importance (Train Set):**
   - Same as above but on training data
   - Helps identify overfitting (large train-test discrepancy)

---

## 6. Visualizations Generated

Four comprehensive figures saved to `../plots/`:

1. **feature_importance_comprehensive.png**  
   12-panel grid showing all three importance methods for all four targets

2. **feature_importance_aggregated.png**  
   Mean importance across all targets for each method

3. **model_performance_prediction.png**  
   Actual vs. Predicted scatter plots with R² scores

4. **feature_importance_heatmap.png**  
   Heatmaps showing importance patterns across targets

---

## 7. Conclusions

### 7.1 Answering the Research Question

**"Which features matter most when predicting spray characteristics without time?"**

1. **Injection Pressure** is the dominant predictor (37-51% importance)
   - Absolutely critical for penetration length (79-81%)
   - Important for all targets

2. **Chamber Temperature** is crucial for Mie scattering angle (72%)
   - Controls evaporation and liquid phase visibility
   - Secondary for other targets (5-23%)

3. **Fuel Properties** (density, viscosity) significantly affect shadowgraphy angle
   - Combined importance of 44-54% for shadowgraphy angle
   - Less important for penetration length

4. **Chamber Pressure** shows surprisingly low importance (7-12%)
   - May reflect dataset limitations (only 6 unique conditions)
   - Or indicate that time-averaged predictions don't capture steady-state aerodynamics

### 7.2 Model Quality Assessment

**Strengths:**
- Excellent training performance (R² > 0.95)
- Good test performance for spray angles (R² = 0.87-0.92)
- Consistent feature rankings across different importance methods
- Results align with known physics (except chamber pressure)

**Weaknesses:**
- Moderate performance for penetration length (R² = 0.64-0.68)
- Poor cross-validation scores (negative R²) due to limited experimental diversity
- Only 6 unique experimental conditions limits generalization
- Time-averaging may obscure transient vs. steady-state physics

### 7.3 Recommendations

**For Future Experiments:**
1. Increase diversity in chamber pressure conditions to improve statistical power
2. Decouple Pc and Tc variations to isolate their individual effects
3. Consider separate models for early transient vs. quasi-steady regimes
4. Add more fuel types to strengthen fuel property importance estimates

**For Modeling:**
1. Consider time-windowed models (early/middle/late injection)
2. Test non-linear feature interactions (Pc × Tc, Pinj × rho, etc.)
3. Investigate why chamber pressure importance is lower than expected
4. Validate on independent test dataset with different conditions

**For Physical Understanding:**
1. The dominance of Pinj_bar suggests momentum-driven spray physics
2. Temperature effects are strongest for liquid phase measurements (Mie)
3. Fuel properties matter most for angle predictions (spreading dynamics)
4. Chamber pressure may need longer observation times to show full effect

---

## 8. Data & Code Availability

- **Analysis Script:** `notebooks/feature_importance_analysis.py`
- **Output Log:** `notebooks/feature_importance_output.txt`
- **Plots:** `plots/feature_importance_*.png`
- **Dataset:** `data/processed/preprocessed_dataset.csv`

---

**Report Generated:** November 28, 2024  
**Analysis Tool:** scikit-learn GradientBoostingRegressor  
**Python Version:** 3.11.14
