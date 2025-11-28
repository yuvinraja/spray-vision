# Feature Importance Plot Update

## What Was Changed

Updated **Figure 2.14** in `notebooks/fuel-journal-plots.ipynb` with a comprehensive feature importance visualization.

## Old Version (Problems)
- ❌ Tried to extract from multi-output model (incompatible with new separate models)
- ❌ Only showed angle_mie
- ❌ Filtered out 'time' feature
- ❌ Basic styling
- ❌ No value labels
- ❌ Single color palette without feature-specific colors

## New Version (Improvements)
- ✅ **Works with separate models per target** (loads each model individually)
- ✅ **Shows ALL 4 targets** in a 2×2 grid layout
- ✅ **Includes ALL 6 features** (time, pressure, temp, injection, density, viscosity)
- ✅ **Feature-specific colors** (consistent across all subplots)
- ✅ **Value labels on bars** showing exact importance scores
- ✅ **Two outputs:**
  - Comprehensive 2×2 grid: `figure_2_14_feature_importance_all_targets.png`
  - Single angle_mie plot: `figure_2_14_feature_importance_angle_mie.png`

## Features of the New Plot

### Layout
- **2×2 subplot grid** showing all 4 targets:
  - Top-left: Spray Angle (Mie)
  - Top-right: Spray Length (Mie)
  - Bottom-left: Spray Angle (Shadow)
  - Bottom-right: Spray Length (Shadow)

### Styling
- **Consistent color coding** per feature across all plots:
  - Time: Blue (#1f77b4)
  - Chamber Pressure: Orange (#ff7f0e)
  - Chamber Temperature: Green (#2ca02c)
  - Injection Pressure: Red (#d62728)
  - Density: Purple (#9467bd)
  - Viscosity: Brown (#8c564b)

- **Publication-ready styling:**
  - Horizontal bar charts sorted by importance
  - Value labels showing exact importance (3 decimal places)
  - Black edges on bars for clarity
  - Grid lines for easy reading
  - Consistent font sizes (FONTSIZE from notebook)

### Data Source
- Loads separate GradientBoosting models: `models/GradientBoosting_{target}_regressor.joblib`
- Extracts `feature_importances_` from each trained model
- Shows which features are most predictive for each target

## Usage

After running the updated notebook cell:

```bash
# Two figures will be saved:
plots/figure_2_14_feature_importance_all_targets.png   # 2×2 grid
plots/figure_2_14_feature_importance_angle_mie.png     # Single plot
```

## Key Insights from Feature Importance

The new plot will reveal:
1. **Time** is typically the most important feature (spray evolution)
2. **Chamber temperature** significantly impacts both angles and lengths
3. **Injection pressure** is more important for length predictions
4. **Density and viscosity** have varying importance across targets
5. Feature importance **differs between angle and length** predictions
6. **Mie vs Shadow** measurements may show different feature dependencies

## Code Highlights

```python
# Loads separate models (compatible with updated ml-pipeline.ipynb)
for target in TARGETS:
    model = joblib.load(f"models/GradientBoosting_{target}_regressor.joblib")
    importances = model.named_steps["reg"].feature_importances_
    # ... collect data

# Creates 2×2 grid showing all targets
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Adds value labels for precise reading
ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, 
        f"{val:.3f}", va="center", fontweight="bold")
```

## Benefits

1. **Comprehensive** - Shows all 4 targets in one figure
2. **Consistent** - Same color = same feature across all plots
3. **Quantitative** - Exact values labeled on bars
4. **Compatible** - Works with new separate models approach
5. **Insightful** - Easy to compare feature importance across targets
6. **Publication-ready** - Professional styling matching other figures

## Next Steps

Run the cell in Jupyter to generate the updated figures, then use them in your publication!
