# Spray Vision: User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Usage Scenarios](#usage-scenarios)
4. [Step-by-Step Tutorials](#step-by-step-tutorials)
5. [Common Tasks](#common-tasks)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Quick Start

Get started with Spray Vision in 5 minutes:

### Prerequisites
- Python 3.10 or 3.11
- 4 GB RAM minimum
- ~500 MB disk space

### Installation
```bash
# Clone repository
git clone <repository-url>
cd spray-vision

# Install dependencies
pip install -r requirements.txt

# Or use make
make requirements
```

### Run Complete Pipeline
```bash
# Automated execution (all steps)
make train-all
```

This will:
1. ✅ Process raw data (30 seconds)
2. ✅ Train 6 ML models (10 minutes)
3. ✅ Train neural network (5 minutes)
4. ✅ Save all trained models
5. ✅ Generate performance metrics

### Make a Prediction
```python
import joblib
import numpy as np

# Load best model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Prepare input: [time, pressure, temp, inj_pressure, density, viscosity]
X = np.array([[1.0, 55.0, 650.0, 800.0, 780.0, 0.002]])

# Predict spray characteristics
pred = model.predict(X)[0]

print(f"Spray angle (shadow): {pred[0]:.2f} degrees")
print(f"Penetration (shadow): {pred[1]:.2f} L/D")
print(f"Spray angle (Mie): {pred[2]:.2f} degrees")
print(f"Penetration (Mie): {pred[3]:.2f} L/D")
```

**Done!** You now have trained models and can make predictions.

---

## Installation

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd spray-vision

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import sklearn, tensorflow, pandas; print('✓ All packages installed')"
```

### Method 2: Development Installation

```bash
# Install with development tools and optional features
make dev-requirements

# This installs:
# - Core packages
# - Jupyter notebooks
# - Ruff linter
# - Visualization tools
# - Extra ML libraries
```

### Method 3: Manual Package Installation

```bash
# Core packages only
pip install numpy pandas scikit-learn tensorflow keras joblib openpyxl

# Visualization (optional)
pip install matplotlib seaborn

# Jupyter (optional, for interactive development)
pip install jupyter notebook ipykernel
```

### Verify Installation

```bash
# Run project status check
make report

# Or check manually
python -c "
import numpy, pandas, sklearn, tensorflow
print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('Scikit-learn:', sklearn.__version__)
print('TensorFlow:', tensorflow.__version__)
print('✓ Installation successful')
"
```

### System Requirements

**Minimum**:
- Python 3.10+
- 4 GB RAM
- 500 MB disk space
- Modern CPU (2+ cores)

**Recommended**:
- Python 3.10
- 8 GB RAM
- 1 GB disk space
- Multi-core CPU
- GPU (optional, for ANN training speedup)

---

## Usage Scenarios

### Scenario 1: Research and Experimentation

**Goal**: Understand spray behavior, compare different ML approaches

**Steps**:
1. Run data preprocessing to understand data structure
2. Perform exploratory data analysis
3. Train multiple models
4. Compare performance
5. Generate publication-quality plots

```bash
# Interactive exploration
jupyter notebook notebooks/exploratory-data-analysis.ipynb

# Train all models
make train-all

# Generate plots
jupyter notebook notebooks/fuel-journal-plots.ipynb
```

**Outputs**:
- Statistical insights
- Model performance comparison
- High-quality visualizations
- Research findings

---

### Scenario 2: Production Deployment

**Goal**: Deploy trained model for real-time predictions

**Steps**:
1. Identify best model (Gradient Boosting recommended)
2. Load model and create prediction API
3. Deploy to server
4. Monitor performance

**Example API**:
```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/GradientBoosting_regressor.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array([[
        data['time'], data['chamber_pressure'], 
        data['chamber_temperature'], data['injection_pressure'],
        data['density'], data['viscosity']
    ]])
    pred = model.predict(X)[0]
    return jsonify({
        'angle_shadow': float(pred[0]),
        'penetration_shadow': float(pred[1]),
        'angle_mie': float(pred[2]),
        'penetration_mie': float(pred[3]),
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run**:
```bash
pip install flask
python app.py
```

**Test**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time": 1.0,
    "chamber_pressure": 55.0,
    "chamber_temperature": 650.0,
    "injection_pressure": 800.0,
    "density": 780.0,
    "viscosity": 0.002
  }'
```

---

### Scenario 3: Engine Design Optimization

**Goal**: Optimize fuel injection parameters for desired spray characteristics

**Steps**:
1. Load trained model
2. Define optimization objectives
3. Search parameter space
4. Validate results

**Example**:
```python
import joblib
import numpy as np
from scipy.optimize import minimize

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Target: Maximize penetration while maintaining angle < 15 degrees
def objective(params):
    # params = [injection_pressure, density, viscosity]
    # Fixed: time=2.0, chamber_pressure=55.0, chamber_temp=650.0
    X = np.array([[2.0, 55.0, 650.0, params[0], params[1], params[2]]])
    pred = model.predict(X)[0]
    
    angle_shadow = pred[0]
    pen_shadow = pred[1]
    
    # Objective: maximize penetration (minimize negative)
    objective_val = -pen_shadow
    
    # Penalty if angle exceeds 15 degrees
    if angle_shadow > 15:
        objective_val += 100 * (angle_shadow - 15)**2
    
    return objective_val

# Bounds: injection_pressure (200-1500), density (700-850), viscosity (0.001-0.005)
bounds = [(200, 1500), (700, 850), (0.001, 0.005)]

# Initial guess
x0 = [800, 780, 0.002]

# Optimize
result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

# Best parameters
best_inj_press = result.x[0]
best_density = result.x[1]
best_viscosity = result.x[2]

# Predict with optimal parameters
X_opt = np.array([[2.0, 55.0, 650.0, best_inj_press, best_density, best_viscosity]])
pred_opt = model.predict(X_opt)[0]

print(f"Optimal Parameters:")
print(f"  Injection Pressure: {best_inj_press:.1f} bar")
print(f"  Density: {best_density:.1f} kg/m³")
print(f"  Viscosity: {best_viscosity:.5f} Pa·s")
print(f"\nPredicted Spray:")
print(f"  Angle: {pred_opt[0]:.2f} degrees")
print(f"  Penetration: {pred_opt[1]:.2f} L/D")
```

---

### Scenario 4: Batch Analysis

**Goal**: Analyze spray characteristics for multiple operating conditions

**Steps**:
1. Prepare batch input data
2. Load model
3. Make batch predictions
4. Export results

**Example**:
```python
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Create batch of operating conditions
conditions = pd.DataFrame({
    'time': np.linspace(0, 5, 50),  # 50 time points
    'chamber_pressure': [55.0] * 50,
    'chamber_temperature': [650.0] * 50,
    'injection_pressure': [800.0] * 50,
    'density': [780.0] * 50,
    'viscosity': [0.002] * 50
})

# Prepare input array
X = conditions[['time', 'chamber_pressure', 'chamber_temperature', 
                'injection_pressure', 'density', 'viscosity']].values

# Batch prediction
predictions = model.predict(X)

# Add predictions to DataFrame
conditions['angle_shadow'] = predictions[:, 0]
conditions['penetration_shadow'] = predictions[:, 1]
conditions['angle_mie'] = predictions[:, 2]
conditions['penetration_mie'] = predictions[:, 3]

# Export results
conditions.to_csv('outputs/batch_predictions.csv', index=False)
print(f"✓ Predicted {len(conditions)} conditions")
print(conditions.head())

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(conditions['time'], conditions['angle_shadow'], label='Shadow')
plt.plot(conditions['time'], conditions['angle_mie'], label='Mie')
plt.xlabel('Time (ms)')
plt.ylabel('Spray Angle (degrees)')
plt.legend()
plt.title('Spray Angle Evolution')

plt.subplot(1, 2, 2)
plt.plot(conditions['time'], conditions['penetration_shadow'], label='Shadow')
plt.plot(conditions['time'], conditions['penetration_mie'], label='Mie')
plt.xlabel('Time (ms)')
plt.ylabel('Penetration (L/D)')
plt.legend()
plt.title('Spray Penetration Evolution')

plt.tight_layout()
plt.savefig('outputs/batch_analysis.png', dpi=300)
plt.show()
```

---

### Scenario 5: Model Retraining with New Data

**Goal**: Update models when new experimental data becomes available

**Steps**:
1. Add new data to `data/raw/`
2. Update preprocessing notebook
3. Retrain models
4. Compare with previous version

```bash
# 1. Add new Excel file
cp NewExperiments.xlsx data/raw/

# 2. Update preprocessing (add new file to notebook)
# Edit notebooks/data-preprocessing-pipeline.ipynb

# 3. Reprocess data
make preprocess

# 4. Retrain all models
make train-all

# 5. Compare performance
python compare_versions.py
```

---

## Step-by-Step Tutorials

### Tutorial 1: From Raw Data to First Prediction

**Duration**: 20 minutes

#### Step 1: Setup (2 min)
```bash
cd spray-vision
pip install -r requirements.txt
```

#### Step 2: Preprocess Data (2 min)
```bash
make preprocess
# Or manually:
jupyter nbconvert --to notebook --execute notebooks/data-preprocessing-pipeline.ipynb
```

**Check output**:
```bash
ls -lh data/processed/preprocessed_dataset.csv
# Should see ~100-200 KB file
```

#### Step 3: Train Models (15 min)
```bash
make train-all
# Or step by step:
make train-ml    # 10 min
make train-ann   # 5 min
```

**Check models**:
```bash
ls -lh models/
# Should see 9 files (6 ML models + ANN + 2 scalers)
```

#### Step 4: Make First Prediction (1 min)
```python
import joblib
import numpy as np

# Load model
model = joblib.load("models/KNN_regressor.joblib")

# Input: time=1ms, Pc=55bar, Tc=650K, Pinj=800bar, ρ=780kg/m³, μ=0.002Pa·s
X = np.array([[1.0, 55.0, 650.0, 800.0, 780.0, 0.002]])

# Predict
pred = model.predict(X)[0]

print("Predictions:")
print(f"  Spray angle (shadow): {pred[0]:.2f} degrees")
print(f"  Penetration (shadow): {pred[1]:.2f} L/D")
print(f"  Spray angle (Mie): {pred[2]:.2f} degrees")
print(f"  Penetration (Mie): {pred[3]:.2f} L/D")
```

**Expected output**:
```
Predictions:
  Spray angle (shadow): 18.23 degrees
  Penetration (shadow): 25.67 L/D
  Spray angle (Mie): 14.89 degrees
  Penetration (Mie): 32.45 L/D
```

**Success!** 🎉 You've trained models and made your first prediction.

---

### Tutorial 2: Comparing Different Models

**Duration**: 10 minutes

#### Step 1: Load All Models
```python
import joblib
import numpy as np
import pandas as pd

models = {
    'LinearRegression': joblib.load("models/LinearRegression_regressor.joblib"),
    'DecisionTree': joblib.load("models/DecisionTree_regressor.joblib"),
    'KNN': joblib.load("models/KNN_regressor.joblib"),
    'SVR': joblib.load("models/SVR_regressor.joblib"),
    'RandomForest': joblib.load("models/RandomForest_regressor.joblib"),
    'GradientBoosting': joblib.load("models/GradientBoosting_regressor.joblib")
}
```

#### Step 2: Prepare Test Input
```python
# Example condition
X = np.array([[2.0, 55.0, 650.0, 800.0, 780.0, 0.002]])
```

#### Step 3: Compare Predictions
```python
results = []

for name, model in models.items():
    pred = model.predict(X)[0]
    results.append({
        'model': name,
        'angle_shadow': pred[0],
        'penetration_shadow': pred[1],
        'angle_mie': pred[2],
        'penetration_mie': pred[3]
    })

df = pd.DataFrame(results)
print(df.round(2))
```

#### Step 4: Analyze Differences
```python
# Calculate std dev across models (measure of disagreement)
print("\nPrediction Uncertainty (std dev across models):")
print(df[['angle_shadow', 'penetration_shadow', 'angle_mie', 'penetration_mie']].std())

# Best and worst case
print("\nRange of predictions:")
for col in ['angle_shadow', 'penetration_shadow', 'angle_mie', 'penetration_mie']:
    print(f"{col}: {df[col].min():.2f} - {df[col].max():.2f}")
```

---

### Tutorial 3: Creating Custom Visualizations

**Duration**: 15 minutes

#### Step 1: Generate Predictions Across Time
```python
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Time series: 0 to 5 ms
time_points = np.linspace(0, 5, 100)

# Fixed conditions
conditions = np.array([
    [t, 55.0, 650.0, 800.0, 780.0, 0.002] 
    for t in time_points
])

# Predict
predictions = model.predict(conditions)

# Create DataFrame
df = pd.DataFrame({
    'time': time_points,
    'angle_shadow': predictions[:, 0],
    'penetration_shadow': predictions[:, 1],
    'angle_mie': predictions[:, 2],
    'penetration_mie': predictions[:, 3]
})
```

#### Step 2: Create Multi-Panel Plot
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Spray angles
axes[0, 0].plot(df['time'], df['angle_shadow'], 'b-', linewidth=2, label='Shadow')
axes[0, 0].plot(df['time'], df['angle_mie'], 'r--', linewidth=2, label='Mie')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Spray Angle (degrees)')
axes[0, 0].legend()
axes[0, 0].set_title('Spray Angle Evolution')

# Penetrations
axes[0, 1].plot(df['time'], df['penetration_shadow'], 'b-', linewidth=2, label='Shadow')
axes[0, 1].plot(df['time'], df['penetration_mie'], 'r--', linewidth=2, label='Mie')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Penetration (L/D)')
axes[0, 1].legend()
axes[0, 1].set_title('Penetration Evolution')

# Rate of change
axes[1, 0].plot(df['time'][1:], np.diff(df['angle_shadow']), 'b-', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Angle Growth Rate (deg/ms)')
axes[1, 0].set_title('Spray Expansion Rate')

axes[1, 1].plot(df['time'][1:], np.diff(df['penetration_shadow']), 'b-', linewidth=2)
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Penetration Rate (L/D per ms)')
axes[1, 1].set_title('Spray Advancement Rate')

plt.tight_layout()
plt.savefig('outputs/custom_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Common Tasks

### Task 1: Check Model Performance

```bash
# View overall metrics
cat outputs/overall_model_metrics.csv

# Or in Python
import pandas as pd
metrics = pd.read_csv("outputs/overall_model_metrics.csv")
print(metrics.sort_values('overall_r2', ascending=False))
```

### Task 2: List Available Models

```bash
make check-models

# Or manually
ls -lh models/
```

### Task 3: Clean Temporary Files

```bash
make clean

# Removes:
# - .pyc files
# - __pycache__ directories
# - .ipynb_checkpoints
# - Temporary files
```

### Task 4: Update Dependencies

```bash
# Update to latest compatible versions
pip install --upgrade -r requirements.txt

# Or freeze current environment
make freeze-requirements
```

### Task 5: Run Specific Notebook

```bash
# Data preprocessing only
make preprocess

# ML models only
make train-ml

# Neural network only
make train-ann

# Interactive mode
jupyter notebook notebooks/ml-pipeline.ipynb
```

### Task 6: Export Predictions

```python
import joblib
import pandas as pd

# Load test data
test_data = pd.read_csv("data/processed/preprocessed_dataset.csv")
X_test = test_data[['Time_ms', 'Pc_bar', 'Tc_K', 'Pinj_bar', 'rho_kgm3', 'mu_Pas']]

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Predict
predictions = model.predict(X_test.values)

# Create results DataFrame
results = test_data.copy()
results['pred_angle_shadow'] = predictions[:, 0]
results['pred_pen_shadow'] = predictions[:, 1]
results['pred_angle_mie'] = predictions[:, 2]
results['pred_pen_mie'] = predictions[:, 3]

# Export
results.to_csv('outputs/all_predictions.csv', index=False)
print(f"✓ Exported {len(results)} predictions")
```

### Task 7: Calculate Custom Metrics

```python
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score

# Load actuals and predictions
actuals = pd.read_csv("outputs/ANN_improved_actuals.csv")
predictions = pd.read_csv("outputs/ANN_improved_predictions.csv")

# Calculate MAPE (Mean Absolute Percentage Error)
for i, col in enumerate(actuals.columns):
    mape = mean_absolute_percentage_error(actuals.iloc[:, i], predictions.iloc[:, i])
    print(f"{col} MAPE: {mape*100:.2f}%")

# Calculate Explained Variance
for i, col in enumerate(actuals.columns):
    ev = explained_variance_score(actuals.iloc[:, i], predictions.iloc[:, i])
    print(f"{col} Explained Variance: {ev:.4f}")
```

---

## Troubleshooting

### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install -r requirements.txt
# Or specific package
pip install scikit-learn
```

---

### Issue 2: Data File Not Found

**Error**: `FileNotFoundError: data/raw/DataSheet_ETH_250902.xlsx`

**Solution**:
```bash
# Ensure raw data exists
make check-data

# If missing, obtain the Excel file and place in data/raw/
```

---

### Issue 3: Out of Memory

**Error**: `MemoryError` during Random Forest training

**Solution**:
```python
# Reduce hyperparameter grid in notebooks/ml-pipeline.ipynb
param_grid = {
    'n_estimators': [50, 100],      # Reduced from [50, 100, 200]
    'max_depth': [10, 20],           # Reduced from [10, 20, 30, None]
    # ... other parameters
}
```

---

### Issue 4: TensorFlow GPU Errors

**Error**: `Could not load dynamic library 'cudart64_110.dll'`

**Solution**: This is a warning, not an error. TensorFlow will use CPU automatically.
To suppress warnings:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import tensorflow as tf
```

---

### Issue 5: Jupyter Kernel Dies

**Error**: Kernel crashes during notebook execution

**Solution**:
```bash
# Increase memory limit
# Run notebooks individually instead of "Run All"

# Or execute from command line
jupyter nbconvert --to notebook --execute notebooks/ml-pipeline.ipynb --ExecutePreprocessor.timeout=600
```

---

### Issue 6: Model Loading Fails

**Error**: `ValueError: cannot reshape array` or version mismatch

**Solution**:
```bash
# Check scikit-learn version
python -c "import sklearn; print(sklearn.__version__)"

# Should match training version (1.7.2)
pip install scikit-learn==1.7.2

# Or retrain models with current version
make train-all
```

---

## FAQ

### Q1: Which model should I use for production?

**A**: **Gradient Boosting** is recommended for most production scenarios:
- Excellent accuracy (R² = 0.9949)
- Reasonable file size (4.3 MB)
- Fast inference
- Good balance of performance and efficiency

Use **KNN** if maximum accuracy is critical and slow inference is acceptable.

---

### Q2: How long does training take?

**A**: On a modern CPU:
- Data preprocessing: ~30 seconds
- ML models (all 6): ~10-15 minutes
- Neural network: ~2-5 minutes
- **Total**: ~15-20 minutes

With GPU (for ANN only): ~10-12 minutes total

---

### Q3: Can I add more input features?

**A**: Yes! Modify the preprocessing notebook:
1. Extract new features from Excel or add new data source
2. Update `INPUTS` list in training notebooks
3. Retrain all models

Example:
```python
INPUTS = ["time", "chamb_pressure", "cham_temp", "injection_pres", 
          "density", "viscosity", "nozzle_diameter"]  # Added new feature
```

---

### Q4: How do I handle missing input values?

**A**: For prediction with missing values:
```python
# Option 1: Use median/mean from training data
X_new[missing_idx] = training_median

# Option 2: Use models that handle missing values (XGBoost)
# Option 3: Impute before prediction
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_new)
```

---

### Q5: Can I deploy models without Python?

**A**: Partially. Options:
1. **ONNX Export**: Convert models to ONNX format for use in C++/Java/C#
2. **Model Server**: Use TensorFlow Serving or TorchServe
3. **REST API**: Wrap Python models in Flask/FastAPI and deploy as service
4. **Mobile**: TensorFlow Lite for mobile deployment

For complete language-agnostic deployment, ONNX is best:
```python
from skl2onnx import convert_sklearn
onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, 6]))])
```

---

### Q6: How accurate are the predictions?

**A**: Very accurate for the ETH dataset:
- Best model (KNN): R² = 0.9988 (99.88% variance explained)
- Recommended model (Gradient Boosting): R² = 0.9949
- Mean Absolute Error: < 1.0 for most targets

**However**, accuracy may decrease for:
- Operating conditions outside training range
- Different fuel types
- Different injector designs

Always validate on your specific data.

---

### Q7: What if my data is in CSV format?

**A**: Modify `data-preprocessing-pipeline.ipynb`:
```python
# Instead of reading Excel:
# df = pd.read_excel(EXCEL_PATH, ...)

# Read CSV:
df = pd.read_csv("data/raw/my_data.csv")

# Ensure same column structure
df = df.rename(columns={
    'your_time_col': 'Time_ms',
    'your_pressure_col': 'Pc_bar',
    # ... etc
})
```

---

### Q8: Can I use this for real-time control?

**A**: Yes, with caveats:
- **Prediction time**: < 1 ms for most models
- **Suitable models**: Decision Tree, Linear Regression, Gradient Boosting
- **Not suitable**: KNN (too slow)
- **Setup**: Load model once at startup, predict in loop

Example:
```python
# Startup
model = joblib.load("models/DecisionTree_regressor.joblib")

# Control loop
while running:
    current_conditions = read_sensors()
    X = np.array([current_conditions])
    predictions = model.predict(X)[0]
    adjust_actuators(predictions)
```

---

### Q9: How do I cite this project?

**A**: If using in research:
```bibtex
@software{spray_vision_2024,
  title={Spray Vision: ML-Based Spray Characteristics Prediction},
  author={Yuvin Raja},
  year={2024},
  version={0.1.0},
  url={<repository-url>}
}
```

---

### Q10: Where can I get help?

**A**: 
1. Check this documentation
2. Review notebook comments
3. Examine example code in this guide
4. Open issue on GitHub repository
5. Review referenced papers in `references/`

---

## Next Steps

After completing this guide:

1. **Experiment**: Try different operating conditions
2. **Visualize**: Generate custom plots for your use case
3. **Optimize**: Use models to find optimal injection parameters
4. **Deploy**: Integrate models into your application
5. **Contribute**: Improve models, add features, share results

**Happy predicting! 🚀**

---

*This user guide provides practical instructions for using the Spray Vision project for various applications.*
