# Spray Vision: Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Machine Learning Models](#machine-learning-models)
4. [Neural Network Architecture](#neural-network-architecture)
5. [Performance Metrics](#performance-metrics)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Technical Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│                                                              │
│  Jupyter Notebooks (Interactive Development & Analysis)     │
│  • data-preprocessing-pipeline.ipynb                        │
│  • exploratory-data-analysis.ipynb                          │
│  • ml-pipeline.ipynb                                        │
│  • ann-pipeline.ipynb                                       │
│  • fuel-journal-plots.ipynb                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│                                                              │
│  Input: Excel (DataSheet_ETH_250902.xlsx)                   │
│  Processing: Pandas DataFrames                              │
│  Storage: CSV files (processed data)                        │
│  Format: Structured tabular data                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  ML/COMPUTATION LAYER                        │
│                                                              │
│  Classical ML: Scikit-learn 1.7.2                           │
│  • Estimators, Transformers, Pipelines                      │
│  • Grid Search CV, Multi-output Regression                  │
│                                                              │
│  Deep Learning: TensorFlow 2.20 + Keras 3.11                │
│  • Sequential Models, Dense Layers                          │
│  • Optimizers (Adam), Callbacks (EarlyStopping)             │
│                                                              │
│  Scientific Computing: NumPy 2.3.3, SciPy 1.16.2            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER                          │
│                                                              │
│  Model Storage: Joblib (scikit-learn), HDF5 (Keras)        │
│  Data Storage: CSV files                                     │
│  Versioning: Git                                             │
│  Outputs: CSV metrics, PNG visualizations                   │
└─────────────────────────────────────────────────────────────┘
```

### Component Dependencies

```
Application Layer (Notebooks)
    ↓
Data Processing (Pandas, NumPy)
    ↓
Feature Engineering (Scikit-learn Preprocessing)
    ↓
┌─────────────────┬─────────────────┐
│   Classical ML  │  Deep Learning  │
│  (Scikit-learn) │ (TensorFlow)    │
└─────────────────┴─────────────────┘
    ↓
Model Serialization (Joblib, HDF5)
    ↓
Visualization (Matplotlib, Seaborn)
```

---

## Data Processing Pipeline

### Stage 1: Excel Data Extraction

**Notebook**: `data-preprocessing-pipeline.ipynb` (10 code cells, 278 lines)

#### Input Specification
- **File**: `DataSheet_ETH_250902.xlsx`
- **Format**: Multi-sheet Excel workbook
- **Size**: Contains 7 experimental runs × 121 time points

#### Excel Structure
The Excel file contains multiple sheets with two-level headers:
- **Level 1**: Variable names (e.g., "Chamber pressure (bar)")
- **Level 2**: Experimental run identifiers (e.g., "ETH-01", "ETH-02", ...)

#### Key Sheets
1. **"Exp. Conditions in Time"**: Input variables
   - Chamber pressure (bar)
   - Chamber temperature (K)
   - Injection pressure (bar)
   - Density (kg/m³)
   - Viscosity (Pa·s)

2. **"Spray Angle (Shadow)"**: Smoothed angle measurements
3. **"Spray Penetration (Shadow)"**: L/D ratio measurements
4. **"Spray Angle (Mie)"**: Mie scattering angle data
5. **"Spray Penetration (Mie)"**: Mie scattering penetration

### Helper Functions

#### 1. `read_multiheader(sheet_name)`
```python
Purpose: Parse Excel sheets with two-row headers
Input: Sheet name string
Output: DataFrame with MultiIndex columns
Process:
  - Read Excel with header=[0, 1]
  - Normalize column names
  - Create MultiIndex with levels ['var', 'run']
```

#### 2. `get_time_series(df_multi)`
```python
Purpose: Extract time column from multi-level DataFrame
Input: DataFrame with MultiIndex columns
Output: Series with time values (renamed to 'Time_ms')
Process:
  - Find columns with top-level name "Time(ms)"
  - Extract and convert to float
  - Validate single time column exists
```

#### 3. `stack_block(df_multi, top_name, value_name)`
```python
Purpose: Convert wide-format block to long-format
Input:
  - df_multi: DataFrame with MultiIndex
  - top_name: Variable block name
  - value_name: Name for values in long format
Output: Long-format DataFrame with ['run', 'Time_ms', value_name]
Process:
  - Extract time series
  - Filter columns for specific variable block
  - Rename columns to run identifiers
  - Melt to long format
  - Convert to numeric types
```

#### 4. `groupwise_ffill_bfill(df, group_cols, order_cols, cols_to_fill)`
```python
Purpose: Fill missing values within groups
Input:
  - df: Input DataFrame
  - group_cols: Columns defining groups
  - order_cols: Columns for sorting
  - cols_to_fill: Columns to apply filling
Output: DataFrame with filled values
Strategy:
  - Sort by group and order columns
  - Within each group:
    - Forward fill (propagate last valid value)
    - Backward fill (propagate next valid value)
  - Ensures no NaN values remain if possible
```

### Data Transformation Steps

#### Step 1: Load and Stack Input Variables
```python
# Extract each input variable separately
inp_pc = stack_block(ect, "Chamber pressure (bar)", "Pc_bar")
inp_tc = stack_block(ect, "Chamber temperature (K)", "Tc_K")
inp_pinj = stack_block(ect, "Injection pressure (bar)", "Pinj_bar")
inp_rho = stack_block(ect, "Density (kg/m3)", "rho_kgm3")
inp_mu = stack_block(ect, "Viscosity (Pas)", "mu_Pas")

# Merge using outer join to preserve all time points
inputs = reduce(
    lambda left, right: pd.merge(left, right, on=["run", "Time_ms"], how="outer"),
    [inp_pc, inp_tc, inp_pinj, inp_rho, inp_mu]
)
```

#### Step 2: Load and Stack Target Variables
```python
# Shadow method targets
tgt_ang_sh = stack_block(shadow_angle_df, "Smoothed angle (deg)", "angle_shadow_deg")
tgt_len_sh = stack_block(shadow_len_df, "Penetration (L/D)", "len_shadow_L_D")

# Mie scattering targets
tgt_ang_mie = stack_block(mie_angle_df, "Smoothed angle (deg)", "angle_mie_deg")
tgt_len_mie = stack_block(mie_len_df, "Penetration (L/D)", "len_mie_L_D")

# Merge all targets
targets = reduce(
    lambda left, right: pd.merge(left, right, on=["run", "Time_ms"], how="outer"),
    [tgt_ang_sh, tgt_len_sh, tgt_ang_mie, tgt_len_mie]
)
```

#### Step 3: Combine Inputs and Targets
```python
# Full outer join to ensure no data loss
merged = pd.merge(inputs, targets, on=["run", "Time_ms"], how="outer")

# Reorder columns for clarity
cols_order = [
    "run", "Time_ms",
    "Pc_bar", "Tc_K", "Pinj_bar", "rho_kgm3", "mu_Pas",
    "angle_shadow_deg", "len_shadow_L_D", "angle_mie_deg", "len_mie_L_D"
]
merged = merged[cols_order]
```

#### Step 4: Handle Missing Values
```python
# Apply forward and backward fill within each experimental run
key_cols = ["run", "Time_ms"]
num_cols = [c for c in merged.columns if c not in key_cols]

filled = groupwise_ffill_bfill(
    merged.copy(),
    group_cols=["run"],
    order_cols=["Time_ms"],
    cols_to_fill=num_cols
)
```

#### Step 5: Validation and Export
```python
# Sort for consistency
filled = filled.sort_values(["run", "Time_ms"]).reset_index(drop=True)

# Integrity checks
assert filled[["run", "Time_ms"]].drop_duplicates().shape[0] == filled.shape[0]
assert expected_cols.issubset(set(filled.columns))

# Save to CSV
filled.to_csv(OUTPUT_CSV, index=False)
```

### Output Specification

**File**: `data/processed/preprocessed_dataset.csv`

**Schema**:
```
Column Name          Type     Description                    Unit
─────────────────────────────────────────────────────────────────
run                  str      Experiment identifier          -
Time_ms              float    Time point                     ms
Pc_bar               float    Chamber pressure               bar
Tc_K                 float    Chamber temperature            K
Pinj_bar             float    Injection pressure             bar
rho_kgm3             float    Fuel density                   kg/m³
mu_Pas               float    Fuel viscosity                 Pa·s
angle_shadow_deg     float    Spray angle (shadow)           degrees
len_shadow_L_D       float    Penetration (shadow)           L/D ratio
angle_mie_deg        float    Spray angle (Mie)              degrees
len_mie_L_D          float    Penetration (Mie)              L/D ratio
```

**Dimensions**: 847 rows × 11 columns

**Characteristics**:
- No missing values after imputation
- Sorted by run and time
- No duplicate (run, time) pairs
- All numeric columns except 'run'

---

## Machine Learning Models

### Training Pipeline Overview

**Notebook**: `ml-pipeline.ipynb` (9 code cells, 279 lines)

### Data Preparation

#### Feature and Target Selection
```python
INPUTS = ["time", "chamb_pressure", "cham_temp", "injection_pres", "density", "viscosity"]
TARGETS = ["angle_shadow", "penetration_shadow", "angle_mie", "penetration_mie"]
```

#### Train-Test Split
```python
# Stratified split by experimental run
X, y = df[INPUTS], df[TARGETS]
runs = df["run"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=runs
)

# Results: Train: 677 samples (80%), Test: 170 samples (20%)
```

### Model Configurations

#### 1. Linear Regression
```python
Model: LinearRegression()
Wrapper: MultiOutputRegressor
Features: No preprocessing
Hyperparameters: None (closed-form solution)
Output File: LinearRegression_regressor.joblib (2.5 KB)

Characteristics:
  - Fastest training
  - Interpretable coefficients
  - Assumes linear relationships
  - Baseline model for comparison
```

#### 2. Decision Tree Regressor
```python
Model: DecisionTreeRegressor
Wrapper: MultiOutputRegressor
Hyperparameters (Grid Search):
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
CV Folds: 5
Output File: DecisionTree_regressor.joblib (327 KB)

Characteristics:
  - Non-linear decision boundaries
  - Interpretable tree structure
  - Prone to overfitting
  - No scaling required
```

#### 3. K-Nearest Neighbors (KNN)
```python
Model: KNeighborsRegressor
Wrapper: Pipeline with StandardScaler
Hyperparameters (Grid Search):
  - n_neighbors: [3, 5, 7, 10, 15]
  - weights: ['uniform', 'distance']
  - p: [1, 2] (Manhattan or Euclidean distance)
CV Folds: 5
Output File: KNN_regressor.joblib (279 KB)

Characteristics:
  - Instance-based learning
  - Sensitive to feature scaling
  - Memory-intensive
  - No training phase
```

#### 4. Support Vector Regression (SVR)
```python
Model: SVR
Wrapper: Pipeline with StandardScaler + MultiOutputRegressor
Hyperparameters (Grid Search):
  - kernel: ['rbf', 'linear']
  - C: [0.1, 1, 10, 100]
  - epsilon: [0.01, 0.1, 0.5]
  - gamma: ['scale', 'auto']
CV Folds: 5
Output File: SVR_regressor.joblib (93 KB)

Characteristics:
  - Robust to outliers
  - Effective in high dimensions
  - Requires careful tuning
  - Scaling mandatory
```

#### 5. Random Forest Regressor
```python
Model: RandomForestRegressor
Wrapper: MultiOutputRegressor
Hyperparameters (Grid Search):
  - n_estimators: [50, 100, 200]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]
  - max_features: ['sqrt', 'log2', None]
CV Folds: 5
Output File: RandomForest_regressor.joblib (41.9 MB)

Characteristics:
  - Ensemble of decision trees
  - Reduces overfitting via bagging
  - Provides feature importance
  - Parallel training
```

#### 6. Gradient Boosting Regressor
```python
Model: GradientBoostingRegressor
Wrapper: MultiOutputRegressor
Hyperparameters (Grid Search):
  - n_estimators: [100, 200, 300]
  - learning_rate: [0.01, 0.05, 0.1]
  - max_depth: [3, 5, 7]
  - subsample: [0.8, 1.0]
CV Folds: 5
Output File: GradientBoosting_regressor.joblib (4.4 MB)

Characteristics:
  - Sequential ensemble method
  - Builds trees to correct errors
  - Often best performance
  - Sequential training (slower)
```

### Training Workflow

```python
# Pseudo-code for training pipeline
for model_name, model_config in MODELS.items():
    # 1. Create model pipeline
    if requires_scaling(model_name):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    else:
        pipeline = base_model
    
    # 2. Wrap for multi-output if needed
    if needs_multi_output_wrapper(model_name):
        pipeline = MultiOutputRegressor(pipeline)
    
    # 3. Hyperparameter tuning
    if has_hyperparameters(model_name):
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_tr, y_tr)
        best_model = grid_search.best_estimator_
    else:
        pipeline.fit(X_tr, y_tr)
        best_model = pipeline
    
    # 4. Evaluation
    y_pred = best_model.predict(X_te)
    metrics = calculate_metrics(y_te, y_pred)
    
    # 5. Save model
    joblib.dump(best_model, f"models/{model_name}_regressor.joblib")
```

### Evaluation Metrics

For each model, three metrics are computed:

#### 1. R² Score (Coefficient of Determination)
```python
r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

Interpretation:
  - 1.0: Perfect prediction
  - 0.0: Model as good as predicting mean
  - Negative: Model worse than baseline
Formula: 1 - (SS_res / SS_tot)
```

#### 2. Mean Absolute Error (MAE)
```python
mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')

Interpretation:
  - Average absolute deviation from true values
  - Same units as target variables
  - Robust to outliers
Formula: mean(|y_true - y_pred|)
```

#### 3. Mean Squared Error (MSE)
```python
mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

Interpretation:
  - Average squared deviation
  - Penalizes large errors more heavily
  - Units are squared
Formula: mean((y_true - y_pred)²)
```

### Model Performance Results

From `outputs/overall_model_metrics.csv`:

```
Model                    R²        MAE      MSE      Rank
────────────────────────────────────────────────────────
KNN                   0.9988     0.74     2.20      1
GradientBoosting      0.9949     0.48     0.78      2
DecisionTree          0.9936     1.11     4.16      3
SVR                   0.9883     1.29    17.13      4
RandomForest          0.9830     2.21    20.73      5
LinearRegression      0.8341     4.78    78.29      6
```

**Analysis**:
- All nonlinear models achieve R² > 0.98
- KNN performs best but stores training data
- Gradient Boosting offers best balance of accuracy and size
- Linear Regression underperforms, indicating nonlinearity
- Decision Tree surprisingly competitive despite overfitting risk

---

## Neural Network Architecture

### Training Pipeline

**Notebook**: `ann-pipeline.ipynb` (8 code cells, 150 lines)

### Data Preparation for ANN

#### Scaling Strategy
```python
# Separate scalers for inputs and outputs
input_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit on training data only
X_train_scaled = input_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train)

# Transform test data
X_test_scaled = input_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test)

# Save scalers for inference
joblib.dump(input_scaler, "models/ann_input_scaler.joblib")
joblib.dump(target_scaler, "models/ann_target_scaler.joblib")
```

**Rationale**:
- Neural networks sensitive to input scale
- Standardization (zero mean, unit variance) improves convergence
- Separate target scaling for proper inverse transform
- Scalers essential for deployment

### Architecture Design

#### Improved ANN Model
```python
model = models.Sequential([
    # Input layer
    layers.Dense(64, activation='relu', input_shape=(n_inputs,)),
    
    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    
    # Output layer
    layers.Dense(n_outputs, activation='linear')
])

Architecture Summary:
  Input: 6 features
  Hidden: [64, 128, 64, 32] neurons with ReLU
  Output: 4 targets with linear activation
  Total Parameters: ~14,000
```

**Design Choices**:
- **ReLU Activation**: Prevents vanishing gradients, fast computation
- **Expanding-Contracting**: 64→128→64→32 captures complex patterns
- **Linear Output**: Regression requires unbounded predictions
- **Depth**: 4 hidden layers balance capacity and overfitting risk

### Compilation Configuration

```python
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

Optimizer: Adam
  - Adaptive learning rates per parameter
  - Momentum and RMSprop combined
  - Learning rate: 0.001 (standard default)

Loss: MSE
  - Differentiable
  - Penalizes large errors
  - Standard for regression

Metrics: MAE
  - Monitoring metric
  - More interpretable than MSE
```

### Training Configuration

```python
# Early stopping callback
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Training
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,  # 20% of training for validation
    epochs=500,            # Maximum epochs
    batch_size=32,         # Mini-batch size
    callbacks=[early_stop],
    verbose=1
)

Training Strategy:
  - Validation split: 20% (from training set)
  - Early stopping patience: 20 epochs
  - Batch size: 32 (memory vs. stability trade-off)
  - Max epochs: 500 (usually stops earlier)
```

### Model Persistence

```python
# Save in HDF5 format
model.save("models/ANN_improved_regressor.h5")

# Loading for inference
from keras.models import load_model
loaded_model = load_model("models/ANN_improved_regressor.h5")

# Prediction with inverse scaling
y_pred_scaled = loaded_model.predict(X_new_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
```

### Neural Network Advantages

1. **Universal Approximation**: Can model any continuous function
2. **Feature Learning**: Automatically discovers useful representations
3. **Scalability**: Efficient for large datasets
4. **Fine-Tuning**: Transfer learning possible
5. **GPU Acceleration**: TensorFlow automatically uses GPUs if available

---

## Performance Metrics

### Overall Metrics

From `outputs/overall_model_metrics.csv`:

| Model | Overall R² | Overall MAE | Overall MSE |
|-------|-----------|-------------|-------------|
| KNN | 0.9988 | 0.7406 | 2.1986 |
| GradientBoosting | 0.9949 | 0.4812 | 0.7828 |
| DecisionTree | 0.9936 | 1.1068 | 4.1551 |
| SVR | 0.9883 | 1.2921 | 17.1278 |
| RandomForest | 0.9830 | 2.2071 | 20.7300 |
| LinearRegression | 0.8341 | 4.7795 | 78.2915 |

### Per-Target Metrics

From `outputs/per_target_metrics.csv`:

Detailed metrics for each of the 4 targets:
1. **angle_shadow**: Spray angle from shadow method
2. **penetration_shadow**: Penetration length from shadow method
3. **angle_mie**: Spray angle from Mie scattering
4. **penetration_mie**: Penetration length from Mie scattering

Each target evaluated independently for:
- R² score
- MAE (in original units)
- MSE (in original units²)

### Model Size Comparison

| Model | File Size | Load Time | Inference Speed |
|-------|-----------|-----------|-----------------|
| LinearRegression | 2.5 KB | < 1ms | Fastest |
| SVR | 93 KB | ~10ms | Fast |
| KNN | 279 KB | ~50ms | Moderate |
| DecisionTree | 327 KB | ~50ms | Fast |
| ANN | 594 KB | ~100ms | Fast (GPU: Fastest) |
| GradientBoosting | 4.4 MB | ~200ms | Moderate |
| RandomForest | 41.9 MB | ~1000ms | Slow |

**Recommendations**:
- **Production Deployment**: GradientBoosting or ANN (balance of accuracy and size)
- **Real-Time Systems**: LinearRegression or DecisionTree (fast inference)
- **Maximum Accuracy**: KNN or GradientBoosting
- **Interpretability**: DecisionTree or LinearRegression

---

## API Reference

### Model Loading and Inference

#### Classical ML Models
```python
import joblib

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Prepare input (must match training features order)
X_new = [[time, chamb_pressure, cham_temp, injection_pres, density, viscosity]]

# Predict
predictions = model.predict(X_new)
# Returns: [angle_shadow, penetration_shadow, angle_mie, penetration_mie]
```

#### Neural Network
```python
import joblib
from keras.models import load_model

# Load model and scalers
model = load_model("models/ANN_improved_regressor.h5")
input_scaler = joblib.load("models/ann_input_scaler.joblib")
target_scaler = joblib.load("models/ann_target_scaler.joblib")

# Prepare and scale input
X_new = [[time, chamb_pressure, cham_temp, injection_pres, density, viscosity]]
X_scaled = input_scaler.transform(X_new)

# Predict and inverse scale
y_scaled = model.predict(X_scaled)
predictions = target_scaler.inverse_transform(y_scaled)
```

### Data Processing Functions

From `data-preprocessing-pipeline.ipynb`:

```python
def read_multiheader(sheet_name):
    """Read Excel sheet with two-row header."""
    pass

def get_time_series(df_multi):
    """Extract time column from MultiIndex DataFrame."""
    pass

def stack_block(df_multi, top_name, value_name):
    """Convert wide-format block to long-format."""
    pass

def groupwise_ffill_bfill(df, group_cols, order_cols, cols_to_fill=None):
    """Fill missing values within groups."""
    pass
```

---

## Configuration

### Makefile Commands

```bash
# Installation
make requirements           # Install dependencies
make dev-requirements      # Install with dev tools
make freeze-requirements   # Update requirements.txt

# Data Processing
make preprocess           # Run data preprocessing
make setup               # Complete project setup

# Model Training
make train-ml            # Train classical ML models
make train-ann           # Train neural network
make train-all           # Run complete pipeline

# Development
make notebook            # Start Jupyter server
make test-notebooks      # Test notebook execution
make lint               # Check code style
make format             # Format code
make clean              # Clean temporary files

# Utilities
make check-data         # Verify data files
make check-models       # List trained models
make report             # Project status report
make help               # Show all commands
```

### Environment Variables

None required. All paths are relative to project root.

### Configuration Files

- **pyproject.toml**: Package metadata, dependencies, tool configuration
- **requirements.txt**: Pinned dependencies
- **Makefile**: Automation commands
- **.gitignore**: Files excluded from version control

---

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
```
Error: ModuleNotFoundError: No module named 'sklearn'
Solution: pip install -r requirements.txt
```

#### 2. Data File Not Found
```
Error: FileNotFoundError: data/raw/DataSheet_ETH_250902.xlsx
Solution: Ensure Excel file is in data/raw/ directory
```

#### 3. CUDA/GPU Errors (TensorFlow)
```
Error: Could not load dynamic library 'cudart64_110.dll'
Solution: TensorFlow will fall back to CPU automatically
Note: GPU not required, training is fast enough on CPU
```

#### 4. Memory Issues
```
Error: MemoryError during RandomForest training
Solution: Reduce max_depth or n_estimators in hyperparameter grid
```

#### 5. Notebook Execution Fails
```
Error: Cell execution timeout
Solution: Increase timeout or run cells individually
```

### Debugging Tips

1. **Check Data Integrity**:
   ```bash
   make check-data
   python -c "import pandas as pd; print(pd.read_csv('data/processed/preprocessed_dataset.csv').info())"
   ```

2. **Verify Model Files**:
   ```bash
   make check-models
   ```

3. **Test Model Loading**:
   ```python
   import joblib
   model = joblib.load("models/KNN_regressor.joblib")
   print(model)
   ```

4. **Validate Predictions**:
   ```python
   import numpy as np
   assert predictions.shape == (n_samples, 4), "Wrong output shape"
   assert not np.isnan(predictions).any(), "NaN in predictions"
   ```

### Performance Optimization

1. **Parallel Training**: Scikit-learn uses `n_jobs=-1` by default
2. **Batch Predictions**: Process multiple samples together
3. **Model Selection**: Use lighter models for deployment
4. **GPU Acceleration**: TensorFlow automatically uses available GPUs

### Getting Help

1. Check README.md for quick start
2. Review notebook comments for detailed explanations
3. Examine output CSVs for diagnostic information
4. Run `make report` for project status
5. Inspect plots/ directory for visual diagnostics

---

## Version History

- **v0.1.0** (Initial Release):
  - Complete data preprocessing pipeline
  - 6 classical ML models + ANN
  - Comprehensive evaluation and visualization
  - Production-ready trained models
  - Full documentation

---

## Technical Specifications

### System Requirements
- **OS**: Windows, Linux, macOS
- **Python**: 3.10 - 3.11
- **RAM**: Minimum 4 GB (8 GB recommended for RandomForest)
- **Storage**: ~500 MB for code, data, and models
- **GPU**: Optional (TensorFlow will use if available)

### Execution Time Estimates
- Data Preprocessing: ~30 seconds
- ML Training (with grid search): ~5-15 minutes
- ANN Training: ~2-5 minutes
- Full Pipeline: ~20-30 minutes

### Scalability
- **Data Size**: Current: 847 samples. Scalable to 100K+ samples
- **Features**: Current: 6 inputs. Easily extensible
- **Targets**: Current: 4 outputs. Supports any number
- **Models**: Easy to add new algorithms

---

*This technical documentation provides comprehensive details for developers, data scientists, and researchers working with the Spray Vision project.*
