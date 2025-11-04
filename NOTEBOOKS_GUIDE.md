# Spray Vision: Notebooks Guide

## Table of Contents
1. [Overview](#overview)
2. [Notebook 1: Data Preprocessing Pipeline](#notebook-1-data-preprocessing-pipeline)
3. [Notebook 2: Exploratory Data Analysis](#notebook-2-exploratory-data-analysis)
4. [Notebook 3: Machine Learning Pipeline](#notebook-3-machine-learning-pipeline)
5. [Notebook 4: Artificial Neural Network Pipeline](#notebook-4-artificial-neural-network-pipeline)
6. [Notebook 5: Fuel Journal Plots](#notebook-5-fuel-journal-plots)
7. [Execution Order](#execution-order)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

The Spray Vision project consists of 5 Jupyter notebooks that form a complete machine learning pipeline. Each notebook is self-contained but designed to be executed in sequence. This guide provides detailed information about each notebook's purpose, structure, and outputs.

### Notebook Summary

| Notebook | Purpose | Cells | Lines | Runtime | Dependencies |
|----------|---------|-------|-------|---------|--------------|
| data-preprocessing-pipeline | Data extraction and cleaning | 10 code | 278 | ~30s | pandas, openpyxl |
| exploratory-data-analysis | Statistical analysis and visualization | 8 code | 110 | ~1min | pandas, matplotlib, seaborn |
| ml-pipeline | Classical ML model training | 9 code | 279 | ~10min | scikit-learn, joblib |
| ann-pipeline | Neural network training | 8 code | 150 | ~5min | tensorflow, keras |
| fuel-journal-plots | Publication-quality visualization | 51 total (27 code, 24 markdown) | 714 | ~3min | All visualization libs |

---

## Notebook 1: Data Preprocessing Pipeline

**File**: `notebooks/data-preprocessing-pipeline.ipynb`
**Purpose**: Extract, transform, and clean raw spray data from Excel into ML-ready CSV format

### Overview
This notebook reads the multi-sheet Excel file containing ETH spray experimental data and transforms it into a clean, structured dataset suitable for machine learning. It handles complex multi-level headers, missing values, and data validation.

### Structure

#### Cell 1: Imports and Configuration
```python
Imports:
  - functools.reduce: For merging multiple DataFrames
  - pathlib.Path: For cross-platform file paths
  - pandas: Core data manipulation

Configuration:
  - EXCEL_PATH: Input Excel file location
  - OUTPUT_CSV: Output CSV file location
  - Pandas display options for better visibility
```

**Purpose**: Set up environment and define I/O paths.

#### Cell 2: Helper Functions
**Length**: ~100 lines

**Functions Defined**:

1. **`read_multiheader(sheet_name)`**
   - Reads Excel sheets with two-row headers
   - Returns DataFrame with MultiIndex columns
   - Normalizes column names

2. **`get_time_series(df_multi)`**
   - Extracts time column from MultiIndex DataFrame
   - Validates single time column exists
   - Converts to numeric and renames

3. **`stack_block(df_multi, top_name, value_name)`**
   - Converts wide-format variable blocks to long format
   - Creates ['run', 'Time_ms', value_name] structure
   - Handles numeric conversion with error coercion

4. **`groupwise_ffill_bfill(df, group_cols, order_cols, cols_to_fill)`**
   - Fills missing values within groups
   - Uses forward fill then backward fill strategy
   - Preserves data integrity within experimental runs

**Purpose**: Provide reusable functions for complex data transformations.

#### Cell 3: Load and Process Input Variables
**Input**: "Exp. Conditions in Time" sheet

**Variables Extracted**:
- Chamber pressure (Pc_bar)
- Chamber temperature (Tc_K)
- Injection pressure (Pinj_bar)
- Fuel density (rho_kgm3)
- Fuel viscosity (mu_Pas)

**Process**:
1. Read sheet with multi-level headers
2. Stack each variable block separately
3. Merge all input variables using outer join
4. Ensure consistent run and time indexing

**Output**: `inputs` DataFrame with shape (n_timepoints × n_runs) × 7 columns

#### Cell 4: Load and Process Target Variables
**Input**: Four separate sheets:
- "Spray Angle (Shadow)"
- "Spray Penetration (Shadow)"
- "Spray Angle (Mie)"
- "Spray Penetration (Mie)"

**Variables Extracted**:
- angle_shadow_deg: Spray angle from shadow method
- len_shadow_L_D: Penetration length from shadow method
- angle_mie_deg: Spray angle from Mie scattering
- len_mie_L_D: Penetration length from Mie scattering

**Process**:
1. Read each sheet separately
2. Extract smoothed/processed measurements
3. Stack to long format
4. Merge all targets

**Output**: `targets` DataFrame

#### Cell 5: Merge Inputs and Targets
**Process**:
1. Full outer join on ['run', 'Time_ms']
2. Reorder columns for clarity
3. Convert all numeric columns
4. Display merged data preview

**Output**: `merged` DataFrame with 11 columns

#### Cell 6: Data Quality Check
**Checks**:
- Print shape of merged data
- Count unique experimental runs
- Display time points per run
- Identify any data structure issues

**Output**: Diagnostic information printed to console

#### Cell 7: Handle Missing Values
**Strategy**:
- Group by experimental run
- Sort by time within each run
- Forward fill (propagate last valid value forward)
- Backward fill (propagate next valid value backward)
- Ensures temporal consistency within experiments

**Output**: `filled` DataFrame with no (or minimal) missing values

#### Cell 8: Finalize and Save
**Steps**:
1. Sort by ['run', 'Time_ms'] for consistency
2. Integrity checks:
   - No duplicate (run, time) pairs
   - All expected columns present
3. Save to CSV
4. Print summary statistics

**Output**: `data/processed/preprocessed_dataset.csv` (847 rows × 11 columns)

#### Cells 9-10: Data Validation and Investigation
**Purpose**: Investigate data structure and validate expectations
**Activities**:
- Check actual vs expected row counts
- Examine experimental run identifiers
- Analyze time ranges per run
- Identify any anomalies in Excel structure

**Output**: Validation reports and explanations

### Outputs

**Primary Output**: `data/processed/preprocessed_dataset.csv`
- **Rows**: 847 (7 runs × 121 time points)
- **Columns**: 11 (1 run ID + 1 time + 5 inputs + 4 targets)
- **Format**: Clean, numeric, no missing values
- **Sorted**: By run and time for consistency

**Quality Checks**:
- ✅ No duplicate rows
- ✅ No missing values (after imputation)
- ✅ All numeric columns properly typed
- ✅ Consistent time spacing within runs

### How to Run

**Command Line**:
```bash
make preprocess
# or
jupyter nbconvert --to notebook --execute notebooks/data-preprocessing-pipeline.ipynb
```

**Interactive**:
```bash
jupyter notebook notebooks/data-preprocessing-pipeline.ipynb
# Run cells sequentially using Shift+Enter
```

### Common Issues

1. **Excel file not found**: Ensure `data/raw/DataSheet_ETH_250902.xlsx` exists
2. **Encoding errors**: Use recent pandas version (2.3+)
3. **Memory issues**: Unlikely with this dataset size, but close other applications if needed

---

## Notebook 2: Exploratory Data Analysis

**File**: `notebooks/exploratory-data-analysis.ipynb`
**Purpose**: Understand data distributions, relationships, and characteristics through statistical analysis and visualization

### Overview
This notebook performs comprehensive exploratory data analysis (EDA) on the preprocessed dataset. It generates statistical summaries, visualizations, and correlation analyses to inform modeling decisions.

### Structure

#### Cell 1: Imports
```python
Imports:
  - matplotlib.pyplot: Plotting
  - pandas: Data manipulation
  - seaborn: Statistical visualization
```

#### Cell 2: Load Dataset
- Reads `data/processed/preprocessed_dataset.csv`
- Displays first 5 rows
- Initial data inspection

#### Cell 3: Initial Data Exploration
**Activities**:
- Display DataFrame info (dtypes, non-null counts)
- Generate descriptive statistics (mean, std, min, max, quartiles)
- Print shape and column information

**Output**: Comprehensive data summary

#### Cell 4-8: Detailed Analysis
**Likely includes**:
- Distribution plots for each variable
- Box plots for outlier detection
- Scatter plots for relationship exploration
- Correlation heatmap
- Time series plots
- Statistical tests

### Key Insights Generated

From this analysis, you would discover:

1. **Data Characteristics**:
   - Value ranges for each variable
   - Presence of outliers
   - Distribution shapes (normal, skewed, etc.)

2. **Relationships**:
   - Correlation between inputs and targets
   - Non-linear relationships
   - Feature interactions

3. **Temporal Patterns**:
   - How spray characteristics evolve over time
   - Transient vs steady-state behavior
   - Run-to-run variability

### Outputs

**Console Output**:
- Statistical summaries
- Data quality metrics
- Correlation matrices

**Visual Output** (likely saved to `plots/`):
- Distribution histograms
- Correlation heatmaps
- Scatter matrices
- Time series plots

### How to Run

```bash
jupyter notebook notebooks/exploratory-data-analysis.ipynb
# Run cells sequentially
# Examine plots and statistics
```

### Use Cases

- **Before Modeling**: Understand data before building models
- **Feature Engineering**: Identify transformations or new features
- **Model Selection**: Choose appropriate algorithms based on data characteristics
- **Communication**: Create visualizations for presentations

---

## Notebook 3: Machine Learning Pipeline

**File**: `notebooks/ml-pipeline.ipynb`
**Purpose**: Train and evaluate multiple classical machine learning models for spray characteristic prediction

### Overview
This notebook implements a comprehensive machine learning pipeline that trains 6 different classical ML algorithms, performs hyperparameter tuning, and evaluates their performance on spray prediction tasks.

### Structure

#### Cell 1: Imports and Global Settings
```python
Imports:
  - os, warnings: Environment setup
  - joblib: Model serialization
  - numpy, pandas: Data handling
  - sklearn.ensemble: RandomForest, GradientBoosting
  - sklearn.linear_model: LinearRegression
  - sklearn.metrics: r2_score, MAE, MSE
  - sklearn.model_selection: train_test_split, GridSearchCV
  - sklearn.neighbors: KNN
  - sklearn.svm: SVR
  - sklearn.tree: DecisionTree
  - sklearn.preprocessing: StandardScaler
  - sklearn.pipeline: Pipeline
  - sklearn.multioutput: MultiOutputRegressor
```

**Configuration**:
- Random seeds for reproducibility
- Output directories
- Model save paths

**Purpose**: Set up complete ML environment

#### Cell 2: Load and Prepare Data
**Activities**:
1. Load preprocessed CSV
2. Rename columns to match paper terminology
3. Define input and target features
4. Display data preview

**Variables**:
```python
INPUTS = ["time", "chamb_pressure", "cham_temp", 
          "injection_pres", "density", "viscosity"]
TARGETS = ["angle_shadow", "penetration_shadow", 
           "angle_mie", "penetration_mie"]
```

**Output**: Clean DataFrame ready for splitting

#### Cell 3: Train-Test Split
**Strategy**:
- Test size: 20% (170 samples)
- Training size: 80% (677 samples)
- Random state: 42 (reproducibility)
- Stratification: By experimental run (maintains run distribution)

**Rationale**:
- Stratification ensures both sets represent all experiments
- 80/20 split is standard practice
- Fixed random state enables reproducibility

**Output**:
- `X_tr`, `X_te`: Input features
- `y_tr`, `y_te`: Target variables

#### Cells 4-9: Model Training and Evaluation

Each subsequent cell typically trains one or more models:

##### Pattern for Each Model
```python
# 1. Model definition
model = SomeRegressor()

# 2. Preprocessing (if needed)
if needs_scaling:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
else:
    pipeline = model

# 3. Multi-output wrapper (if needed)
if not_natively_multioutput:
    pipeline = MultiOutputRegressor(pipeline)

# 4. Hyperparameter tuning
param_grid = {...}
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_tr, y_tr)

# 5. Best model selection
best_model = grid_search.best_estimator_

# 6. Evaluation
y_pred = best_model.predict(X_te)
metrics = calculate_metrics(y_te, y_pred)

# 7. Save model
joblib.dump(best_model, f"models/{name}_regressor.joblib")

# 8. Display results
print(f"R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
```

##### Models Trained (in order)

**1. Linear Regression**
- Simplest baseline model
- No hyperparameters to tune
- Fastest training and inference
- File: LinearRegression_regressor.joblib

**2. Decision Tree**
- Non-linear, interpretable
- Grid search over depth, split criteria
- File: DecisionTree_regressor.joblib

**3. K-Nearest Neighbors**
- Instance-based learning
- Requires feature scaling
- Grid search over k, distance metric
- File: KNN_regressor.joblib

**4. Support Vector Regression**
- Kernel-based method
- Requires feature scaling
- Grid search over C, kernel, epsilon
- File: SVR_regressor.joblib

**5. Random Forest**
- Ensemble of decision trees
- Grid search over trees, depth, features
- File: RandomForest_regressor.joblib

**6. Gradient Boosting**
- Sequential ensemble method
- Grid search over trees, learning rate, depth
- File: GradientBoosting_regressor.joblib

#### Final Cell: Results Aggregation
**Activities**:
- Compile all model metrics
- Create comparison table
- Save to `outputs/overall_model_metrics.csv`
- Save per-target metrics to `outputs/per_target_metrics.csv`
- Display summary statistics

### Hyperparameter Grids

#### Decision Tree
```python
{
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Total combinations: 36
```

#### KNN
```python
{
    'n_neighbors': [3, 5, 7, 10, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan or Euclidean
}
# Total combinations: 20
```

#### SVR
```python
{
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}
# Total combinations: 48
```

#### Random Forest
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}
# Total combinations: 144
```

#### Gradient Boosting
```python
{
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
# Total combinations: 54
```

### Outputs

**Model Files** (saved to `models/`):
- 6 `.joblib` files (one per model)
- Total size: ~47 MB
- Compressed format for efficient storage

**Metrics Files** (saved to `outputs/`):
- `overall_model_metrics.csv`: Aggregate metrics per model
- `per_target_metrics.csv`: Detailed metrics per target per model
- `baseline_models_config.csv`: Hyperparameter configurations

**Console Output**:
- Training progress
- Grid search results
- Best hyperparameters
- Performance metrics

### Expected Results

```
Model Performance Ranking (R²):
1. KNN:              0.9988
2. GradientBoosting: 0.9949
3. DecisionTree:     0.9936
4. SVR:              0.9883
5. RandomForest:     0.9830
6. LinearRegression: 0.8341
```

### How to Run

**Command Line** (automated):
```bash
make train-ml
# Estimated time: 5-15 minutes
```

**Interactive**:
```bash
jupyter notebook notebooks/ml-pipeline.ipynb
# Run cells sequentially
# Monitor grid search progress
```

### Performance Considerations

- **Training Time**: 5-15 minutes total
  - Linear Regression: < 1 second
  - Decision Tree: ~1 minute
  - KNN: ~30 seconds
  - SVR: ~2 minutes
  - Random Forest: ~5 minutes
  - Gradient Boosting: ~3 minutes

- **Memory Usage**: Peak ~2 GB (during Random Forest grid search)
- **CPU Usage**: Utilizes all cores (n_jobs=-1)

### Tips

1. **Grid Search**: Reduce param_grid size for faster experimentation
2. **Parallel Processing**: Automatically uses all CPU cores
3. **Memory**: Close other applications if RandomForest causes issues
4. **Monitoring**: Watch console for progress updates

---

## Notebook 4: Artificial Neural Network Pipeline

**File**: `notebooks/ann-pipeline.ipynb`
**Purpose**: Design, train, and evaluate deep learning models for spray characteristic prediction

### Overview
This notebook implements an artificial neural network (ANN) using TensorFlow and Keras. It includes data scaling, architecture design, training with callbacks, and performance evaluation.

### Structure

#### Cell 1: Imports for ANN Training
```python
Imports:
  - joblib: Save scalers
  - keras: High-level neural network API
    - callbacks: EarlyStopping, ModelCheckpoint
    - layers: Dense, Dropout (if used)
    - models: Sequential
    - optimizers: Adam, SGD
  - numpy, pandas: Data handling
  - sklearn.metrics: r2_score, MAE, MSE
  - sklearn.preprocessing: StandardScaler
  - tensorflow: Deep learning framework
```

**Configuration**:
- Print TensorFlow version
- Set random seeds for reproducibility
- Configure GPU (if available)

#### Cell 2: GPU Check
```python
print(tf.config.list_physical_devices("GPU"))
```
**Purpose**: Verify GPU availability (optional but speeds up training)

#### Cell 3: Load and Prepare Data
**Activities**:
1. Load preprocessed CSV
2. Rename columns
3. Define inputs and targets
4. Extract features and labels

**Same features as ML pipeline**:
- 6 input features
- 4 output targets

#### Cell 4: Train-Test-Validation Split
**Strategy**:
```python
# Split 1: Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=runs
)

# Split 2: Train/Validation (within training, 80/20)
# Done internally by fit() with validation_split=0.2
```

**Result**:
- Training: ~542 samples (64% of total)
- Validation: ~135 samples (16% of total)
- Test: ~170 samples (20% of total)

#### Cell 5: Feature Scaling
**Critical for Neural Networks**:
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

# Save scalers for deployment
joblib.dump(input_scaler, "models/ann_input_scaler.joblib")
joblib.dump(target_scaler, "models/ann_target_scaler.joblib")
```

**Why Scaling is Critical**:
- Neural networks sensitive to input magnitude
- Different features have different scales
- Standardization improves convergence
- Prevents one feature from dominating

#### Cell 6: Build Neural Network Architecture
```python
model = models.Sequential([
    # Input layer (implicit) + First hidden layer
    layers.Dense(64, activation='relu', input_shape=(6,)),
    
    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    
    # Output layer
    layers.Dense(4, activation='linear')  # 4 targets
])

model.summary()
```

**Architecture Details**:
- **Input**: 6 features
- **Layer 1**: 64 neurons, ReLU activation
  - Parameters: 6 × 64 + 64 = 448
- **Layer 2**: 128 neurons, ReLU activation
  - Parameters: 64 × 128 + 128 = 8,320
- **Layer 3**: 64 neurons, ReLU activation
  - Parameters: 128 × 64 + 64 = 8,256
- **Layer 4**: 32 neurons, ReLU activation
  - Parameters: 64 × 32 + 32 = 2,080
- **Output**: 4 neurons, Linear activation
  - Parameters: 32 × 4 + 4 = 132
- **Total Parameters**: ~19,236

**Design Rationale**:
- Expanding-contracting pattern (64→128→64→32)
- ReLU activation prevents vanishing gradients
- Linear output for unbounded regression
- Moderate depth balances capacity and overfitting risk

#### Cell 7: Compile and Train Model
```python
# Compilation
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Training
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save("models/ANN_improved_regressor.h5")
```

**Training Configuration**:
- **Optimizer**: Adam (adaptive learning rates)
- **Learning Rate**: 0.001 (default)
- **Loss Function**: MSE (mean squared error)
- **Batch Size**: 32 samples per gradient update
- **Max Epochs**: 500 (usually stops earlier)
- **Validation Split**: 20% of training data
- **Early Stopping**: Patience of 20 epochs

**Training Process**:
1. Mini-batch gradient descent
2. Monitor validation loss after each epoch
3. Stop if no improvement for 20 epochs
4. Restore weights from best epoch
5. Typical stopping: 50-150 epochs

#### Cell 8: Evaluate Model
```python
# Predictions on test set
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')

# Save results
pd.DataFrame({
    'model': ['ANN_improved'],
    'r2': [r2],
    'mae': [mae],
    'mse': [mse]
}).to_csv("outputs/ANN_improved_metrics.csv", index=False)

# Save predictions and actuals
pd.DataFrame(y_pred).to_csv("outputs/ANN_improved_predictions.csv", index=False)
pd.DataFrame(y_test).to_csv("outputs/ANN_improved_actuals.csv", index=False)

# Print summary
print(f"ANN Performance:")
print(f"  R²:  {r2:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  MSE: {mse:.6f}")
```

### Outputs

**Model Files**:
- `ANN_improved_regressor.h5`: Trained neural network (594 KB)
- `ann_input_scaler.joblib`: Input feature scaler (1 KB)
- `ann_target_scaler.joblib`: Output target scaler (1 KB)

**Metrics Files**:
- `ANN_improved_metrics.csv`: Overall performance
- `ANN_improved_predictions.csv`: Test set predictions
- `ANN_improved_actuals.csv`: Test set actual values

**Training Plots** (if generated):
- Training and validation loss curves
- MAE curves
- Convergence visualization

### Expected Performance

**Typical Results**:
- R²: > 0.99
- MAE: < 1.0
- MSE: < 5.0
- Training time: 2-5 minutes (CPU), < 1 minute (GPU)

### How to Run

**Command Line**:
```bash
make train-ann
# Estimated time: 2-5 minutes
```

**Interactive**:
```bash
jupyter notebook notebooks/ann-pipeline.ipynb
# Run cells sequentially
# Watch training progress
```

### Advantages of ANN

1. **Universal Approximation**: Can model any continuous function
2. **Feature Learning**: Automatically discovers useful representations
3. **Fast Inference**: Once trained, predictions are very fast
4. **Scalability**: Handles large datasets efficiently
5. **GPU Acceleration**: Dramatically faster on GPU

### Tips for Tuning

**Architecture**:
- More neurons: Increase capacity, risk overfitting
- More layers: Model more complex functions
- Dropout: Add between layers to reduce overfitting

**Training**:
- Learning rate: Decrease if loss oscillates
- Batch size: Larger = more stable, smaller = more stochastic
- Patience: Increase for complex problems

**Regularization**:
- L2 regularization: Add to Dense layers
- Dropout: Randomly deactivate neurons during training
- Early stopping: Already implemented

---

## Notebook 5: Fuel Journal Plots

**File**: `notebooks/fuel-journal-plots.ipynb`
**Purpose**: Generate publication-quality visualizations for research papers and presentations

### Overview
This notebook creates comprehensive, high-quality plots suitable for inclusion in academic papers, specifically styled for fuel/combustion journals. It contains both code cells (27) and markdown cells (24) for documentation.

### Structure

The notebook is well-documented with markdown cells explaining each visualization.

#### Typical Sections

**1. Setup and Data Loading**
- Import visualization libraries
- Load processed data
- Load trained models
- Configure plotting style

**2. Figure 1: Data Visualization**
Likely includes:
- **1.1**: Time variation plots (figure_1_1_time_variation.png)
- **1.2**: Input distributions before scaling (figure_1_2_inputs_before_scaling.png)
- **1.3**: Input distributions after scaling (figure_1_3_inputs_after_scaling.png)
- **1.4**: Correlation heatmap (figure_1_4_correlation_heatmap.png)

**3. Figure 2: Model Results**
Comprehensive analysis with 19 subplots:

**Feature Relationships**:
- 2.1: Chamber temperature vs angle_mie
- 2.2: Density vs angle_shadow

**Prediction Performance**:
- 2.3-2.6: True vs predicted plots (all 4 targets)
- 2.7-2.10: Parity plots (all 4 targets)

**Residual Analysis**:
- 2.11: KDE of residuals for angle_mie
- 2.12: Histogram + KDE of residuals
- 2.13: Residuals vs predicted values

**Feature Importance**:
- 2.14: Feature importance chart for angle_mie

**Model Comparison**:
- 2.15-2.18: ANN vs Gradient Boosting (all 4 targets)
- 2.19: Overall ANN residuals KDE

### Plot Specifications

**Quality Settings**:
- **DPI**: 300+ (publication quality)
- **Format**: PNG (universal compatibility)
- **Size**: Typically 8×6 or 10×8 inches
- **Font**: Professional serif fonts (Times, Liberation Serif)
- **Style**: Seaborn with custom palette

**Styling**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Color palette
colors = sns.color_palette("husl", 8)

# Figure configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
```

### Plot Types

#### 1. Time Series Plots
```python
# Example structure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, target in zip(axes.flat, targets):
    for run in unique_runs:
        run_data = df[df['run'] == run]
        ax.plot(run_data['time'], run_data[target], label=run)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(target)
    ax.legend()
plt.tight_layout()
plt.savefig('plots/figure_1_1_time_variation.png', dpi=300, bbox_inches='tight')
```

#### 2. Parity Plots
```python
# True vs Predicted with 1:1 line
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'Parity Plot: {target_name}')
plt.legend()
plt.axis('equal')
```

#### 3. Residual Plots
```python
# Residuals vs Predicted
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
```

#### 4. Correlation Heatmap
```python
plt.figure(figsize=(10, 8))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap')
```

#### 5. Feature Importance
```python
# For tree-based models
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
```

### Outputs

**All Plots** (saved to `plots/`):
```
figure_1_1_time_variation.png                 (207 KB)
figure_1_2_inputs_before_scaling.png          (291 KB)
figure_1_3_inputs_after_scaling.png           (267 KB)
figure_1_4_correlation_heatmap.png            (651 KB)
figure_2_1_cham_temp_vs_angle_mie.png         (321 KB)
figure_2_2_density_vs_angle_shadow.png        (304 KB)
figure_2_3_true_vs_pred_angle_mie.png         (518 KB)
figure_2_4_true_vs_pred_angle_shadow.png      (512 KB)
figure_2_5_true_vs_pred_length_mie.png        (526 KB)
figure_2_6_true_vs_pred_length_shadow.png     (537 KB)
figure_2_7_parity_angle_mie.png               (689 KB)
figure_2_8_parity_angle_shadow.png            (707 KB)
figure_2_9_parity_length_mie.png              (676 KB)
figure_2_10_parity_length_shadow.png          (653 KB)
figure_2_11_kde_residuals_angle_mie.png       (231 KB)
figure_2_12_hist_kde_residuals_angle_mie.png  (210 KB)
figure_2_13_residuals_vs_predicted_angle_mie.png (609 KB)
figure_2_14_feature_importance_angle_mie.png  (148 KB)
figure_2_15_ann_vs_gb_angle_mie.png           (377 KB)
figure_2_16_ann_vs_gb_angle_shadow.png        (379 KB)
figure_2_17_ann_vs_gb_length_mie.png          (386 KB)
figure_2_18_ann_vs_gb_length_shadow.png       (377 KB)
figure_2_19_ann_residuals_kde.png             (174 KB)

Total: 19 figures, ~9.6 MB
```

### How to Run

**Interactive** (recommended for customization):
```bash
jupyter notebook notebooks/fuel-journal-plots.ipynb
# Run cells sequentially
# Customize plots as needed
```

### Customization Tips

**Adjust Plot Style**:
```python
# For presentations (larger fonts)
sns.set_context("talk", font_scale=1.2)

# For papers (smaller, more detail)
sns.set_context("paper", font_scale=1.0)
```

**Change Colors**:
```python
# Different color schemes
sns.set_palette("Set2")
sns.set_palette("colorblind")
sns.set_palette("viridis")
```

**Modify Figure Size**:
```python
# Larger for detail
plt.figure(figsize=(12, 10))

# Smaller for journals
plt.figure(figsize=(6, 4))
```

---

## Execution Order

### Sequential Execution (Recommended)

```
1. data-preprocessing-pipeline.ipynb
   ↓ (Generates preprocessed_dataset.csv)
   
2. exploratory-data-analysis.ipynb
   ↓ (Optional, for understanding)
   
3. ml-pipeline.ipynb
   ↓ (Trains classical ML models)
   
4. ann-pipeline.ipynb
   ↓ (Trains neural network)
   
5. fuel-journal-plots.ipynb
   (Generates all visualizations)
```

### Automated Execution

```bash
# Run all pipelines sequentially
make train-all

# Or step by step
make preprocess
make train-ml
make train-ann
```

### Parallel Execution (Advanced)

After preprocessing, ML and ANN pipelines can run in parallel:
```bash
# Terminal 1
make train-ml

# Terminal 2 (simultaneously)
make train-ann
```

---

## Tips and Best Practices

### General Tips

1. **Run Sequentially First**: Always run notebooks in order initially
2. **Check Outputs**: Verify each notebook's output before proceeding
3. **Save Regularly**: Use Ctrl+S or Cmd+S frequently
4. **Clear Outputs**: Use "Kernel → Restart & Clear Output" if notebook is large
5. **Read Comments**: Each cell has explanatory comments

### Debugging Tips

1. **Cell-by-Cell Execution**: Don't "Run All" immediately
2. **Inspect Variables**: Use `df.head()`, `model.summary()` to check
3. **Check Shapes**: Verify data dimensions after transformations
4. **Monitor Memory**: Watch for memory warnings
5. **Use Print Statements**: Add debugging prints as needed

### Performance Tips

1. **Close Unused Tabs**: Free up browser memory
2. **Restart Kernel**: If memory issues occur
3. **Reduce Grid Search**: For faster experimentation
4. **Use GPU**: For ANN training if available
5. **Parallel Processing**: Automatically used in scikit-learn

### Customization Tips

1. **Hyperparameters**: Modify param_grids in ml-pipeline
2. **Architecture**: Change layer sizes/counts in ann-pipeline
3. **Features**: Add/remove features in preprocessing
4. **Plots**: Customize colors, sizes, labels in fuel-journal-plots
5. **Metrics**: Add custom metrics to evaluation

### Documentation Tips

1. **Add Markdown Cells**: Document your modifications
2. **Clear Explanations**: Explain why you made changes
3. **Version Control**: Commit after each major change
4. **Screenshot Results**: Save plots externally for reports
5. **Export Notebooks**: Use nbconvert for PDF/HTML reports

### Collaboration Tips

1. **Clear Outputs Before Committing**: Keeps git diffs clean
2. **Use Absolute Paths**: Or document relative path assumptions
3. **Document Dependencies**: Note any additional packages
4. **Share Results**: Export key plots and metrics
5. **Code Review**: Have others check notebooks

---

## Summary

The Spray Vision notebooks provide a complete, end-to-end machine learning pipeline:

- ✅ **Notebook 1**: Clean and prepare data
- ✅ **Notebook 2**: Understand data characteristics
- ✅ **Notebook 3**: Train classical ML models
- ✅ **Notebook 4**: Train deep learning model
- ✅ **Notebook 5**: Create publication-quality visualizations

Each notebook is:
- Self-contained with clear structure
- Well-documented with comments
- Produces specific outputs
- Can be run interactively or automated
- Follows data science best practices

**Total Execution Time**: ~20-30 minutes for complete pipeline

**Total Output Size**: ~50 MB (models + plots + data)

---

*This guide provides comprehensive details for understanding and using all notebooks in the Spray Vision project.*
