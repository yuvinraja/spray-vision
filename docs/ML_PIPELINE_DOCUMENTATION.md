# Machine Learning Pipeline Documentation
## Spray Vision: ML-Based Spray Characteristics Prediction

**Project:** Spray Vision  
**Author:** Yuvin Raja  
**Version:** 0.1.0  
**Last Updated:** October 16, 2025  
**Python Version:** 3.10+

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Dataset Characteristics](#dataset-characteristics)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Classical Machine Learning Models](#classical-machine-learning-models)
8. [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
9. [Model Performance Comparison](#model-performance-comparison)
10. [Hyperparameter Optimization](#hyperparameter-optimization)
11. [Model Persistence and Deployment](#model-persistence-and-deployment)
12. [Validation Strategy](#validation-strategy)
13. [Feature Importance Analysis](#feature-importance-analysis)
14. [Dependencies and Environment](#dependencies-and-environment)

---

## Executive Summary

The Spray Vision project implements a comprehensive machine learning pipeline for predicting spray characteristics based on experimental conditions. The pipeline processes raw experimental data from ETH spray experiments and trains multiple machine learning models to predict four key spray characteristics:

- **Spray cone angle (Mie scattering imaging)** - `angle_mie`
- **Spray penetration length (Mie scattering imaging)** - `length_mie`
- **Spray cone angle (Shadow imaging)** - `angle_shadow`
- **Spray penetration length (Shadow imaging)** - `length_shadow`

The project achieves exceptional performance across all models, with R² scores exceeding 0.90 for baseline models and reaching up to 0.9988 for the K-Nearest Neighbors model.

---

## Data Pipeline Architecture

### Overview

The data pipeline follows a three-stage architecture:

```
Raw Excel Data → Preprocessing → Feature Engineering → Model Training → Evaluation
```

### Pipeline Stages

1. **Data Extraction**: Multi-sheet Excel workbook processing
2. **Data Transformation**: Long-format conversion and temporal alignment
3. **Data Cleaning**: Missing value imputation and outlier handling
4. **Feature Engineering**: Harmonization and standardization
5. **Model Training**: Multiple algorithm training with hyperparameter tuning
6. **Model Evaluation**: Comprehensive performance metrics

---

## Data Preprocessing Pipeline

### 1. Input Data Source

**File:** `data/raw/DataSheet_ETH_250902.xlsx`

The raw data is stored in an Excel workbook with multiple sheets:
- **Exp. Conditions in Time**: Experimental input variables
- **Spray Angle (Shadow)**: Shadow imaging angle measurements
- **Spray Penetration (Shadow)**: Shadow imaging penetration measurements
- **Spray Angle (Mie)**: Mie scattering angle measurements
- **Spray Penetration (Mie)**: Mie scattering penetration measurements

### 2. Data Structure

Each sheet uses a **two-row header format**:
- **Top row**: Variable name/block identifier
- **Second row**: Experimental run identifiers (ETH-01, ETH-02, etc.)

### 3. Preprocessing Steps

#### Step 3.1: Multi-Header Reading

**Function:** `read_multiheader(sheet_name)`

```python
def read_multiheader(sheet_name):
    """
    Read a sheet that uses a two-row header (top: variable block,
    second: ETH-xx run identifiers), returning a DataFrame with a
    MultiIndex for columns.
    """
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(str(t[0]).strip(), (str(t[1]).strip() if not pd.isna(t[1]) else ""))
         for t in df.columns],
        names=["var", "run"],
    )
    return df
```

**Purpose:** Converts Excel's hierarchical column structure into a pandas MultiIndex DataFrame.

#### Step 3.2: Time Series Extraction

**Function:** `get_time_series(df_multi)`

```python
def get_time_series(df_multi):
    """Extract the Time(ms) series from a two-level column DataFrame."""
    time_cols = [c for c in df_multi.columns if c[0] == "Time(ms)"]
    assert len(time_cols) == 1, "Expected exactly one Time(ms) column"
    return df_multi[time_cols[0]].astype(float).rename("Time_ms")
```

**Purpose:** Extracts and normalizes the temporal dimension across all measurements.

#### Step 3.3: Data Reshaping (Wide to Long Format)

**Function:** `stack_block(df_multi, top_name, value_name)`

```python
def stack_block(df_multi, top_name, value_name):
    """
    For a given top-level column block (e.g., 'Chamber pressure (bar)'),
    produce a long DataFrame with ['run', 'Time_ms', value_name].
    """
    time = get_time_series(df_multi)
    block_cols = [c for c in df_multi.columns if c[0] == top_name and c[1] != ""]
    sub = df_multi[block_cols].copy()
    sub.columns = [c[1] for c in block_cols]
    sub.insert(0, "Time_ms", time.values)
    long = sub.melt(id_vars="Time_ms", var_name="run", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long["Time_ms"] = pd.to_numeric(long["Time_ms"], errors="coerce")
    return long
```

**Purpose:** Converts wide-format data (one column per run) to long format (one row per run-time combination).

**Applied to:**
- Chamber pressure (bar) → `Pc_bar`
- Chamber temperature (K) → `Tc_K`
- Injection pressure (bar) → `Pinj_bar`
- Density (kg/m³) → `rho_kgm3`
- Viscosity (Pa·s) → `mu_Pas`
- Smoothed angle (deg) → `angle_shadow_deg`, `angle_mie_deg`
- Penetration (L/D) → `len_shadow_L_D`, `len_mie_L_D`

#### Step 3.4: Data Merging

All individual data blocks are merged using **outer joins** on `['run', 'Time_ms']`:

```python
inputs = reduce(
    lambda left, right: pd.merge(left, right, on=["run", "Time_ms"], how="outer"),
    [inp_pc, inp_tc, inp_pinj, inp_rho, inp_mu],
)

merged = inputs.merge(tgt_ang_sh, on=["run", "Time_ms"], how="outer")
merged = merged.merge(tgt_len_sh, on=["run", "Time_ms"], how="outer")
merged = merged.merge(tgt_ang_mie, on=["run", "Time_ms"], how="outer")
merged = merged.merge(tgt_len_mie, on=["run", "Time_ms"], how="outer")
```

**Rationale:** Outer joins preserve all time points from all sources, ensuring no data loss.

#### Step 3.5: Missing Value Imputation

**Function:** `groupwise_ffill_bfill(df, group_cols, order_cols, cols_to_fill=None)`

```python
def groupwise_ffill_bfill(df, group_cols, order_cols, cols_to_fill=None):
    """
    Within each group, sort and apply ffill followed by bfill for selected columns.
    """
    df = df.sort_values(group_cols + order_cols).copy()
    if cols_to_fill is None:
        keyset = set(group_cols + order_cols)
        cols_to_fill = [
            c for c in df.columns 
            if c not in keyset and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _fill(g):
        g[cols_to_fill] = g[cols_to_fill].ffill().bfill()
        return g

    return df.groupby(group_cols, as_index=False, group_keys=False).apply(_fill)
```

**Strategy:**
1. **Group by run**: Each experimental run is processed independently
2. **Sort by time**: Ensure temporal ordering
3. **Forward fill**: Propagate last valid observation forward
4. **Backward fill**: Fill remaining gaps from future observations

**Rationale:** Temporal continuity within experiments makes forward/backward filling appropriate for missing sensor readings.

#### Step 3.6: Data Validation and Export

```python
# Integrity checks
assert filled[["run", "Time_ms"]].drop_duplicates().shape[0] == filled.shape[0], (
    "Duplicate (run, Time_ms) rows detected"
)

expected_cols = {
    "run", "Time_ms", "Pc_bar", "Tc_K", "Pinj_bar", "rho_kgm3", "mu_Pas",
    "angle_shadow_deg", "len_shadow_L_D", "angle_mie_deg", "len_mie_L_D",
}
assert expected_cols.issubset(set(filled.columns)), "Missing expected columns"

# Export
filled.to_csv("data/processed/eth_spray_merged_preprocessed.csv", index=False)
```

**Output:** `data/processed/eth_spray_merged_preprocessed.csv`

**Final Dataset Characteristics:**
- **Shape:** 847 rows × 11 columns
- **Runs:** 7 experimental runs (ETH-01 through ETH-06.1)
- **Time points per run:** ~121 measurements
- **No missing values** after imputation

---

## Feature Engineering

### 1. Column Name Harmonization

Raw column names are standardized for consistency:

```python
rename_map = {
    "Time_ms": "time",
    "Pc_bar": "chamb_pressure",
    "Tc_K": "cham_temp",
    "Pinj_bar": "injection_pres",
    "rho_kgm3": "density",
    "mu_Pas": "viscosity",
    "angle_shadow_deg": "angle_shadow",
    "len_shadow_L_D": "length_shadow",
    "angle_mie_deg": "angle_mie",
    "len_mie_L_D": "length_mie",
}
df = raw.rename(columns=rename_map)
```

### 2. Feature Categorization

#### Input Features (6 variables)

| Feature | Original Name | Description | Unit | Type |
|---------|--------------|-------------|------|------|
| `time` | Time_ms | Elapsed time since injection start | ms | Continuous |
| `chamb_pressure` | Pc_bar | Combustion chamber ambient pressure | bar | Continuous |
| `cham_temp` | Tc_K | Combustion chamber ambient temperature | K | Continuous |
| `injection_pres` | Pinj_bar | Injection pressure at nozzle inlet | bar | Continuous |
| `density` | rho_kgm3 | Fluid density | kg/m³ | Continuous |
| `viscosity` | mu_Pas | Fluid dynamic viscosity | Pa·s | Continuous |

#### Target Features (4 variables)

| Target | Original Name | Description | Unit | Type |
|--------|--------------|-------------|------|------|
| `angle_mie` | angle_mie_deg | Spray cone angle (Mie scattering) | degrees | Continuous |
| `length_mie` | len_mie_L_D | Non-dimensional spray length (Mie scattering) | L/D | Continuous |
| `angle_shadow` | angle_shadow_deg | Spray cone angle (Shadow imaging) | degrees | Continuous |
| `length_shadow` | len_shadow_L_D | Non-dimensional spray length (Shadow imaging) | L/D | Continuous |

### 3. Feature Scaling

Different scaling strategies are applied depending on the model:

#### For Classical ML Models (in pipeline):
```python
StandardScaler()  # Applied within sklearn Pipeline
```

#### For Neural Networks (separate):
```python
# Input features
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

# Output targets
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)
```

**Saved artifacts:**
- `models/ann_input_scaler.joblib`
- `models/ann_target_scaler.joblib`

### 4. No Explicit Feature Engineering

**Important Note:** The pipeline does **NOT** create derived features such as:
- Polynomial features
- Interaction terms
- Dimensionality reduction (PCA)
- Log transformations

**Rationale:** The raw features provide sufficient predictive power, and tree-based models can capture non-linear relationships automatically.

---

## Dataset Characteristics

### Statistical Summary

Based on the preprocessed dataset (`data/processed/preprocessed_dataset.csv`):

| Variable | Role | Mean | Std | Min | 25% | 50% | 75% | Max |
|----------|------|------|-----|-----|-----|-----|-----|-----|
| time | Input | Varies by run | - | 0 | - | - | - | ~12 ms |
| chamb_pressure | Input | ~30-100 bar | - | - | - | - | - | - |
| cham_temp | Input | ~600-900 K | - | - | - | - | - | - |
| injection_pres | Input | ~200-400 bar | - | - | - | - | - | - |
| density | Input | ~700-850 kg/m³ | - | - | - | - | - | - |
| viscosity | Input | ~0.001-0.003 Pa·s | - | - | - | - | - | - |
| angle_mie | Output | ~10-15° | - | - | - | - | - | - |
| length_mie | Output | ~50-250 L/D | - | - | - | - | - | - |
| angle_shadow | Output | ~8-12° | - | - | - | - | - | - |
| length_shadow | Output | ~60-280 L/D | - | - | - | - | - | - |

### Data Distribution

- **Total samples:** 847 observations
- **Experimental runs:** 7 (ETH-01, ETH-02, ETH-03, ETH-04, ETH-05, ETH-06, ETH-06.1)
- **Time points per run:** Approximately 121 measurements
- **Temporal resolution:** Variable (non-uniform time sampling)

### Data Quality

- **Missing values:** 0 (after preprocessing)
- **Duplicates:** None
- **Outliers:** Retained (represent valid experimental conditions)

---

## Model Training Pipeline

### 1. Train-Test Split

```python
X, y = df[INPUTS], df[TARGETS]
runs = df["run"]  # stratification label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=runs
)
```

**Configuration:**
- **Test size:** 20% (169 samples)
- **Train size:** 80% (678 samples)
- **Stratification:** By experimental run (ensures proportional representation)
- **Random seed:** 42 (reproducibility)

**Rationale:** Stratification by run ensures test set contains data from all experimental conditions.

### 2. Multi-Output Regression Strategy

All models are wrapped in `MultiOutputRegressor` to handle the 4 target variables:

```python
from sklearn.multioutput import MultiOutputRegressor

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", MultiOutputRegressor(base_estimator))
])
```

**Advantages:**
- Trains independent model for each target
- Allows target-specific performance analysis
- Simplifies model architecture

---

## Classical Machine Learning Models

### Overview

Six classical machine learning algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting Regressor

### Model Training Function

```python
def train_model(name, cfg, X_tr, y_tr, X_te, y_te):
    """Return best estimator + dicts of overall & per-target metrics."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MultiOutputRegressor(cfg["estimator"])),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe, 
        param_grid=cfg["param_grid"], 
        cv=cv, 
        scoring="r2", 
        n_jobs=-1, 
        refit=True, 
        verbose=0
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_

    y_pred = best.predict(X_te)
    
    # Per-target metrics
    r2 = [r2_score(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]
    mae = [mean_absolute_error(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]
    mse = [mean_squared_error(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]

    overall = {
        "model": name,
        "overall_r2": np.mean(r2),
        "overall_mae": np.mean(mae),
        "overall_mse": np.mean(mse),
    }
    
    per_target = [
        {"model": name, "target": col, "r2": r2[i], "mae": mae[i], "mse": mse[i]}
        for i, col in enumerate(y_te.columns)
    ]

    # Save model
    joblib.dump(best, f"models/{name}_regressor.joblib")

    return best, overall, per_target
```

### Cross-Validation Strategy

- **Method:** 5-Fold Cross-Validation
- **Type:** KFold (not StratifiedKFold, since targets are continuous)
- **Shuffle:** True (with random_state=42)
- **Scoring metric:** R² (coefficient of determination)

---

## Classical Machine Learning Models - Detailed Specifications

### 1. Linear Regression

**Algorithm:** Ordinary Least Squares (OLS)

**Scikit-learn class:** `LinearRegression()`

**Hyperparameters:**
```python
{
    # No hyperparameters (deterministic)
}
```

**Model equation:** $y = X\beta + \epsilon$

**Characteristics:**
- **Assumptions:** Linear relationship, independence, homoscedasticity
- **Advantages:** Fast, interpretable, no hyperparameter tuning
- **Limitations:** Cannot capture non-linear patterns

**Saved model:** `models/LinearRegression_regressor.joblib`

**Performance:**
- Overall R²: 0.8341
- Overall MAE: 4.7795
- Overall MSE: 78.2915

---

### 2. Decision Tree Regressor

**Algorithm:** CART (Classification and Regression Trees)

**Scikit-learn class:** `DecisionTreeRegressor(random_state=42)`

**Hyperparameters (Grid Search):**
```python
{
    "reg__estimator__max_depth": [None, 5, 10, 20],
    "reg__estimator__min_samples_leaf": [1, 2, 5],
}
```

**Search space:** 4 × 3 = 12 combinations

**Best hyperparameters** (selected by GridSearchCV):
- `max_depth`: Typically None or 20
- `min_samples_leaf`: Typically 1 or 2

**Characteristics:**
- **Splitting criterion:** MSE minimization
- **Advantages:** Captures non-linearity, interpretable
- **Limitations:** Prone to overfitting without pruning

**Saved model:** `models/DecisionTree_regressor.joblib`

**Performance:**
- Overall R²: 0.9936
- Overall MAE: 1.1068
- Overall MSE: 4.1551

---

### 3. Random Forest Regressor

**Algorithm:** Ensemble of decision trees with bootstrap aggregating

**Scikit-learn class:** `RandomForestRegressor(random_state=42)`

**Hyperparameters (Grid Search):**
```python
{
    "reg__estimator__n_estimators": [200, 400],
    "reg__estimator__max_depth": [None, 10, 20],
    "reg__estimator__max_features": ["sqrt", "auto"],
}
```

**Search space:** 2 × 3 × 2 = 12 combinations

**Best hyperparameters** (selected by GridSearchCV):
- `n_estimators`: Typically 400
- `max_depth`: Typically None
- `max_features`: Typically "sqrt" (√6 ≈ 2-3 features per split)

**Characteristics:**
- **Bagging:** Bootstrap sampling for each tree
- **Feature randomness:** Subset of features considered at each split
- **Advantages:** Reduces overfitting, robust, handles non-linearity
- **Limitations:** Less interpretable, computationally expensive

**Saved model:** `models/RandomForest_regressor.joblib`

**Performance:**
- Overall R²: 0.9830
- Overall MAE: 2.2071
- Overall MSE: 20.7300

---

### 4. Support Vector Regression (SVR)

**Algorithm:** Support Vector Machine with ε-insensitive loss

**Scikit-learn class:** `SVR()`

**Hyperparameters (Grid Search):**
```python
{
    "reg__estimator__C": [10, 100],
    "reg__estimator__gamma": ["scale"],
    "reg__estimator__epsilon": [0.01, 0.1],
    "reg__estimator__kernel": ["rbf"],
}
```

**Search space:** 2 × 1 × 2 × 1 = 4 combinations

**Best hyperparameters** (selected by GridSearchCV):
- `C`: Regularization parameter (typically 100)
- `gamma`: "scale" = 1 / (n_features × X.var())
- `epsilon`: ε-tube width (typically 0.01)
- `kernel`: Radial Basis Function (RBF)

**Kernel function:** $K(x, x') = \exp(-\gamma ||x - x'||^2)$

**Characteristics:**
- **Optimization:** Quadratic programming (QP)
- **Advantages:** Effective in high-dimensional spaces
- **Limitations:** Sensitive to scaling, computationally intensive

**Saved model:** `models/SVR_regressor.joblib`

**Performance:**
- Overall R²: 0.9883
- Overall MAE: 1.2921
- Overall MSE: 17.1278

---

### 5. K-Nearest Neighbors (KNN)

**Algorithm:** Instance-based learning with distance weighting

**Scikit-learn class:** `KNeighborsRegressor()`

**Hyperparameters (Grid Search):**
```python
{
    "reg__estimator__n_neighbors": [5, 7, 9],
    "reg__estimator__weights": ["uniform", "distance"],
}
```

**Search space:** 3 × 2 = 6 combinations

**Best hyperparameters** (selected by GridSearchCV):
- `n_neighbors`: Typically 5 or 7
- `weights`: "distance" (inverse distance weighting)

**Prediction function:** $\hat{y} = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i}$

where $w_i = \frac{1}{d(x, x_i)}$ for distance weighting.

**Characteristics:**
- **Distance metric:** Euclidean (L2 norm)
- **Advantages:** Simple, non-parametric, no training phase
- **Limitations:** Sensitive to feature scaling, memory-intensive

**Saved model:** `models/KNN_regressor.joblib`

**Performance:**
- Overall R²: 0.9988
- Overall MAE: 0.7406
- Overall MSE: 2.1986

**Top performer among classical models!**

---

### 6. Gradient Boosting Regressor

**Algorithm:** Sequential ensemble of decision trees with gradient descent optimization

**Scikit-learn class:** `GradientBoostingRegressor(random_state=42)`

**Hyperparameters (Grid Search):**
```python
{
    "reg__estimator__n_estimators": [300, 500],
    "reg__estimator__learning_rate": [0.05, 0.1],
    "reg__estimator__max_depth": [3, 4],
}
```

**Search space:** 2 × 2 × 2 = 8 combinations

**Best hyperparameters** (selected by GridSearchCV):
- `n_estimators`: Typically 500
- `learning_rate`: Typically 0.05 (slower, more accurate)
- `max_depth`: Typically 4

**Boosting equation:** $F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$

where:
- $F_m(x)$: ensemble after m iterations
- $\nu$: learning rate
- $h_m(x)$: weak learner (decision tree)

**Characteristics:**
- **Loss function:** Mean Squared Error (MSE)
- **Optimization:** Gradient descent on residuals
- **Advantages:** High accuracy, handles non-linearity
- **Limitations:** Prone to overfitting, requires tuning

**Saved model:** `models/GradientBoosting_regressor.joblib`

**Performance:**
- Overall R²: 0.9949
- Overall MAE: 0.4812
- Overall MSE: 0.7828

**Second-best overall performance!**

---

## Artificial Neural Network (ANN)

### Architecture Overview

**Framework:** TensorFlow 2.20.0 with Keras 3.11.3

**Model Type:** Deep Feedforward Neural Network (Multi-Layer Perceptron)

**Purpose:** Multi-output regression (4 simultaneous targets)

### Network Architecture

```python
def build_deep_ann(input_dim, output_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        
        # Layer 1: Wide dense layer
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Layer 2: Medium dense layer
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Layer 3: Narrow dense layer
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(output_dim, activation="linear"),
    ])
    return model
```

### Layer-by-Layer Breakdown

#### 1. Input Layer
- **Input shape:** (6,) — six input features
- **No activation:** Direct pass-through

#### 2. Hidden Layer 1
- **Neurons:** 256
- **Activation:** ReLU (Rectified Linear Unit)
  - $f(x) = \max(0, x)$
- **BatchNormalization:** Normalizes activations (mean=0, std=1)
  - Reduces internal covariate shift
  - Accelerates training
- **Dropout:** 0.4 (40% of neurons dropped during training)
  - Prevents overfitting
  - Provides regularization

**Parameter count:** 6×256 + 256 = 1,792 parameters

#### 3. Hidden Layer 2
- **Neurons:** 128
- **Activation:** ReLU
- **BatchNormalization:** Applied
- **Dropout:** 0.3 (30% dropout rate)

**Parameter count:** 256×128 + 128 = 32,896 parameters

#### 4. Hidden Layer 3
- **Neurons:** 64
- **Activation:** ReLU
- **BatchNormalization:** Applied
- **Dropout:** 0.2 (20% dropout rate)

**Parameter count:** 128×64 + 64 = 8,256 parameters

#### 5. Output Layer
- **Neurons:** 4 (one per target variable)
- **Activation:** Linear (no activation)
  - $f(x) = x$
  - Suitable for continuous regression

**Parameter count:** 64×4 + 4 = 260 parameters

### Total Parameters

**Trainable parameters:** ~43,000 (approximate, including BatchNorm parameters)

### Compilation Configuration

```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

**Optimizer:** Adam (Adaptive Moment Estimation)
- **Learning rate:** Default (0.001)
- **β₁:** 0.9 (exponential decay rate for 1st moment)
- **β₂:** 0.999 (exponential decay rate for 2nd moment)
- **ε:** 1e-7 (numerical stability constant)

**Loss function:** Mean Squared Error (MSE)
- $\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$

**Metrics:** Mean Absolute Error (MAE)
- $\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$

### Training Configuration

```python
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=30, 
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=10
)

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)
```

### Training Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| **Epochs** | 500 | Maximum training iterations |
| **Batch size** | 32 | Samples per gradient update |
| **Validation split** | 0.2 (20%) | Held-out data for monitoring |
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Initial learning rate** | 0.001 | Default Adam learning rate |

### Regularization Techniques

#### 1. Early Stopping
- **Monitor:** Validation loss
- **Patience:** 30 epochs
- **Restore best weights:** Yes
- **Purpose:** Prevent overfitting by stopping when validation loss stops improving

#### 2. Learning Rate Reduction
- **Monitor:** Validation loss
- **Factor:** 0.5 (halve learning rate)
- **Patience:** 10 epochs
- **Purpose:** Fine-tune learning as training plateaus

#### 3. Dropout
- **Layer 1:** 40% dropout
- **Layer 2:** 30% dropout
- **Layer 3:** 20% dropout
- **Purpose:** Prevent co-adaptation of neurons

#### 4. Batch Normalization
- **Applied after:** Each dense layer
- **Purpose:** Stabilize and accelerate training

### Data Preprocessing for ANN

Unlike classical ML models, the ANN requires **separate scaling** for inputs and outputs:

```python
# Input features
input_scaler = StandardScaler()
X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

# Output targets
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)
```

**Critical:** Both input and target variables are scaled to zero mean and unit variance.

### Prediction and Inverse Scaling

```python
# Predict on scaled test data
y_pred_scaled = model.predict(X_test_scaled)

# Transform back to original scale
y_pred = target_scaler.inverse_transform(y_pred_scaled)
```

### Model Persistence

```python
# Save Keras model
model.save("models/ANN_improved_regressor.h5")

# Save scalers
joblib.dump(input_scaler, "models/ann_input_scaler.joblib")
joblib.dump(target_scaler, "models/ann_target_scaler.joblib")
```

**Saved artifacts:**
1. `models/ANN_improved_regressor.h5` — Keras model (HDF5 format)
2. `models/ann_input_scaler.joblib` — Input StandardScaler
3. `models/ann_target_scaler.joblib` — Target StandardScaler

### ANN Performance

**Overall metrics:**
- **R²:** 0.9910 (mean across 4 targets)
- **MAE:** 1.9456 (mean across 4 targets)
- **MSE:** 22.9918 (mean across 4 targets)

**Per-target performance:**

| Target | R² | MAE | MSE |
|--------|-----|-----|-----|
| angle_mie | 0.9890 | 0.0972 | 0.0171 |
| length_mie | 0.9909 | 3.7677 | 24.7810 |
| angle_shadow | 0.9828 | 0.1258 | 0.0245 |
| length_shadow | 0.9913 | 3.7914 | 25.4446 |

---

## Model Performance Comparison

### Overall Performance Ranking

| Rank | Model | R² Score | MAE | MSE | Status |
|------|-------|----------|-----|-----|--------|
| 1 | **K-Nearest Neighbors** | 0.9988 | 0.7406 | 2.1986 | ✅ Best |
| 2 | **Gradient Boosting** | 0.9949 | 0.4812 | 0.7828 | ✅ |
| 3 | **Decision Tree** | 0.9936 | 1.1068 | 4.1551 | ✅ |
| 4 | **Artificial Neural Network** | 0.9910 | 1.9456 | 22.9918 | ✅ |
| 5 | **Support Vector Regression** | 0.9883 | 1.2921 | 17.1278 | ✅ |
| 6 | **Random Forest** | 0.9830 | 2.2071 | 20.7300 | ✅ |
| 7 | **Linear Regression** | 0.8341 | 4.7795 | 78.2915 | ❌ |

**Threshold:** R² > 0.90 ✅

All models except Linear Regression exceed the 0.90 R² threshold.

### Per-Target Performance Analysis

#### 1. Angle (Mie Scattering) — `angle_mie`

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| KNN | **0.9994** | 0.0187 | 0.0009 |
| ANN | 0.9890 | 0.0972 | 0.0171 |
| SVR | 0.9949 | 0.0697 | 0.0080 |
| Gradient Boosting | 0.9873 | 0.0570 | 0.0199 |
| Decision Tree | 0.9878 | 0.0444 | 0.0191 |
| Random Forest | 0.9829 | 0.1042 | 0.0267 |
| Linear Regression | 0.6567 | 0.5693 | 0.5357 |

**Winner:** KNN (R² = 0.9994)

#### 2. Length (Mie Scattering) — `length_mie`

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| Gradient Boosting | **0.9994** | 0.9084 | 1.5791 |
| ANN | 0.9909 | 3.7677 | 24.7810 |
| KNN | 0.9984 | 1.4234 | 4.2954 |
| SVR | 0.9880 | 2.4328 | 32.7417 |
| Random Forest | 0.9858 | 4.2516 | 38.7348 |
| Decision Tree | 0.9965 | 2.2803 | 9.4785 |
| Linear Regression | 0.9386 | 9.3952 | 167.0818 |

**Winner:** Gradient Boosting (R² = 0.9994)

#### 3. Angle (Shadow Imaging) — `angle_shadow`

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| KNN | **0.9989** | 0.0278 | 0.0016 |
| Gradient Boosting | 0.9933 | 0.0688 | 0.0096 |
| Decision Tree | 0.9927 | 0.0595 | 0.0105 |
| ANN | 0.9828 | 0.1258 | 0.0245 |
| SVR | 0.9826 | 0.1101 | 0.0248 |
| Random Forest | 0.9784 | 0.1215 | 0.0308 |
| Linear Regression | 0.7905 | 0.4249 | 0.2988 |

**Winner:** KNN (R² = 0.9989)

#### 4. Length (Shadow Imaging) — `length_shadow`

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| Gradient Boosting | **0.9995** | 0.8907 | 1.5227 |
| Decision Tree | 0.9976 | 2.0431 | 7.1123 |
| KNN | 0.9985 | 1.4925 | 4.4964 |
| ANN | 0.9913 | 3.7914 | 25.4446 |
| SVR | 0.9878 | 2.5559 | 35.7368 |
| Random Forest | 0.9850 | 4.3512 | 44.1277 |
| Linear Regression | 0.9505 | 8.7286 | 145.2497 |

**Winner:** Gradient Boosting (R² = 0.9995)

### Key Insights

1. **KNN dominates angle predictions** (both Mie and Shadow)
2. **Gradient Boosting dominates length predictions** (both Mie and Shadow)
3. **ANN provides balanced performance** across all targets
4. **Linear Regression struggles** with angle predictions (non-linear relationship)
5. **All tree-based models** perform exceptionally well (R² > 0.98)

---

## Hyperparameter Optimization

### Optimization Strategy

**Method:** Grid Search with Cross-Validation

**Implementation:** `GridSearchCV` from scikit-learn

**Search strategy:** Exhaustive search over specified parameter grid

### Cross-Validation Configuration

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    refit=True,
    verbose=0
)
```

**Parameters:**
- **cv:** 5-fold cross-validation
- **scoring:** R² (coefficient of determination)
- **n_jobs:** -1 (use all CPU cores)
- **refit:** True (retrain on full training set with best parameters)

### Model-Specific Hyperparameter Grids

#### Linear Regression
```python
param_grid = {}  # No hyperparameters
```
**Search space:** 1 configuration

#### Decision Tree
```python
param_grid = {
    "reg__estimator__max_depth": [None, 5, 10, 20],
    "reg__estimator__min_samples_leaf": [1, 2, 5],
}
```
**Search space:** 4 × 3 = **12 configurations**

**Best configuration (typical):**
- max_depth: None or 20
- min_samples_leaf: 1 or 2

#### Random Forest
```python
param_grid = {
    "reg__estimator__n_estimators": [200, 400],
    "reg__estimator__max_depth": [None, 10, 20],
    "reg__estimator__max_features": ["sqrt", "auto"],
}
```
**Search space:** 2 × 3 × 2 = **12 configurations**

**Best configuration (typical):**
- n_estimators: 400
- max_depth: None
- max_features: "sqrt"

#### Support Vector Regression
```python
param_grid = {
    "reg__estimator__C": [10, 100],
    "reg__estimator__gamma": ["scale"],
    "reg__estimator__epsilon": [0.01, 0.1],
    "reg__estimator__kernel": ["rbf"],
}
```
**Search space:** 2 × 1 × 2 × 1 = **4 configurations**

**Best configuration (typical):**
- C: 100
- gamma: "scale"
- epsilon: 0.01
- kernel: "rbf"

#### K-Nearest Neighbors
```python
param_grid = {
    "reg__estimator__n_neighbors": [5, 7, 9],
    "reg__estimator__weights": ["uniform", "distance"],
}
```
**Search space:** 3 × 2 = **6 configurations**

**Best configuration (typical):**
- n_neighbors: 5 or 7
- weights: "distance"

#### Gradient Boosting
```python
param_grid = {
    "reg__estimator__n_estimators": [300, 500],
    "reg__estimator__learning_rate": [0.05, 0.1],
    "reg__estimator__max_depth": [3, 4],
}
```
**Search space:** 2 × 2 × 2 = **8 configurations**

**Best configuration (typical):**
- n_estimators: 500
- learning_rate: 0.05
- max_depth: 4

### Total Hyperparameter Search Effort

| Model | Grid Combinations | CV Folds | Total Fits |
|-------|-------------------|----------|------------|
| Linear Regression | 1 | 5 | 5 |
| Decision Tree | 12 | 5 | 60 |
| Random Forest | 12 | 5 | 60 |
| SVR | 4 | 5 | 20 |
| KNN | 6 | 5 | 30 |
| Gradient Boosting | 8 | 5 | 40 |
| **Total** | **43** | **5** | **215** |

### Hyperparameter Selection Criteria

**Primary metric:** Cross-validated R² score

**Selection process:**
1. Train model with each hyperparameter combination
2. Evaluate using 5-fold cross-validation
3. Calculate mean R² across folds
4. Select configuration with highest mean R²
5. Retrain final model on full training set

---

## Model Persistence and Deployment

### Saved Model Artifacts

#### Classical ML Models (Joblib format)

| Model | Filename | Size | Format |
|-------|----------|------|--------|
| Linear Regression | `LinearRegression_regressor.joblib` | ~10 KB | Joblib |
| Decision Tree | `DecisionTree_regressor.joblib` | ~50 KB | Joblib |
| Random Forest | `RandomForest_regressor.joblib` | ~5 MB | Joblib |
| SVR | `SVR_regressor.joblib` | ~2 MB | Joblib |
| KNN | `KNN_regressor.joblib` | ~1 MB | Joblib |
| Gradient Boosting | `GradientBoosting_regressor.joblib` | ~3 MB | Joblib |

**Storage location:** `models/`

**Serialization:** Joblib (efficient for NumPy arrays)

#### Neural Network Model (HDF5 format)

| Artifact | Filename | Size | Format |
|----------|----------|------|--------|
| ANN Model | `ANN_improved_regressor.h5` | ~500 KB | HDF5 |
| Input Scaler | `ann_input_scaler.joblib` | ~5 KB | Joblib |
| Target Scaler | `ann_target_scaler.joblib` | ~5 KB | Joblib |

**Storage location:** `models/`

### Loading and Using Saved Models

#### Classical ML Models

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("models/KNN_regressor.joblib")

# Prepare input data
X_new = pd.DataFrame({
    "time": [5.0],
    "chamb_pressure": [60.0],
    "cham_temp": [700.0],
    "injection_pres": [300.0],
    "density": [780.0],
    "viscosity": [0.002]
})

# Predict (scaling handled by pipeline)
predictions = model.predict(X_new)

# Extract results
angle_mie = predictions[0, 0]
length_mie = predictions[0, 1]
angle_shadow = predictions[0, 2]
length_shadow = predictions[0, 3]
```

**Note:** The pipeline includes `StandardScaler`, so no manual scaling is required.

#### Neural Network Model

```python
import joblib
import pandas as pd
from keras.models import load_model

# Load model and scalers
model = load_model("models/ANN_improved_regressor.h5")
input_scaler = joblib.load("models/ann_input_scaler.joblib")
target_scaler = joblib.load("models/ann_target_scaler.joblib")

# Prepare input data
X_new = pd.DataFrame({
    "time": [5.0],
    "chamb_pressure": [60.0],
    "cham_temp": [700.0],
    "injection_pres": [300.0],
    "density": [780.0],
    "viscosity": [0.002]
})

# Scale inputs
X_new_scaled = input_scaler.transform(X_new)

# Predict (scaled outputs)
y_pred_scaled = model.predict(X_new_scaled)

# Inverse scale to original units
predictions = target_scaler.inverse_transform(y_pred_scaled)

# Extract results
angle_mie = predictions[0, 0]
length_mie = predictions[0, 1]
angle_shadow = predictions[0, 2]
length_shadow = predictions[0, 3]
```

**Critical:** Inputs and outputs must be scaled/unscaled using saved scalers.

### Output Files

#### Model Metrics

1. **Overall metrics:** `outputs/overall_model_metrics.csv`
   - Columns: model, overall_r2, overall_mae, overall_mse
   - 7 rows (one per model including ANN)

2. **Per-target metrics:** `outputs/per_target_metrics.csv`
   - Columns: model, target, r2, mae, mse
   - 28 rows (7 models × 4 targets)

3. **ANN metrics:** `outputs/ANN_improved_metrics.csv`
   - Columns: Model, target, r2, mae, mse
   - 4 rows (one per target)

#### Predictions

1. **ANN predictions:** `outputs/ANN_improved_predictions.csv`
   - 169 rows (test set)
   - 4 columns (predicted targets)

2. **ANN actuals:** `outputs/ANN_improved_actuals.csv`
   - 169 rows (test set)
   - 4 columns (actual targets)

---

## Validation Strategy

### Data Splitting Strategy

**Primary split:** Train-test split with stratification

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=runs
)
```

**Split ratios:**
- Training: 80% (678 samples)
- Testing: 20% (169 samples)

**Stratification variable:** `run` (experimental run identifier)

**Purpose:** Ensure test set contains proportional representation from all experimental conditions.

### Cross-Validation During Hyperparameter Tuning

**Method:** K-Fold Cross-Validation

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Configuration:**
- **n_splits:** 5 folds
- **shuffle:** True (randomize order)
- **random_state:** 42 (reproducibility)

**Workflow:**
1. Split training set into 5 folds
2. For each fold:
   - Train on 4 folds (543 samples)
   - Validate on 1 fold (135 samples)
3. Average validation scores across folds
4. Select best hyperparameters

### Neural Network Validation

**Method:** Hold-out validation split

```python
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.2,  # 20% of training data
    epochs=500,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
)
```

**Validation split:** 20% of training data (136 samples)

**Usage:**
- Monitor validation loss during training
- Trigger early stopping
- Adjust learning rate dynamically

### Evaluation Metrics

#### 1. R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

**Interpretation:**
- R² = 1: Perfect predictions
- R² = 0: Model performs as well as mean baseline
- R² < 0: Model performs worse than mean baseline

**Primary metric** for model selection.

#### 2. Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**Interpretation:**
- Average absolute deviation from true values
- Same units as target variable
- Robust to outliers

#### 3. Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Interpretation:**
- Average squared deviation from true values
- Units: (target unit)²
- Penalizes large errors more heavily

### Per-Target Evaluation

Metrics are calculated **separately for each target**:

```python
r2 = [r2_score(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]
mae = [mean_absolute_error(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]
mse = [mean_squared_error(y_te[col], y_pred[:, i]) for i, col in enumerate(y_te.columns)]
```

**Targets:**
1. angle_mie
2. length_mie
3. angle_shadow
4. length_shadow

**Overall metrics:** Mean of per-target metrics

---

## Feature Importance Analysis

### Analysis Tool: Random Forest Feature Importance

**Model used:** Random Forest Regressor (trained for multi-output regression)

**Method:** Gini importance (mean decrease in impurity)

### Extraction Process

```python
# Load trained Random Forest model
rf_model = joblib.load("models/RandomForest_regressor.joblib")

# Extract MultiOutputRegressor
multi_output_regressor = rf_model.named_steps["reg"]

# Get individual estimators (one per target)
estimators = multi_output_regressor.estimators_

# Extract feature importances for specific target
rf_angle_mie = estimators[0]  # angle_mie predictor
feature_importances = rf_angle_mie.feature_importances_
```

### Feature Names

```python
feature_names = [
    "time",
    "chamb_pressure",
    "cham_temp",
    "injection_pres",
    "density",
    "viscosity"
]
```

### Importance Interpretation

**Gini importance formula:**

For feature $j$, importance is the sum of weighted impurity decreases across all nodes where feature $j$ is used for splitting:

$$\text{Importance}_j = \sum_{t \in \text{nodes using } j} \frac{n_t}{N} \cdot \Delta i(t)$$

where:
- $n_t$: number of samples reaching node $t$
- $N$: total number of samples
- $\Delta i(t)$: decrease in impurity at node $t$

**Normalization:** Importances sum to 1.0 across all features.

### Expected Feature Patterns

Based on spray physics:

1. **Time:** Likely most important (spray evolves temporally)
2. **Injection pressure:** Strong influence on spray velocity and penetration
3. **Chamber pressure:** Affects atomization and spray angle
4. **Density & viscosity:** Influence droplet formation
5. **Chamber temperature:** Affects evaporation and spray characteristics

### Visualization

Feature importance is typically visualized as:
- **Bar plot:** Features ranked by importance
- **Percentage contribution:** Each feature's relative contribution

**Notebook:** `notebooks/feature-importance.ipynb`

---

## Dependencies and Environment

### Core Dependencies

#### Data Processing
- **pandas:** 2.3.2 — DataFrame manipulation
- **numpy:** 2.3.3 — Numerical computing
- **scipy:** 1.16.2 — Scientific computing
- **openpyxl:** 3.1.5 — Excel file reading

#### Machine Learning
- **scikit-learn:** 1.7.2 — Classical ML algorithms
- **joblib:** 1.5.2 — Model serialization

#### Deep Learning
- **tensorflow:** 2.20.0 — Neural network framework
- **keras:** 3.11.3 — High-level neural network API

#### Visualization
- **matplotlib:** 3.10.6 — Plotting library
- **seaborn:** 0.13.2 — Statistical visualization

#### Jupyter Support
- **jupyter-core:** 5.8.1
- **jupyter-client:** 8.6.3
- **ipykernel:** 6.30.1

### Python Version

**Requirement:** Python 3.10 or 3.11

**Specified in:** `pyproject.toml`

```toml
requires-python = ">=3.10,<3.12"
```

### Installation

#### Using pip (requirements.txt)

```bash
pip install -r requirements.txt
```

#### Using flit (pyproject.toml)

```bash
pip install flit
flit install --deps=production
```

#### Development dependencies

```bash
flit install --deps=develop
```

**Includes:**
- jupyter, notebook, ipykernel
- ruff (linter/formatter)

### Hardware Requirements

**Minimum:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8 GB
- Storage: 500 MB for models and data

**Recommended:**
- CPU: 8+ cores (for parallel GridSearchCV)
- RAM: 16 GB
- GPU: NVIDIA GPU with CUDA support (for TensorFlow acceleration)
  - Optional but speeds up ANN training

### GPU Support (Optional)

**Check GPU availability:**

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

**If GPU is available:** TensorFlow automatically uses it for ANN training.

**If no GPU:** CPU training works but is slower (~10x for ANN).

---

## Reproducibility

### Random Seeds

All random processes use fixed seeds for reproducibility:

**Scikit-learn models:**
```python
random_state=42
```

**Train-test split:**
```python
train_test_split(..., random_state=42, ...)
```

**Cross-validation:**
```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

**TensorFlow/Keras:**
```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

**Note:** The notebooks do not explicitly set TensorFlow seeds, so ANN results may vary slightly between runs.

### File Paths

**Input data:**
- Raw: `data/raw/DataSheet_ETH_250902.xlsx`
- Preprocessed: `data/processed/eth_spray_merged_preprocessed.csv`
- Final: `data/processed/preprocessed_dataset.csv`

**Models:**
- Directory: `models/`
- 6 classical ML models (Joblib)
- 1 ANN model (HDF5) + 2 scalers (Joblib)

**Outputs:**
- Directory: `outputs/`
- Metrics: CSV files
- Predictions: CSV files

### Execution Order

**Complete pipeline:**
1. `notebooks/data-preprocessing-pipeline.ipynb` — Data preprocessing
2. `notebooks/ml-pipeline.ipynb` — Classical ML training
3. `notebooks/ann-pipeline.ipynb` — ANN training
4. `notebooks/feature-importance.ipynb` — Feature analysis (optional)
5. `notebooks/exploratory-data-analysis.ipynb` — EDA (optional)

### Makefile Automation

**Quick commands:**

```bash
make preprocess      # Run data preprocessing
make train-ml        # Train classical ML models
make train-ann       # Train ANN model
make train-all       # Run complete pipeline
```

---

## Future Improvements

### Potential Enhancements

1. **Hyperparameter optimization:**
   - Replace GridSearchCV with Bayesian optimization (e.g., Optuna)
   - Expand search space for better performance

2. **Feature engineering:**
   - Create interaction terms (e.g., pressure × density)
   - Add polynomial features
   - Time-based features (derivatives, lag features)

3. **Ensemble methods:**
   - Stacking ensemble of top models
   - Weighted averaging of predictions

4. **Neural network improvements:**
   - Experiment with different architectures (e.g., ResNet-style)
   - Try different activation functions (e.g., LeakyReLU, ELU)
   - Hyperparameter tuning (learning rate, dropout rates, layer sizes)

5. **Temporal modeling:**
   - Recurrent Neural Networks (LSTM, GRU)
   - Temporal Convolutional Networks (TCN)
   - Transformer models

6. **Cross-validation for ANN:**
   - Implement K-fold CV for neural network training
   - Average predictions across multiple trained models

7. **Explainability:**
   - SHAP values for feature importance
   - Partial dependence plots
   - LIME for local interpretability

---

## Conclusion

The Spray Vision ML pipeline successfully predicts spray characteristics with exceptional accuracy across multiple modeling approaches. The combination of thorough data preprocessing, comprehensive hyperparameter tuning, and diverse model architectures results in a robust and reliable prediction system.

**Key achievements:**
- ✅ All models exceed 0.90 R² threshold (except Linear Regression)
- ✅ Top model (KNN) achieves R² = 0.9988
- ✅ Reproducible pipeline with saved models and scalers
- ✅ Comprehensive documentation and evaluation metrics

**Best model recommendations:**
- **For angle prediction:** K-Nearest Neighbors
- **For length prediction:** Gradient Boosting
- **For balanced performance:** Artificial Neural Network
- **For interpretability:** Decision Tree or Random Forest

---

## References

### Notebooks
- `notebooks/data-preprocessing-pipeline.ipynb`
- `notebooks/ml-pipeline.ipynb`
- `notebooks/ann-pipeline.ipynb`
- `notebooks/feature-importance.ipynb`
- `notebooks/exploratory-data-analysis.ipynb`

### Output Files
- `outputs/overall_model_metrics.csv`
- `outputs/per_target_metrics.csv`
- `outputs/ANN_improved_metrics.csv`
- `outputs/dataset_description.csv`

### Configuration Files
- `requirements.txt`
- `pyproject.toml`
- `Makefile`

### Project Repository
- **GitHub:** yuvinraja/spray-vision
- **Branch:** main

---

**Document Version:** 1.0  
**Last Updated:** October 16, 2025  
**Author:** Yuvin Raja  
**Project:** Spray Vision ML Pipeline
