# Spray Vision: Models Documentation

## Table of Contents
1. [Overview](#overview)
2. [Model Inventory](#model-inventory)
3. [Model Details](#model-details)
4. [Performance Comparison](#performance-comparison)
5. [Model Selection Guide](#model-selection-guide)
6. [Deployment Guide](#deployment-guide)
7. [Retraining Guide](#retraining-guide)

---

## Overview

The Spray Vision project includes **7 trained machine learning models** for predicting spray characteristics. This document provides comprehensive information about each model, including architecture, performance, file specifications, and usage guidelines.

### Model Categories

**Classical Machine Learning (6 models)**:
- Linear Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting

**Deep Learning (1 model)**:
- Artificial Neural Network (ANN)

All models are trained on the same data and predict the same 4 target variables:
1. Spray angle (shadow method)
2. Spray penetration length (shadow method)
3. Spray angle (Mie scattering method)
4. Spray penetration length (Mie scattering method)

---

## Model Inventory

### Stored Models

All trained models are stored in the `models/` directory:

| Model Name | File | Size | Format | Compression |
|------------|------|------|--------|-------------|
| Linear Regression | LinearRegression_regressor.joblib | 2.6 KB | Joblib | Yes |
| Decision Tree | DecisionTree_regressor.joblib | 321 KB | Joblib | Yes |
| KNN | KNN_regressor.joblib | 274 KB | Joblib | Yes |
| SVR | SVR_regressor.joblib | 91 KB | Joblib | Yes |
| Random Forest | RandomForest_regressor.joblib | 40 MB | Joblib | Yes |
| Gradient Boosting | GradientBoosting_regressor.joblib | 4.3 MB | Joblib | Yes |
| ANN | ANN_improved_regressor.h5 | 581 KB | HDF5 | Yes |
| ANN Input Scaler | ann_input_scaler.joblib | 1.1 KB | Joblib | Yes |
| ANN Target Scaler | ann_target_scaler.joblib | 1.0 KB | Joblib | Yes |

**Total Storage**: ~45 MB

### File Formats

**Joblib (.joblib)**:
- Format: Serialized Python objects
- Library: joblib (scikit-learn ecosystem)
- Advantages: Fast loading, compression, cross-platform
- Disadvantages: Python-specific, version-dependent

**HDF5 (.h5)**:
- Format: Hierarchical Data Format
- Library: h5py (TensorFlow/Keras)
- Advantages: Language-independent, efficient, structured
- Disadvantages: Requires h5py library

---

## Model Details

### 1. Linear Regression

**Type**: Parametric, Linear
**Wrapper**: MultiOutputRegressor
**File**: LinearRegression_regressor.joblib (2.6 KB)

#### Architecture
```
Input Features (6) → Linear Transformation → Output Targets (4)
```

#### Mathematical Form
```
y = Xβ + ε

where:
  y: Target variables (4-dimensional)
  X: Input features (6-dimensional)
  β: Coefficients (learned parameters)
  ε: Error term
```

#### Hyperparameters
- **fit_intercept**: True (includes bias term)
- **normalize**: False (features pre-scaled if needed)
- **n_jobs**: None (single-threaded, fast anyway)

#### Training Configuration
- **Solver**: Ordinary Least Squares (closed-form solution)
- **Training Time**: < 1 second
- **Memory**: Minimal (~10 MB during training)

#### Performance Metrics
- **R² Score**: 0.8341
- **MAE**: 4.7795
- **MSE**: 78.2915
- **Rank**: 6 / 7

#### Characteristics
✅ **Strengths**:
- Fastest training and inference
- Most interpretable (coefficient inspection)
- No hyperparameter tuning needed
- Minimal storage (2.6 KB)
- Mathematically elegant

❌ **Weaknesses**:
- Assumes linear relationships (poor fit for non-linear data)
- Sensitive to outliers
- No feature interactions captured
- Lowest accuracy among all models

#### Use Cases
- Baseline model for comparison
- When interpretability is critical
- Real-time systems with extreme latency constraints
- Educational purposes

#### Model Coefficients
The model learns 6 coefficients per target (4 targets × 6 features = 24 coefficients plus 4 intercepts).

**Example Interpretation**:
```
angle_shadow = β₀ + β₁×time + β₂×pressure + β₃×temp + β₄×inj_press + β₅×density + β₆×viscosity
```

---

### 2. Decision Tree Regressor

**Type**: Non-parametric, Tree-based
**Wrapper**: MultiOutputRegressor
**File**: DecisionTree_regressor.joblib (321 KB)

#### Architecture
```
Input Features → Binary Splits → Leaf Nodes (Predictions)
```

#### Tree Structure
- **Depth**: Varies (optimized via grid search)
- **Splits**: Binary decisions on single features
- **Leaves**: Terminal nodes with predicted values
- **Splitting Criterion**: MSE minimization

#### Best Hyperparameters (from Grid Search)
- **max_depth**: 20 (maximum tree depth)
- **min_samples_split**: 2 (minimum samples to split a node)
- **min_samples_leaf**: 1 (minimum samples in leaf)
- **splitter**: 'best' (best split at each node)
- **criterion**: 'squared_error' (MSE)

#### Training Configuration
- **Grid Search CV**: 5 folds
- **Parameter Combinations Tested**: 36
- **Training Time**: ~1 minute
- **Memory**: ~500 MB during training

#### Performance Metrics
- **R² Score**: 0.9936
- **MAE**: 1.1068
- **MSE**: 4.1551
- **Rank**: 3 / 7

#### Characteristics
✅ **Strengths**:
- Captures non-linear relationships
- No feature scaling required
- Can visualize tree structure
- Handles feature interactions
- Fast inference
- Interpretable rules

❌ **Weaknesses**:
- Prone to overfitting (high variance)
- Unstable (small data changes → different tree)
- Biased toward features with many levels
- Not optimal for extrapolation

#### Use Cases
- When interpretability is important
- Mixed data types (categorical + numerical)
- Feature importance analysis
- Quick prototyping

#### Tree Statistics
```python
import joblib
model = joblib.load("models/DecisionTree_regressor.joblib")

# For each target's tree:
print(f"Number of nodes: ~{model.tree_.node_count}")
print(f"Number of leaves: ~{model.tree_.n_leaves}")
print(f"Max depth: {model.tree_.max_depth}")
```

---

### 3. K-Nearest Neighbors (KNN)

**Type**: Instance-based, Non-parametric
**Wrapper**: Pipeline(StandardScaler + KNeighborsRegressor) + MultiOutputRegressor
**File**: KNN_regressor.joblib (274 KB)

#### Architecture
```
Input Features → Scaling → Distance Calculation → K-Nearest Selection → Average
```

#### Algorithm
1. Store all training samples
2. For new input, calculate distance to all training samples
3. Select K nearest neighbors
4. Predict as weighted average of neighbors' targets

#### Best Hyperparameters (from Grid Search)
- **n_neighbors**: 5 (number of neighbors)
- **weights**: 'distance' (inverse distance weighting)
- **p**: 2 (Euclidean distance metric)
- **algorithm**: 'auto' (best algorithm selection)

#### Training Configuration
- **Grid Search CV**: 5 folds
- **Parameter Combinations Tested**: 20
- **Training Time**: ~30 seconds
- **Memory**: ~300 MB during training

#### Performance Metrics
- **R² Score**: 0.9988 ⭐ **BEST**
- **MAE**: 0.7406 ⭐ **BEST**
- **MSE**: 2.1986
- **Rank**: 1 / 7 ⭐

#### Characteristics
✅ **Strengths**:
- **Highest accuracy** among all models
- No training phase (just stores data)
- Naturally handles multi-output regression
- Adapts to local patterns
- No assumptions about data distribution

❌ **Weaknesses**:
- Slow inference (computes distances to all training points)
- Memory-intensive (stores entire training set)
- Sensitive to feature scaling (requires preprocessing)
- Poor for high-dimensional data (curse of dimensionality)
- No model interpretation

#### Use Cases
- When maximum accuracy is required
- Small to medium datasets
- Non-parametric applications
- Benchmarking other models

#### Memory Consideration
KNN stores ~677 training samples × 6 features = 4,062 float values ≈ 32 KB raw data.
The 274 KB file includes scaler parameters and metadata.

---

### 4. Support Vector Regression (SVR)

**Type**: Kernel-based, Maximum-margin
**Wrapper**: Pipeline(StandardScaler + SVR) + MultiOutputRegressor
**File**: SVR_regressor.joblib (91 KB)

#### Architecture
```
Input Features → Scaling → Kernel Transformation → Support Vectors → Prediction
```

#### Algorithm
Find hyperplane in high-dimensional space that:
- Fits most data within ε-tube
- Minimizes slack variable penalty (C)
- Maximizes margin

#### Best Hyperparameters (from Grid Search)
- **kernel**: 'rbf' (Radial Basis Function)
- **C**: 100 (regularization parameter)
- **epsilon**: 0.01 (ε-tube width)
- **gamma**: 'scale' (kernel coefficient)

#### Training Configuration
- **Grid Search CV**: 5 folds
- **Parameter Combinations Tested**: 48
- **Training Time**: ~2 minutes
- **Memory**: ~1 GB during training

#### Performance Metrics
- **R² Score**: 0.9883
- **MAE**: 1.2921
- **MSE**: 17.1278
- **Rank**: 4 / 7

#### Characteristics
✅ **Strengths**:
- Robust to outliers
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Kernel trick allows non-linear boundaries
- Strong theoretical foundation

❌ **Weaknesses**:
- Slow training for large datasets
- Requires careful hyperparameter tuning
- Sensitive to feature scaling
- Not naturally multi-output (needs wrapper)
- Less interpretable

#### Use Cases
- Regression with outliers
- High-dimensional feature spaces
- When theoretical guarantees matter
- Medium-sized datasets

#### Support Vector Statistics
The model internally stores only the support vectors (subset of training data that define the decision boundary). Typically 20-50% of training samples become support vectors.

---

### 5. Random Forest Regressor

**Type**: Ensemble, Tree-based
**Wrapper**: MultiOutputRegressor
**File**: RandomForest_regressor.joblib (40 MB)

#### Architecture
```
Input Features → Multiple Decision Trees (ensemble) → Average Predictions
```

#### Ensemble Strategy
- **Bagging**: Bootstrap aggregating (sample with replacement)
- **Random Feature Selection**: Each split considers subset of features
- **Averaging**: Final prediction is mean of all trees

#### Best Hyperparameters (from Grid Search)
- **n_estimators**: 200 (number of trees)
- **max_depth**: 30 (maximum depth per tree)
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **max_features**: 'sqrt' (√6 ≈ 2.45 features per split)
- **bootstrap**: True

#### Training Configuration
- **Grid Search CV**: 5 folds
- **Parameter Combinations Tested**: 144
- **Training Time**: ~5 minutes
- **Memory**: ~2 GB during training
- **Parallelization**: All CPU cores (n_jobs=-1)

#### Performance Metrics
- **R² Score**: 0.9830
- **MAE**: 2.2071
- **MSE**: 20.7300
- **Rank**: 5 / 7

#### Characteristics
✅ **Strengths**:
- Reduces overfitting vs single tree
- Provides feature importance
- Handles non-linearity well
- Robust to outliers and noise
- Parallel training
- No feature scaling needed

❌ **Weaknesses**:
- **Large file size** (40 MB)
- Slower inference than single tree
- Memory-intensive
- Less interpretable than single tree
- Can overfit with too many trees

#### Use Cases
- When accuracy and robustness are important
- Feature importance analysis needed
- Sufficient memory and storage available
- Classification or regression tasks

#### Forest Statistics
```python
import joblib
model = joblib.load("models/RandomForest_regressor.joblib")

print(f"Number of trees: {model.n_estimators}")
print(f"Features considered per split: {model.max_features}")

# Feature importance (averaged across trees)
importances = model.feature_importances_
```

The 40 MB size comes from storing 200 trees × 4 targets = 800 decision trees.

---

### 6. Gradient Boosting Regressor

**Type**: Ensemble, Sequential Tree-based
**Wrapper**: MultiOutputRegressor
**File**: GradientBoosting_regressor.joblib (4.3 MB)

#### Architecture
```
Input Features → Sequential Trees (each corrects previous errors) → Weighted Sum
```

#### Boosting Strategy
- **Sequential Training**: Each tree fits residuals of previous ensemble
- **Gradient Descent**: Minimizes loss function via gradient descent
- **Shrinkage**: Learning rate controls contribution of each tree
- **Subsampling**: Stochastic gradient boosting for regularization

#### Best Hyperparameters (from Grid Search)
- **n_estimators**: 300 (number of sequential trees)
- **learning_rate**: 0.05 (shrinkage parameter)
- **max_depth**: 5 (depth per tree, typically shallow)
- **subsample**: 0.8 (80% data per tree)
- **loss**: 'squared_error' (MSE)

#### Training Configuration
- **Grid Search CV**: 5 folds
- **Parameter Combinations Tested**: 54
- **Training Time**: ~3 minutes
- **Memory**: ~1.5 GB during training

#### Performance Metrics
- **R² Score**: 0.9949 🥈 **Second Best**
- **MAE**: 0.4812 🥇 **Lowest MAE**
- **MSE**: 0.7828 🥇 **Lowest MSE**
- **Rank**: 2 / 7

#### Characteristics
✅ **Strengths**:
- **Excellent accuracy** (2nd best R², best MAE/MSE)
- Often best performing in competitions
- Handles feature interactions
- Built-in feature importance
- Moderate file size (4.3 MB)
- Good generalization

❌ **Weaknesses**:
- Sequential training (cannot parallelize)
- Slower training than Random Forest
- More hyperparameters to tune
- Can overfit if not careful
- Sensitive to outliers

#### Use Cases
- **Recommended for deployment** (best accuracy/size trade-off)
- Kaggle competitions
- Production systems
- When accuracy is paramount
- Feature importance needed

#### Boosting Process
Each tree is trained to predict residuals:
```
Tree 1: Predict y from X
Tree 2: Predict (y - pred1) from X
Tree 3: Predict (y - pred1 - pred2) from X
...
Final: pred = α₁×tree1 + α₂×tree2 + ... + αₙ×treeₙ
```

---

### 7. Artificial Neural Network (ANN)

**Type**: Deep Learning, Feedforward
**Files**: 
- ANN_improved_regressor.h5 (581 KB)
- ann_input_scaler.joblib (1.1 KB)
- ann_target_scaler.joblib (1.0 KB)

#### Architecture
```
Input (6) → Dense(64, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output (4)
```

#### Layer Details

| Layer | Type | Neurons | Activation | Parameters |
|-------|------|---------|------------|------------|
| Input | - | 6 | - | 0 |
| Hidden 1 | Dense | 64 | ReLU | 448 |
| Hidden 2 | Dense | 128 | ReLU | 8,320 |
| Hidden 3 | Dense | 64 | ReLU | 8,256 |
| Hidden 4 | Dense | 32 | ReLU | 2,080 |
| Output | Dense | 4 | Linear | 132 |
| **Total** | | | | **19,236** |

#### Network Configuration
- **Optimizer**: Adam
  - Learning rate: 0.001
  - β₁: 0.9 (momentum)
  - β₂: 0.999 (RMSprop)
- **Loss Function**: Mean Squared Error
- **Batch Size**: 32
- **Epochs**: Up to 500 (with early stopping)

#### Training Configuration
- **Validation Split**: 20%
- **Early Stopping**: 
  - Monitor: validation loss
  - Patience: 20 epochs
  - Restore best weights: Yes
- **Training Time**: 2-5 minutes (CPU), < 1 minute (GPU)
- **Typical Stopping**: 50-150 epochs

#### Scaling Requirements
**Critical**: ANN requires feature scaling
```python
# Load scalers
input_scaler = joblib.load("models/ann_input_scaler.joblib")
target_scaler = joblib.load("models/ann_target_scaler.joblib")

# Scale inputs
X_scaled = input_scaler.transform(X)

# Predict
y_scaled = model.predict(X_scaled)

# Inverse scale outputs
y_pred = target_scaler.inverse_transform(y_scaled)
```

#### Performance Metrics
- **R² Score**: ~0.99 (typical, varies with training)
- **MAE**: < 1.0
- **MSE**: < 5.0
- **Rank**: Competitive with top models

#### Characteristics
✅ **Strengths**:
- Universal approximation capability
- Automatically learns features
- Fast inference (especially on GPU)
- Moderate file size (581 KB)
- Scalable to larger datasets
- Can incorporate additional techniques (dropout, regularization)

❌ **Weaknesses**:
- Requires feature scaling
- "Black box" (hard to interpret)
- Requires more hyperparameter tuning
- Can overfit without proper regularization
- Training can be unstable

#### Use Cases
- Large datasets (scales well)
- When maximum flexibility needed
- GPU available for inference
- Transfer learning applications
- Ensemble with other models

#### Architecture Rationale
- **Expanding-Contracting**: 64→128→64→32 captures complex patterns
- **ReLU**: Prevents vanishing gradients, enables deep networks
- **No Dropout**: Dataset small enough, early stopping sufficient
- **Linear Output**: Unbounded regression targets

---

## Performance Comparison

### Overall Rankings

| Rank | Model | R² Score | MAE | MSE | File Size | Speed |
|------|-------|----------|-----|-----|-----------|-------|
| 1 🥇 | **KNN** | **0.9988** | **0.74** | 2.20 | 274 KB | Slow |
| 2 🥈 | **Gradient Boosting** | **0.9949** | 0.48 | **0.78** | 4.3 MB | Fast |
| 3 🥉 | Decision Tree | 0.9936 | 1.11 | 4.16 | 321 KB | Fast |
| 4 | SVR | 0.9883 | 1.29 | 17.13 | 91 KB | Fast |
| 5 | Random Forest | 0.9830 | 2.21 | 20.73 | 40 MB | Moderate |
| 6 | ANN | ~0.99 | ~1.0 | ~5.0 | 581 KB | Very Fast |
| 7 | Linear Regression | 0.8341 | 4.78 | 78.29 | 2.6 KB | Fastest |

### Performance Visualization

```
R² Score Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNN                 ████████████████████ 0.9988
Gradient Boosting   ███████████████████▊ 0.9949
Decision Tree       ███████████████████▋ 0.9936
SVR                 ███████████████████▌ 0.9883
Random Forest       ███████████████████▍ 0.9830
ANN                 ███████████████████▊ ~0.99
Linear Regression   ████████████████▋    0.8341
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    0.0              1.0
```

### Metric Trade-offs

**Accuracy vs Size**:
- Best accuracy/size ratio: **Gradient Boosting** (0.9949 R² at 4.3 MB)
- Smallest size: Linear Regression (2.6 KB, but poor accuracy)
- Largest size: Random Forest (40 MB, good but not best accuracy)

**Accuracy vs Speed**:
- Fastest accurate model: **Gradient Boosting**
- Slowest model: KNN (computes distances at inference)
- Speed champion: Linear Regression (but lowest accuracy)

**Interpretability vs Accuracy**:
- Most interpretable: Decision Tree, Linear Regression
- Least interpretable: ANN, KNN
- Best balance: Gradient Boosting (feature importance + accuracy)

---

## Model Selection Guide

### By Use Case

#### 1. Production Deployment
**Recommended**: **Gradient Boosting**
- Excellent accuracy (R² = 0.9949)
- Reasonable file size (4.3 MB)
- Fast inference
- Feature importance available

**Alternative**: ANN (if GPU available)

#### 2. Maximum Accuracy
**Recommended**: **KNN**
- Highest R² (0.9988)
- Lowest MAE (0.74)

**Trade-off**: Slow inference, higher memory usage

#### 3. Real-Time Systems
**Recommended**: **Decision Tree** or **Linear Regression**
- Microsecond inference
- Small memory footprint
- No dependencies

**Trade-off**: Lower accuracy (especially Linear Regression)

#### 4. Interpretability
**Recommended**: **Decision Tree**
- Visualize decision rules
- Understand feature splits
- Good accuracy (R² = 0.9936)

**Alternative**: Linear Regression (coefficients interpretation)

#### 5. Research and Benchmarking
**Recommended**: Train all models
- Compare performance
- Understand trade-offs
- Publish comprehensive results

#### 6. Edge Devices / Embedded Systems
**Recommended**: **Linear Regression** or **SVR**
- Minimal size (< 100 KB)
- No external dependencies
- Fast inference

**Trade-off**: Reduced accuracy

#### 7. Large-Scale Predictions
**Recommended**: **ANN** (with GPU)
- Batch predictions
- GPU parallelization
- Scalable architecture

**Alternative**: Gradient Boosting (CPU)

### Decision Tree

```
                    Need interpretability?
                          /        \
                        Yes         No
                         |           |
                 Decision Tree   Need max accuracy?
                                    /        \
                                  Yes         No
                                   |           |
                                  KNN      Budget for size?
                                             /        \
                                           Yes         No
                                            |           |
                                    Gradient      Decision Tree
                                    Boosting      or SVR
```

---

## Deployment Guide

### Loading Models

#### Classical ML Models
```python
import joblib
import numpy as np

# Load model
model = joblib.load("models/GradientBoosting_regressor.joblib")

# Prepare input (must match training feature order)
X_new = np.array([[
    time,           # Time_ms
    chamb_press,    # Pc_bar
    cham_temp,      # Tc_K
    inj_press,      # Pinj_bar
    density,        # rho_kgm3
    viscosity       # mu_Pas
]])

# Predict
predictions = model.predict(X_new)
# Returns: [angle_shadow, len_shadow, angle_mie, len_mie]

# Extract individual predictions
angle_shadow = predictions[0][0]
len_shadow = predictions[0][1]
angle_mie = predictions[0][2]
len_mie = predictions[0][3]
```

#### Neural Network
```python
import joblib
import numpy as np
from keras.models import load_model

# Load model and scalers
model = load_model("models/ANN_improved_regressor.h5")
input_scaler = joblib.load("models/ann_input_scaler.joblib")
target_scaler = joblib.load("models/ann_target_scaler.joblib")

# Prepare input
X_new = np.array([[time, chamb_press, cham_temp, inj_press, density, viscosity]])

# Scale input
X_scaled = input_scaler.transform(X_new)

# Predict (scaled)
y_scaled = model.predict(X_scaled, verbose=0)

# Inverse transform to original scale
predictions = target_scaler.inverse_transform(y_scaled)

# Extract predictions (same as above)
angle_shadow, len_shadow, angle_mie, len_mie = predictions[0]
```

### Batch Predictions

```python
# Multiple samples
X_batch = np.array([
    [time1, press1, temp1, inj1, dens1, visc1],
    [time2, press2, temp2, inj2, dens2, visc2],
    [time3, press3, temp3, inj3, dens3, visc3],
])

# Predict all at once
predictions = model.predict(X_batch)
# Shape: (3, 4) - 3 samples, 4 targets each
```

### API Wrapper Example

```python
class SprayPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        
    def predict(self, time, chamber_pressure, chamber_temp, 
                injection_pressure, density, viscosity):
        """Predict spray characteristics."""
        X = np.array([[time, chamber_pressure, chamber_temp,
                       injection_pressure, density, viscosity]])
        pred = self.model.predict(X)[0]
        
        return {
            'angle_shadow_deg': float(pred[0]),
            'penetration_shadow_LD': float(pred[1]),
            'angle_mie_deg': float(pred[2]),
            'penetration_mie_LD': float(pred[3])
        }

# Usage
predictor = SprayPredictor("models/GradientBoosting_regressor.joblib")
result = predictor.predict(1.0, 55.0, 650.0, 800.0, 780.0, 0.002)
print(result)
```

### Web Service Example (Flask)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/GradientBoosting_regressor.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array([[
        data['time'],
        data['chamber_pressure'],
        data['chamber_temperature'],
        data['injection_pressure'],
        data['density'],
        data['viscosity']
    ]])
    
    pred = model.predict(X)[0]
    
    return jsonify({
        'angle_shadow': float(pred[0]),
        'penetration_shadow': float(pred[1]),
        'angle_mie': float(pred[2]),
        'penetration_mie': float(pred[3])
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY api.py .

EXPOSE 5000

CMD ["python", "api.py"]
```

---

## Retraining Guide

### When to Retrain

- **New data available**: Additional experimental runs
- **Concept drift**: Operating conditions change
- **Performance degradation**: Prediction accuracy drops
- **New features**: Additional input variables
- **Different targets**: New spray characteristics to predict

### Retraining Process

#### 1. Prepare New Data
```bash
# Add new Excel file to data/raw/
# Update data-preprocessing-pipeline.ipynb to include new data
make preprocess
```

#### 2. Retrain Classical ML Models
```bash
# Automatically retrains all models
make train-ml

# Or manually:
jupyter nbconvert --to notebook --execute notebooks/ml-pipeline.ipynb
```

#### 3. Retrain Neural Network
```bash
make train-ann

# Or manually:
jupyter nbconvert --to notebook --execute notebooks/ann-pipeline.ipynb
```

#### 4. Evaluate New Models
```python
# Compare metrics with previous version
import pandas as pd

old_metrics = pd.read_csv("outputs/overall_model_metrics_old.csv")
new_metrics = pd.read_csv("outputs/overall_model_metrics.csv")

comparison = pd.merge(old_metrics, new_metrics, on='model', 
                     suffixes=('_old', '_new'))
comparison['r2_diff'] = comparison['r2_new'] - comparison['r2_old']
print(comparison[['model', 'r2_old', 'r2_new', 'r2_diff']])
```

#### 5. Version Control
```bash
# Save old models
mkdir models/v1.0
cp models/*.joblib models/*.h5 models/v1.0/

# New models are already in models/
# Document version in README or CHANGELOG
```

### Hyperparameter Retuning

To adjust hyperparameter grids, edit `ml-pipeline.ipynb`:

```python
# Example: Expand Decision Tree grid
param_grid = {
    'max_depth': [10, 15, 20, 25, 30, None],  # Add 15, 25
    'min_samples_split': [2, 5, 10, 20],      # Add 20
    'min_samples_leaf': [1, 2, 4, 8]          # Add 8
}
```

### Transfer Learning (ANN)

```python
# Load existing model
base_model = load_model("models/ANN_improved_regressor.h5")

# Freeze early layers
for layer in base_model.layers[:2]:
    layer.trainable = False

# Compile and train on new data
base_model.compile(optimizer='adam', loss='mse')
base_model.fit(X_new, y_new, epochs=50)

# Save updated model
base_model.save("models/ANN_improved_v2.h5")
```

---

## Summary

The Spray Vision project provides a comprehensive suite of machine learning models:

✅ **7 Different Approaches**: From simple linear regression to deep neural networks
✅ **Trained and Ready**: All models pre-trained on ETH spray dataset
✅ **Well-Documented**: Complete specifications and usage guidelines
✅ **Deployment-Ready**: Examples for various deployment scenarios
✅ **Reproducible**: Full retraining pipeline available

**Recommended for Production**: **Gradient Boosting** (best accuracy/efficiency balance)
**Recommended for Research**: Train and compare all models
**Recommended for Edge Devices**: Linear Regression or SVR

All models predict the same 4 spray characteristics with excellent accuracy (R² > 0.98 for most models).

---

*This document provides complete information about all trained models in the Spray Vision project.*
