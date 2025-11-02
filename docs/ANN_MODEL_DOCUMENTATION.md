# Artificial Neural Network (ANN) Documentation

**Project:** Spray Vision  
**Model:** ANN_improved_regressor.h5  
**Author:** Yuvin Raja  
**Last Updated:** October 16, 2025

---

## Overview

The Artificial Neural Network (ANN) implemented in Spray Vision is a deep feedforward multi-layer perceptron designed for multi-output regression. It predicts four spray characteristics simultaneously:

- Spray cone angle (Mie scattering imaging) — `angle_mie`
- Spray penetration length (Mie scattering imaging) — `length_mie`
- Spray cone angle (Shadow imaging) — `angle_shadow`
- Spray penetration length (Shadow imaging) — `length_shadow`

---

## Model Architecture

**Framework:** TensorFlow 2.20.0 + Keras 3.11.3

**Type:** Sequential deep neural network

**Input dimension:** 6 (features)
**Output dimension:** 4 (targets)

### Layer Structure

| Layer         | Neurons | Activation | BatchNorm | Dropout |
|---------------|---------|------------|-----------|---------|
| Input         | 6       | -          | -         | -       |
| Dense 1       | 256     | ReLU       | Yes       | 0.4     |
| Dense 2       | 128     | ReLU       | Yes       | 0.3     |
| Dense 3       | 64      | ReLU       | Yes       | 0.2     |
| Output        | 4       | Linear     | -         | -       |

**Total trainable parameters:** ~43,000

### Code Example

```python
def build_deep_ann(input_dim, output_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(output_dim, activation="linear"),
    ])
    return model
```

---

## Data Preprocessing

- **Input features:** Standardized using `StandardScaler` (zero mean, unit variance)
- **Target outputs:** Standardized using `StandardScaler`
- **Scalers saved as:**
  - `models/ann_input_scaler.joblib`
  - `models/ann_target_scaler.joblib`

**Critical:** Always scale inputs and inverse-transform outputs when using the model.

---

## Training Configuration

| Parameter         | Value         |
|-------------------|--------------|
| Epochs            | 500          |
| Batch size        | 32           |
| Validation split  | 0.2 (20%)    |
| Optimizer         | Adam         |
| Learning rate     | 0.001 (default) |
| Loss function     | Mean Squared Error (MSE) |
| Metrics           | Mean Absolute Error (MAE) |

### Regularization

- **Dropout:** 0.4, 0.3, 0.2 (per layer)
- **Batch Normalization:** After each dense layer
- **Early Stopping:** Patience 30 epochs, restores best weights
- **ReduceLROnPlateau:** Halves learning rate if validation loss plateaus (patience 10)

### Callbacks Example

```python
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)

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

---

## Evaluation Metrics

- **R² (Coefficient of Determination):** Measures explained variance
- **MAE (Mean Absolute Error):** Average absolute error
- **MSE (Mean Squared Error):** Average squared error

### Performance (Test Set)

| Target         | R²      | MAE     | MSE     |
|----------------|---------|---------|---------|
| angle_mie      | 0.9890  | 0.0972  | 0.0171  |
| length_mie     | 0.9909  | 3.7677  | 24.7810 |
| angle_shadow   | 0.9828  | 0.1258  | 0.0245  |
| length_shadow  | 0.9913  | 3.7914  | 25.4446 |

**Overall mean R²:** 0.9910

---

## Model Persistence

- **Model file:** `models/ANN_improved_regressor.h5` (Keras HDF5 format)
- **Input scaler:** `models/ann_input_scaler.joblib`
- **Target scaler:** `models/ann_target_scaler.joblib`

### Usage Example

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

---

## Best Practices

- Always use the saved scalers for input and output transformation
- Monitor validation loss during training to avoid overfitting
- Use callbacks for robust training
- Save model and scalers for reproducibility

---

## References

- `notebooks/ann-pipeline.ipynb`
- `models/ANN_improved_regressor.h5`
- `models/ann_input_scaler.joblib`
- `models/ann_target_scaler.joblib`
- `outputs/ANN_improved_metrics.csv`

---

**Document Version:** 1.0  
**Last Updated:** October 16, 2025  
**Author:** Yuvin Raja
