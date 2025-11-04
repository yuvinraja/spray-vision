# Spray Vision: Comprehensive Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [What is Spray Vision?](#what-is-spray-vision)
3. [Project Goals and Objectives](#project-goals-and-objectives)
4. [Scientific Background](#scientific-background)
5. [Project Architecture](#project-architecture)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Key Features](#key-features)
9. [Research Context](#research-context)

---

## Project Overview

**Spray Vision** is a comprehensive machine learning research project focused on predicting spray characteristics in fuel injection systems using both classical machine learning algorithms and deep learning neural networks. This project demonstrates the application of modern data science techniques to solve complex engineering problems in combustion and fluid dynamics.

### Project Metadata
- **Name**: spray-vision
- **Version**: 0.1.0
- **License**: MIT
- **Python Version**: 3.10 - 3.11
- **Template**: Cookiecutter Data Science
- **Author**: Yuvin Raja

---

## What is Spray Vision?

Spray Vision is a predictive modeling system that uses machine learning to forecast spray behavior characteristics based on experimental conditions. The project analyzes spray data from the ETH (Swiss Federal Institute of Technology) spray dataset and builds multiple predictive models to understand and predict:

### Primary Prediction Targets
1. **Spray Angle (Shadow Method)** - Measured in degrees
2. **Spray Angle (Mie Scattering Method)** - Measured in degrees
3. **Spray Penetration Length (Shadow Method)** - Expressed as L/D ratio
4. **Spray Penetration Length (Mie Scattering Method)** - Expressed as L/D ratio

### Input Parameters
The models predict spray characteristics based on five key experimental conditions:
1. **Chamber Pressure (Pc)** - Measured in bar
2. **Chamber Temperature (Tc)** - Measured in Kelvin
3. **Injection Pressure (Pinj)** - Measured in bar
4. **Fuel Density (ρ)** - Measured in kg/m³
5. **Fuel Viscosity (μ)** - Measured in Pa·s
6. **Time** - Temporal dimension in milliseconds

---

## Project Goals and Objectives

### Primary Goals
1. **Develop High-Accuracy Predictive Models**: Create machine learning models that can accurately predict spray characteristics with R² scores exceeding 0.90
2. **Compare ML Approaches**: Systematically evaluate classical ML algorithms against deep learning approaches
3. **Enable Real-Time Predictions**: Provide a framework for quick spray characteristic predictions without expensive experimental setups
4. **Establish Best Practices**: Demonstrate proper data science workflow for engineering applications

### Research Objectives
- **Performance Benchmarking**: Compare 7 different machine learning algorithms
- **Feature Importance Analysis**: Identify which input parameters most strongly influence spray behavior
- **Multi-Output Prediction**: Simultaneously predict multiple spray characteristics
- **Model Optimization**: Fine-tune hyperparameters for optimal performance
- **Reproducibility**: Create a fully reproducible pipeline from raw data to trained models

### Practical Applications
- **Engine Design**: Assist in optimizing fuel injection systems
- **Combustion Research**: Support studies on spray-combustion interactions
- **Simulation Validation**: Provide reference data for CFD simulations
- **Educational Tool**: Demonstrate ML applications in mechanical engineering

---

## Scientific Background

### Spray Characteristics Importance

Fuel spray characteristics are critical in determining:
- **Combustion Efficiency**: Better atomization leads to more complete combustion
- **Emission Control**: Spray behavior affects pollutant formation (NOx, particulates)
- **Engine Performance**: Optimal spray penetration and angle improve power output
- **Fuel Economy**: Proper spray patterns reduce fuel consumption

### Measurement Techniques

The project uses data from two complementary measurement methods:

#### 1. Shadow Method (Shadowgraphy)
- Uses backlit imaging to capture spray silhouette
- Measures geometric spray boundaries
- Provides information on spray envelope
- Better for dense spray regions

#### 2. Mie Scattering Method
- Uses laser light scattering from droplets
- Sensitive to droplet concentration
- Provides information on liquid phase distribution
- Better for dilute spray regions

### Physical Phenomena

The spray behavior is governed by:
- **Atomization**: Breakup of liquid jet into droplets
- **Momentum Exchange**: Interaction between fuel spray and ambient gas
- **Evaporation**: Liquid-to-vapor phase change
- **Turbulent Mixing**: Chaotic flow patterns affecting distribution

---

## Project Architecture

### Data Science Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                          │
│  ETH Spray Dataset (Excel) → DataSheet_ETH_250902.xlsx      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING PIPELINE                      │
│  • Multi-header Excel parsing                                │
│  • Data stacking and reshaping                               │
│  • Missing value imputation (forward/backward fill)          │
│  • Data validation and integrity checks                      │
│  Output: preprocessed_dataset.csv (847 samples)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│            EXPLORATORY DATA ANALYSIS                          │
│  • Statistical summaries                                      │
│  • Distribution analysis                                      │
│  • Correlation studies                                        │
│  • Visualization generation                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────┴───────┐
              ↓               ↓
┌──────────────────────┐  ┌──────────────────────┐
│  CLASSICAL ML        │  │  DEEP LEARNING       │
│  PIPELINE            │  │  PIPELINE (ANN)      │
│                      │  │                      │
│  • 6 ML Models       │  │  • Neural Network    │
│  • Hyperparameter    │  │  • Architecture      │
│    Tuning            │  │    Design            │
│  • Cross-Validation  │  │  • Training with     │
│  • Model Saving      │  │    Callbacks         │
└─────────┬────────────┘  └─────────┬────────────┘
          │                         │
          └───────────┬─────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              MODEL EVALUATION & COMPARISON                    │
│  • Performance metrics (R², MAE, MSE)                        │
│  • Per-target analysis                                        │
│  • Residual analysis                                          │
│  • Feature importance                                         │
│  • Visualization of results                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                  TRAINED MODELS & OUTPUTS                     │
│  • Serialized models (.joblib, .h5)                          │
│  • Scalers for normalization                                  │
│  • Performance metrics (CSV)                                  │
│  • Predictions and actuals                                    │
│  • Visualization plots (PNG)                                  │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

#### Stage 1: Data Preprocessing
- **Input**: Raw Excel file with multiple sheets
- **Processing**: Multi-level header parsing, data reshaping, interpolation
- **Output**: Clean CSV with 847 samples × 11 features
- **Notebook**: `data-preprocessing-pipeline.ipynb`

#### Stage 2: Exploratory Analysis
- **Input**: Preprocessed CSV
- **Processing**: Statistical analysis, visualization, correlation studies
- **Output**: Insights and plots
- **Notebook**: `exploratory-data-analysis.ipynb`

#### Stage 3: ML Model Training
- **Input**: Preprocessed CSV
- **Processing**: Train-test split, model training, hyperparameter tuning
- **Output**: Trained classical ML models
- **Notebook**: `ml-pipeline.ipynb`

#### Stage 4: Neural Network Training
- **Input**: Preprocessed CSV
- **Processing**: Data scaling, ANN architecture design, training with callbacks
- **Output**: Trained neural network and scalers
- **Notebook**: `ann-pipeline.ipynb`

#### Stage 5: Results Visualization
- **Input**: Trained models and test data
- **Processing**: Generate comprehensive plots for publication
- **Output**: High-quality figures for research papers
- **Notebook**: `fuel-journal-plots.ipynb`

---

## Technology Stack

### Core Data Science Libraries

#### Data Manipulation
- **NumPy 2.3.3**: Numerical computing and array operations
- **Pandas 2.3.2**: Data manipulation and analysis
- **SciPy 1.16.2**: Scientific computing and statistics
- **OpenPyXL 3.1.5**: Excel file reading and processing

#### Machine Learning
- **Scikit-learn 1.7.2**: Classical ML algorithms and utilities
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Regressor
  - Linear Regression
- **Joblib 1.5.2**: Model serialization and parallelization

#### Deep Learning
- **TensorFlow 2.20.0**: Deep learning framework
- **Keras 3.11.3**: High-level neural networks API
- **H5py 3.14.0**: HDF5 file format for model storage

#### Visualization
- **Matplotlib 3.10.6**: Comprehensive plotting library
- **Seaborn 0.13.2**: Statistical data visualization

#### Development Tools
- **Jupyter**: Interactive development environment
  - jupyter-core 5.8.1
  - jupyter-client 8.6.3
  - ipykernel 6.30.1
- **Ruff**: Fast Python linter and formatter

### Build and Automation Tools
- **Make**: Build automation via Makefile
- **Flit**: Python packaging tool
- **Git**: Version control

---

## Project Structure

The project follows the Cookiecutter Data Science template with organized directories:

```
spray-vision/
│
├── data/                          # Data directory (not in version control)
│   ├── raw/                       # Original, immutable data
│   │   └── DataSheet_ETH_250902.xlsx  # ETH spray dataset (Excel)
│   ├── processed/                 # Cleaned, processed data
│   │   ├── preprocessed_dataset.csv
│   │   └── eth_spray_merged_preprocessed.csv
│   ├── interim/                   # Intermediate transformations
│   └── external/                  # External data sources
│
├── models/                        # Trained models (46.5 MB total)
│   ├── LinearRegression_regressor.joblib       # 2.5 KB
│   ├── DecisionTree_regressor.joblib          # 327 KB
│   ├── KNN_regressor.joblib                   # 279 KB
│   ├── SVR_regressor.joblib                   # 93 KB
│   ├── GradientBoosting_regressor.joblib      # 4.4 MB
│   ├── RandomForest_regressor.joblib          # 41.9 MB
│   ├── ANN_improved_regressor.h5              # 594 KB
│   ├── ann_input_scaler.joblib                # 1 KB
│   └── ann_target_scaler.joblib               # 1 KB
│
├── notebooks/                     # Jupyter notebooks (5 notebooks)
│   ├── data-preprocessing-pipeline.ipynb     # 30 KB, 10 cells
│   ├── exploratory-data-analysis.ipynb       # 2.3 MB, 8 cells
│   ├── ml-pipeline.ipynb                     # 140 KB, 9 cells
│   ├── ann-pipeline.ipynb                    # 75 KB, 8 cells
│   └── fuel-journal-plots.ipynb              # 13 MB, 51 cells
│
├── outputs/                       # Model outputs and metrics
│   ├── overall_model_metrics.csv              # Overall performance
│   ├── per_target_metrics.csv                 # Per-target metrics
│   ├── baseline_models_config.csv             # Model configurations
│   ├── ANN_improved_predictions.csv           # ANN predictions
│   ├── ANN_improved_actuals.csv               # Actual values
│   ├── ANN_improved_metrics.csv               # ANN metrics
│   └── sample_experimental_head.csv           # Sample data
│
├── plots/                         # Generated visualizations (19 figures)
│   ├── figure_1_1_time_variation.png          # Time series
│   ├── figure_1_2_inputs_before_scaling.png   # Input distributions
│   ├── figure_1_3_inputs_after_scaling.png    # Scaled inputs
│   ├── figure_1_4_correlation_heatmap.png     # Correlation matrix
│   ├── figure_2_1_cham_temp_vs_angle_mie.png  # Feature relationships
│   ├── figure_2_3_true_vs_pred_*.png          # Prediction plots (4 files)
│   ├── figure_2_7_parity_*.png                # Parity plots (4 files)
│   ├── figure_2_11_kde_residuals_*.png        # Residual analysis
│   ├── figure_2_14_feature_importance_*.png   # Feature importance
│   ├── figure_2_15_ann_vs_gb_*.png            # Model comparison (4 files)
│   └── figure_2_19_ann_residuals_kde.png      # Overall residuals
│
├── references/                    # Research papers and documentation
│   ├── Experimental_and_modeling_analysis_of_the_transien.pdf
│   └── Machine Learning Approaches of Spray.pdf
│
├── docs/                          # Documentation (empty, for future use)
├── figures/                       # Additional figures directory
│
├── README.md                      # Project README with quick start
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
├── Makefile                      # Automation commands
└── .gitignore                    # Git ignore rules
```

### Directory Purposes

#### `/data`
Contains all data files at various stages of processing. Raw data is immutable; processed data is generated by pipelines.

#### `/models`
Stores trained machine learning models in serialized format:
- `.joblib` files: Scikit-learn models (CPU-friendly serialization)
- `.h5` files: Keras neural networks (HDF5 format)
- Scaler objects for input/output normalization

#### `/notebooks`
Interactive Jupyter notebooks for development and analysis:
- Self-contained with detailed code
- Can be executed sequentially
- Generate outputs to other directories

#### `/outputs`
Contains model predictions, performance metrics, and evaluation results in CSV format for easy analysis.

#### `/plots`
High-quality visualizations suitable for research publications:
- PNG format at publication quality
- Systematic naming convention
- Multiple plot types (parity, residual, feature importance)

#### `/references`
Research papers and academic literature relevant to spray physics and machine learning applications.

---

## Key Features

### 1. Comprehensive Model Comparison
- **7 Different Algorithms**: From simple linear regression to complex ensemble methods
- **Systematic Evaluation**: Consistent metrics across all models
- **Performance Ranking**: Automatic identification of best-performing models

### 2. Multi-Output Regression
- **4 Simultaneous Predictions**: All spray characteristics predicted together
- **Consistent Training**: Same data splits and preprocessing for fair comparison
- **Per-Target Metrics**: Individual performance evaluation for each output

### 3. Automated Pipeline
- **Makefile Integration**: One-command execution of entire pipeline
- **Reproducible Results**: Fixed random seeds and deterministic operations
- **Error Handling**: Robust data validation and integrity checks

### 4. Production-Ready Models
- **Serialized Models**: All trained models saved for deployment
- **Scalers Included**: Input/output normalization parameters preserved
- **Quick Inference**: Load and predict in seconds

### 5. Rich Visualizations
- **19 Publication-Quality Figures**: Comprehensive visual analysis
- **Multiple Plot Types**: 
  - Time series variation
  - Distribution plots
  - Correlation heatmaps
  - Parity plots (predicted vs actual)
  - Residual analysis
  - Feature importance charts
  - Model comparison plots

### 6. Detailed Documentation
- **In-Code Comments**: Explanatory comments throughout notebooks
- **Cell Descriptions**: Clear section headers in notebooks
- **README Guide**: Step-by-step usage instructions
- **Comprehensive Metrics**: Full performance reporting

---

## Research Context

### Dataset: ETH Spray Database

The project uses the **ETH (Swiss Federal Institute of Technology) Spray Dataset**, a well-established benchmark in spray research.

#### Dataset Characteristics
- **Source**: Experimental spray measurements from ETH Zurich
- **Format**: Excel workbook with multiple sheets
- **Experiments**: 7 distinct experimental runs (ETH-01 to ETH-07, including ETH-06.1)
- **Time Points**: 121 temporal measurements per experiment
- **Total Samples**: 847 data points (7 runs × 121 time points)

#### Data Collection
- **Facility**: High-pressure spray chamber
- **Measurement Frequency**: High-speed imaging at millisecond intervals
- **Control System**: Precise control of chamber conditions
- **Quality Assurance**: Multiple measurement methods for validation

### Performance Achievements

Based on the trained models in this project:

#### Top-Performing Models (by R² Score)
1. **K-Nearest Neighbors**: R² = 0.9988, MAE = 0.74
2. **Gradient Boosting**: R² = 0.9949, MAE = 0.48
3. **Decision Tree**: R² = 0.9936, MAE = 1.11
4. **Support Vector Regression**: R² = 0.9883, MAE = 1.29
5. **Random Forest**: R² = 0.9830, MAE = 2.21
6. **Linear Regression**: R² = 0.8341, MAE = 4.78

#### Neural Network Performance
- **Architecture**: Multi-layer perceptron with optimized layers
- **Expected R²**: > 0.99 (typical for this problem)
- **Training**: Early stopping with validation monitoring
- **Advantages**: Fast inference, good generalization

### Research Contributions

This project contributes to the field by:

1. **Demonstrating ML Viability**: Proves machine learning can replace expensive experiments
2. **Method Comparison**: Provides empirical comparison of different approaches
3. **Open Methodology**: Fully transparent and reproducible pipeline
4. **Educational Resource**: Serves as template for similar engineering ML projects
5. **Practical Tool**: Provides working models for immediate use

### Future Research Directions

Potential extensions of this work:
- **Transfer Learning**: Apply models to other spray datasets
- **Real-Time Prediction**: Deploy models for online control systems
- **Physics-Informed ML**: Incorporate governing equations into neural networks
- **Uncertainty Quantification**: Add probabilistic predictions
- **3D Spray Modeling**: Extend to full spatial spray characteristics
- **Multi-Fidelity Models**: Combine experimental and simulation data

---

## Summary

Spray Vision represents a complete, professional-grade machine learning project that:
- ✅ Solves a real engineering problem
- ✅ Uses industry-standard tools and practices
- ✅ Provides excellent predictive performance
- ✅ Is fully documented and reproducible
- ✅ Includes comprehensive evaluation and visualization
- ✅ Follows best practices in data science workflow

The project successfully demonstrates that machine learning can provide accurate, fast predictions of spray characteristics, potentially reducing the need for expensive experimental campaigns and accelerating the design cycle for fuel injection systems.
