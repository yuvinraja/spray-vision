# Spray Vision: ML-Based Spray Characteristics Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive machine learning project for predicting spray characteristics using various classical ML approaches and artificial neural networks.

## Project Overview

This project implements multiple machine learning models to predict spray characteristics based on experimental data. The system compares the performance of various algorithms including traditional ML methods and deep learning approaches to identify the most effective model for spray prediction.

## Architecture

The project follows a structured data science workflow:

- **Data Processing**: ETH spray dataset preprocessing and feature engineering
- **Model Training**: Multiple ML algorithms including ANN, Random Forest, Gradient Boosting, SVM, etc.
- **Model Evaluation**: Comprehensive performance metrics and comparison
- **Prediction Pipeline**: End-to-end prediction system with trained models

## Models Implemented

- **Artificial Neural Network (ANN)** - Standard and improved versions
- **Random Forest Regressor**
- **Gradient Boosting Regressor**  
- **Support Vector Regression (SVR)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**
- **Linear Regression**

## Project Structure

```
├── data/
│   ├── raw/                    # Original ETH spray dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── interim/                # Intermediate data transformations
├── notebooks/
│   ├── data-preprocessing-pipeline.ipynb    # Data cleaning and preprocessing
│   ├── ml-pipeline.ipynb                    # Classical ML models
│   └── ann-pipeline.ipynb                   # Neural network implementation
├── models/                     # Trained model files and scalers
├── outputs/                    # Model predictions and performance metrics
├── reports/                    # Analysis reports and documentation
└── references/                 # Research papers and documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spray-vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Use

### Quick Start (Automated with Makefile)

The project includes a comprehensive Makefile for easy automation. Here are the key commands:

#### **Initial Setup**
```bash
# Complete project setup (creates directories, installs dependencies, checks data)
make setup

# Install development dependencies with optional features
make dev-requirements
```

#### **Data Processing & Training**
```bash
# Run data preprocessing pipeline
make preprocess

# Train classical ML models
make train-ml

# Train artificial neural network
make train-ann

# Run complete pipeline (preprocessing + all training)
make train-all
```

#### **Development & Testing**
```bash
# Start Jupyter notebook server
make notebook

# Test all notebooks without saving outputs
make test-notebooks

# Check project status and files
make report

# Clean temporary files and cache
make clean
```

#### **Dependency Management**
```bash
# Install basic requirements
make requirements

# Update requirements.txt from current environment
make freeze-requirements
```

### Manual Usage (Step-by-Step)

If you prefer to run things manually or need more control:

#### **1. Environment Setup**
```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Or install with optional features
pip install -e ".[dev,viz,extra]"
```

#### **2. Data Preparation**
```bash
# Ensure raw data exists
# Place DataSheet_ETH_250902.xlsx in data/raw/

# Create necessary directories
mkdir -p data/processed data/interim models outputs

# Run preprocessing notebook
jupyter nbconvert --to notebook --execute notebooks/data-preprocessing-pipeline.ipynb
```

#### **3. Model Training**

**Classical ML Models:**
```bash
# Execute ML pipeline notebook
jupyter nbconvert --to notebook --execute notebooks/ml-pipeline.ipynb

# This will generate:
# - Trained model files in models/
# - Performance metrics in outputs/
```

**Neural Network:**
```bash
# Execute ANN pipeline notebook
jupyter nbconvert --to notebook --execute notebooks/ann-pipeline.ipynb

# This will generate:
# - ANN model files (.h5 format)
# - Scaler files for input/output normalization
# - Performance comparisons
```

#### **4. Interactive Development**
```bash
# Start Jupyter for interactive development
jupyter notebook notebooks/

# Or use Jupyter Lab
jupyter lab notebooks/
```

### Project Workflow

1. **Data Preprocessing** (`data-preprocessing-pipeline.ipynb`)
   - Loads raw ETH spray dataset from Excel
   - Performs data cleaning and preprocessing
   - Saves processed data to `data/processed/`

2. **Classical ML Training** (`ml-pipeline.ipynb`)
   - Trains multiple ML algorithms
   - Performs hyperparameter tuning
   - Evaluates and compares models
   - Saves best models to `models/`

3. **Neural Network Training** (`ann-pipeline.ipynb`)
   - Implements standard and improved ANN architectures
   - Trains with proper validation splits
   - Saves trained networks and scalers
   - Generates prediction comparisons

4. **Results Analysis**
   - Review `outputs/overall_model_metrics.csv` for model comparison
   - Check `outputs/per_target_metrics.csv` for detailed performance
   - Examine actual vs predicted values in output CSV files

### Troubleshooting

**Common Issues:**

- **Missing data file**: Ensure `DataSheet_ETH_250902.xlsx` is in `data/raw/`
- **Import errors**: Run `make requirements` or `pip install -r requirements.txt`
- **Notebook execution fails**: Try `make test-notebooks` to identify issues
- **Memory issues**: Close other applications, consider smaller batch sizes in ANN training

**Getting Help:**
```bash
# See all available Makefile commands
make help

# Check project status
make report

# Verify data and model files
make check-data
make check-models
```

## Results & Outputs

After running the complete pipeline, you'll find several important files:

### Model Files (`models/` directory)
- **Classical ML Models**: `*.joblib` files (RandomForest, SVM, etc.)
- **Neural Networks**: `*.h5` files for trained ANN models
- **Scalers**: `*_scaler.joblib` files for input/output normalization

### Performance Metrics (`outputs/` directory)
- **`overall_model_metrics.csv`**: Comprehensive comparison of all models
- **`per_target_metrics.csv`**: Detailed performance per prediction target
- **`ANN_improved_predictions.csv`**: Best model predictions
- **`ANN_improved_actuals.csv`**: Corresponding actual values

### Interpreting Results
- **R² Score**: Higher is better (closer to 1.0 indicates better fit)
- **MAE (Mean Absolute Error)**: Lower is better (average prediction error)
- **MSE (Mean Squared Error)**: Lower is better (penalizes larger errors more)

### Expected Performance
The project typically achieves:
- Classical ML models: R² scores ranging from 0.7-0.9
- Neural Networks: Often the best performing with R² > 0.9
- Best models are automatically saved for future use

## Configuration

The project uses:
- `pyproject.toml` for package configuration
- `Makefile` for automation tasks
- Joblib for model serialization
- HDF5 format for neural network models

## Documentation

Detailed analysis and results are available in the `reports/` directory, including:
- Review reports with comprehensive analysis
- Performance comparisons and insights

## Documentation

Comprehensive documentation is available in multiple guides:

### 📚 Complete Documentation Suite

1. **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Complete project overview
   - What is Spray Vision?
   - Project goals and objectives
   - Scientific background
   - Architecture and technology stack
   - Research context

2. **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Technical specifications
   - System architecture details
   - Data processing pipeline
   - Machine learning models
   - Neural network architecture
   - API reference and configuration

3. **[NOTEBOOKS_GUIDE.md](NOTEBOOKS_GUIDE.md)** - Detailed notebook documentation
   - All 5 notebooks explained in detail
   - Cell-by-cell breakdown
   - Execution order and dependencies
   - Tips and best practices

4. **[DATA_PIPELINE.md](DATA_PIPELINE.md)** - Data pipeline documentation
   - Raw data specification
   - Data extraction and transformation
   - Quality assurance procedures
   - Processed data format

5. **[MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)** - Model documentation
   - All 7 models detailed
   - Performance comparison
   - Deployment guidelines
   - Model selection guide

6. **[USER_GUIDE.md](USER_GUIDE.md)** - Practical user guide
   - Quick start tutorial
   - Step-by-step instructions
   - Common tasks and workflows
   - Troubleshooting and FAQ

### Quick Links

- **Getting Started**: See [USER_GUIDE.md](USER_GUIDE.md) Quick Start section
- **Understanding the Models**: See [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)
- **Deploying Models**: See [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md#deployment-guide)
- **Technical Details**: See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

## License

This project is structured using the Cookiecutter Data Science template.