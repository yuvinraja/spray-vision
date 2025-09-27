#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = spray-vision
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Directories
DATA_DIR = data
MODELS_DIR = models
NOTEBOOKS_DIR = notebooks
OUTPUTS_DIR = outputs

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install development dependencies
.PHONY: dev-requirements
dev-requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev,viz,extra]"

## Update requirements.txt from current environment
.PHONY: freeze-requirements
freeze-requirements:
	$(PYTHON_INTERPRETER) -m pip freeze > requirements.txt
	



## Delete all compiled Python files and temporary files
.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@echo "Cleaning on Windows..."
	@for /r . %%i in (*.pyc) do @del "%%i" 2>nul || echo >nul
	@for /d /r . %%i in (__pycache__) do @rmdir /s /q "%%i" 2>nul || echo >nul
	@for /d /r . %%i in (.ipynb_checkpoints) do @rmdir /s /q "%%i" 2>nul || echo >nul
else
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
	find . -name "*.tmp" -delete
endif


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check $(NOTEBOOKS_DIR)
	ruff check $(NOTEBOOKS_DIR)

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix $(NOTEBOOKS_DIR)
	ruff format $(NOTEBOOKS_DIR)

## Run Jupyter notebooks for data preprocessing
.PHONY: preprocess
preprocess: requirements
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/data-preprocessing-pipeline.ipynb --output-dir=$(OUTPUTS_DIR)

## Run ML pipeline notebooks
.PHONY: train-ml
train-ml: requirements
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/ml-pipeline.ipynb --output-dir=$(OUTPUTS_DIR)

## Run ANN pipeline notebook
.PHONY: train-ann
train-ann: requirements
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/ann-pipeline.ipynb --output-dir=$(OUTPUTS_DIR)

## Run complete ML pipeline (preprocessing + training)
.PHONY: train-all
train-all: preprocess train-ml train-ann

## Start Jupyter notebook server
.PHONY: notebook
notebook: requirements
	$(PYTHON_INTERPRETER) -m jupyter notebook $(NOTEBOOKS_DIR)

## Test notebook execution without saving outputs
.PHONY: test-notebooks
test-notebooks: requirements
	@echo "Testing data preprocessing notebook..."
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/data-preprocessing-pipeline.ipynb --stdout > /dev/null
	@echo "Testing ML pipeline notebook..."
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/ml-pipeline.ipynb --stdout > /dev/null
	@echo "Testing ANN pipeline notebook..."
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/ann-pipeline.ipynb --stdout > /dev/null
	@echo "✓ All notebooks executed successfully"

## Full project setup
.PHONY: setup
setup: setup-dirs requirements check-data
	@echo "✓ Project setup complete"





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Verify data files exist
.PHONY: check-data
check-data:
	@echo "Checking for required data files..."
	@test -f $(DATA_DIR)/raw/DataSheet_ETH_250902.xlsx || (echo "ERROR: Raw data file not found" && exit 1)
	@echo "✓ Raw data file found"

## Create necessary directories
.PHONY: setup-dirs
setup-dirs:
	mkdir -p $(DATA_DIR)/processed $(DATA_DIR)/interim $(MODELS_DIR) $(OUTPUTS_DIR)
	@echo "✓ Project directories created"

## Validate model files
.PHONY: check-models
check-models:
	@echo "Checking trained models..."
	@ls -la $(MODELS_DIR)/ || echo "No models found yet"

## Generate project report
.PHONY: report
report:
	@echo "=== Spray Vision ML Project Status ==="
	@echo "Data files:"
	@ls -la $(DATA_DIR)/raw/ 2>/dev/null || echo "  No raw data"
	@ls -la $(DATA_DIR)/processed/ 2>/dev/null || echo "  No processed data"
	@echo "Models:"
	@ls -la $(MODELS_DIR)/ 2>/dev/null || echo "  No models"
	@echo "Outputs:"
	@ls -la $(OUTPUTS_DIR)/ 2>/dev/null || echo "  No outputs"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
