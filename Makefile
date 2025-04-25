#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Employee-Attrition
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies with pip
.PHONY: requirements-pip
requirements-pip:
	pip install -e .

## Install Python dependencies with Poetry
.PHONY: requirements
requirements:
	poetry install

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Start MLflow tracking server
.PHONY: mlflow
mlflow:
	mlflow ui --port 5000

## Run hyperparameter optimization for logistic regression
.PHONY: hpo-logreg
hpo-logreg:
	python scripts/hpo.py --model-type logistic_regression

## Run hyperparameter optimization for random forest
.PHONY: hpo-rf
hpo-rf:
	python scripts/hpo.py --model-type random_forest

## Train final model with best parameters
.PHONY: train
train:
	python scripts/train.py

## Train and register production model
.PHONY: train-prod
train-prod:
	python scripts/train.py --register-as AttritionProductionModel

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