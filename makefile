# Define variables for common paths and arguments
URL=https://archive.ics.uci.edu/static/public/222/data.csv
RAW_DATA_PATH=data/raw
PROCESSED_DATA_PATH=data/processed
MODELS_DIR=results/models
FIGURES_DIR=results/figures
SEED=522
CONTAINER_NAME=jupyter-lab

# Phony targets for commands without file outputs
.PHONY: all up data_download split_process eda fit_classifier feature_importance build_report

# Default target
all: up data_download split_process eda fit_classifier feature_importance build_report

# Target to start the Docker containers
up:
	docker compose up -d

# Target for downloading data
data_download:
	docker compose exec $(CONTAINER_NAME) /bin/bash -c "\
	cd work && \
	python scripts/data_download.py \
	--url=$(URL) \
	--save_path=$(RAW_DATA_PATH)\
	"

# Target for splitting and processing data
split_process: data_download
	docker compose exec $(CONTAINER_NAME) /bin/bash -c "\
	cd work && \
	python scripts/split_and_process.py \
	--raw_data=$(RAW_DATA_PATH)/bank-full.csv \
	--save_to=$(PROCESSED_DATA_PATH) \
	--preprocessor_to=$(MODELS_DIR) \
	--seed=$(SEED) \
	"

# Target for EDA plots
eda: split_process
	docker compose exec $(CONTAINER_NAME) /bin/bash -c "\
	cd work && \
	python scripts/eda.py \
	--training_data=$(PROCESSED_DATA_PATH)/bank_train.csv \
	--save_plot_to=$(FIGURES_DIR) \
	"

# Target for fitting the classifier
fit_classifier: eda
	docker compose exec $(CONTAINER_NAME) /bin/bash -c "\
	cd work && \
	python scripts/fit_bank_classifier.py \
	--resampled_training_data=$(PROCESSED_DATA_PATH)/X_train_resmp.csv \
	--resampled_training_response=$(PROCESSED_DATA_PATH)/y_train_resmp.csv \
	--test_data=$(PROCESSED_DATA_PATH)/X_test.csv \
	--test_response=$(PROCESSED_DATA_PATH)/y_test.csv \
	--preprocessor_pipe=$(MODELS_DIR)/bank_preprocessor.pickle \
	--save_pipelines_to=$(MODELS_DIR) \
	--save_plot_to=$(FIGURES_DIR) \
	--seed=$(SEED) \
	"

# Target for feature importance
feature_importance: fit_classifier
	docker compose exec $(CONTAINER_NAME) /bin/bash -c "\
	cd work && \
	python scripts/feat_imp.py \
	--transformed_training_data=$(PROCESSED_DATA_PATH)/X_train_trans.csv \
	--pipeline_model=$(MODELS_DIR)/logistic_pipeline.pickle \
	--save_plot_to=$(FIGURES_DIR) \
	--seed=$(SEED) \
	"

# Target for building the report
build_report: feature_importance
	docker compose exec $(CONTAINER_NAME) /bin/bash -c " \
	cd work && \
	jupyter-book build report && \
	cp -r report/_build/html/* docs"

clean:
	# Remove the generated files and directories
	rm -rf $(PROCESSED_DATA_PATH)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(FIGURES_DIR)/*
	rm -rf docs/*

	# Stop and remove the Docker container
	docker compose down