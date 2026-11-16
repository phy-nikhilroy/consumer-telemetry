# Convenience targets for the consumer telemetry pipeline.
# Every target runs through the project virtual environment in .venv/.

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
JUPYTER := $(VENV)/bin/jupyter

NOTEBOOKS := 04_transformer_ner 14_feature_eda 16_model_evaluation 20_model_evaluation_report

.PHONY: help setup venv-check test check run-pipeline simulate sessionize features models \
        pipeline notebooks report benchmark

help:
	@echo "Available commands:"
	@echo "  make setup         - Create .venv, install requirements, download the spaCy model"
	@echo "  make venv-check    - Verify the Python and pytest binaries exist"
	@echo "  make test          - Run the full pytest suite"
	@echo "  make check         - Show git status"
	@echo "  make run-pipeline  - Push one sample screen string through the ingestion pipeline"
	@echo "  make simulate      - Regenerate the simulated population and reload the warehouse"
	@echo "  make sessionize    - Sessionize events and detect cross-app comparisons"
	@echo "  make features      - Build RFM and behavioral features, the feature store, and selection"
	@echo "  make models        - Train and evaluate propensity, switching, CLV, and segmentation"
	@echo "  make pipeline      - Run simulate, sessionize, features, and models in order"
	@echo "  make notebooks     - Execute every notebook in place"
	@echo "  make report        - Export the evaluation report notebook to HTML"
	@echo "  make benchmark     - Run the real-data extraction benchmark (needs data/external/)"

setup:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

venv-check:
	@test -f $(PYTHON) && echo "Python binary verified: $(PYTHON)" || (echo "Missing $(PYTHON)" && exit 1)
	@test -f $(PYTEST) && echo "Pytest binary verified: $(PYTEST)" || (echo "Missing $(PYTEST)" && exit 1)

test:
	$(PYTEST) -v

check:
	git status

run-pipeline:
	$(PYTHON) -c "from src.ingestion.pipeline import process; print(process('Samsung Galaxy S24 Ultra 256GB Titanium \$$1,299.99 4.7★ Add to Cart').model_dump_json(indent=2))"

simulate:
	$(PYTHON) src/ingestion/simulate_users.py --users 400 --days 30

sessionize:
	$(PYTHON) src/features/sessionizer.py

features:
	$(PYTHON) src/features/rfm_features.py
	$(PYTHON) src/features/behavioral_features.py
	$(PYTHON) src/features/feature_store.py
	$(PYTHON) src/features/feature_validation.py

models:
	$(PYTHON) src/models/propensity.py
	$(PYTHON) src/models/evaluation.py
	$(PYTHON) src/models/switching.py
	$(PYTHON) src/models/clv.py
	$(PYTHON) src/models/segmentation.py

pipeline: simulate sessionize features models

notebooks:
	@for nb in $(NOTEBOOKS); do \
		$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/$$nb.ipynb || exit 1; \
	done

report:
	$(JUPYTER) nbconvert --to html notebooks/20_model_evaluation_report.ipynb

benchmark:
	$(PYTHON) src/ingestion/real_data_benchmark.py
