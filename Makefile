# Convenience targets for the consumer telemetry pipeline.
# Every target runs through the project virtual environment in .venv/.

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PYTEST    := $(VENV)/bin/pytest
JUPYTER   := $(VENV)/bin/jupyter
UVICORN   := $(VENV)/bin/uvicorn
STREAMLIT := $(VENV)/bin/streamlit
API_URL   ?= http://localhost:8000
VERSION   ?= 1.1

NOTEBOOKS := 04_transformer_ner 14_feature_eda 16_model_evaluation 20_model_evaluation_report

.PHONY: gate help setup venv-check test check run-pipeline simulate sessionize features models \
        pipeline notebooks report benchmark serving-db serving-reset api dashboard stream \
        monitor data-checks api-benchmark docker-build compose-up compose-down

help:
	@echo "Available commands:"
	@echo "  make setup         - Create .venv, install requirements, download the spaCy model"
	@echo "  make venv-check    - Verify the Python and pytest binaries exist"
	@echo "  make test          - Run the full pytest suite"
	@echo "  make gate          - Run release gates: make gate VERSION=1.1"
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
	@echo ""
	@echo "Serving:"
	@echo "  make serving-db    - Create the API's serving warehouse if it is missing"
	@echo "  make serving-reset - Rebuild the serving warehouse, discarding live events"
	@echo "  make api           - Run the FastAPI service on port 8000 with auto-reload"
	@echo "  make dashboard     - Run the Streamlit dashboard on port 8501"
	@echo "  make stream        - Stream simulated events to the API (API_URL=...)"
	@echo "  make monitor       - Print live API activity to the terminal (API_URL=...)"
	@echo "  make api-benchmark - Measure p50/p95/p99 latency per endpoint, in process"
	@echo "  make docker-build  - Build the API and dashboard images"
	@echo "  make compose-up    - Start API, dashboard, and stream with Docker Compose"
	@echo "  make compose-down  - Stop the Compose stack"

setup:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

data-checks:
	$(PYTHON) src/features/data_checks.py

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

serving-db:
	$(PYTHON) src/api/warehouse.py

serving-reset:
	$(PYTHON) src/api/warehouse.py --reset

api:
	$(UVICORN) src.api.main:app --reload --port 8000

dashboard:
	CT_API_URL=$(API_URL) $(STREAMLIT) run src/dashboard/app.py

stream:
	$(PYTHON) src/simulation/stream_generator.py --api-url $(API_URL)

monitor:
	$(PYTHON) src/simulation/monitor.py --api-url $(API_URL)

api-benchmark:
	$(PYTHON) src/api/benchmark.py --requests 100

docker-build:
	docker build -t consumer-telemetry-api -f docker/Dockerfile .
	docker build -t consumer-telemetry-dashboard -f docker/Dockerfile.dashboard .

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

gate:
	$(PYTHON) src/release/gates.py --version $(VERSION)
