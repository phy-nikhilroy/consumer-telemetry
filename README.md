# Consumer Data ML Pipeline (Consumer Telemetry)

A cross-app consumer telemetry platform that ingests raw screen text, extracts product entities, models consumer behavior, and serves predictions via an API.

## Project Structure

```
├── data/
│   ├── raw/           # Simulated screen dumps
│   ├── processed/     # Cleaned & extracted data
│   └── features/      # Feature matrices
├── src/
│   ├── ingestion/     # Week 1: NLP extraction
│   ├── features/      # Week 2: Feature engineering
│   ├── models/        # Week 3: ML models
│   └── api/           # Week 4: FastAPI service
├── notebooks/         # Exploration & evaluation
├── tests/             # Unit and integration tests
├── docker/            # Container specifications
└── README.md
```

## Setup

1. **Virtual Environment**:
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
