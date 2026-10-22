# Email Spam Classifier 📧

A full MLOps pipeline for email spam classification — built to demonstrate end-to-end ML engineering practices including data versioning, experiment tracking, containerized serving, CI/CD, and caching. Designed to reflect real-world engineering workflows rather than a simple notebook model.

## Tech Stack

| Layer | Tool |
|-------|------|
| ML & Preprocessing | scikit-learn, NLTK, BeautifulSoup4 |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + Uvicorn |
| Caching | Redis |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Load Testing | Locust |
| Version Control | Git + GitHub |

## Project Architecture

```
email_classifier/
├── data/
│   └── raw/                  # SpamAssassin dataset (tracked by DVC)
├── notebooks/
│   └── eda_01.ipynb          # Exploratory data analysis & model training
├── src/
│   ├── data_prep.py          # Custom sklearn transformers (pipeline)
│   ├── fetch_data.py         # Data download script
│   └── app.py                # FastAPI serving application
├── tests/
│   └── test_data_prep.py     # Unit tests (pytest)
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD pipeline
├── Dockerfile                # FastAPI container
├── Dockerfile.mlflow         # MLflow server container
├── docker-compose.yml        # Orchestrates all services
├── locustfile.py             # Load testing configuration
├── requirements.txt          # Production dependencies
└── data/raw.dvc              # DVC pointer to raw data
```

## How It Works

```
Raw Email Text
      ↓
EmailToWordCountTransformer
(strip headers · lowercase · replace URLs/numbers · stemming)
      ↓
WordCounterToVectorTransformer
(top-1000 vocabulary → sparse matrix)
      ↓
LogisticRegression
      ↓
{ "label": "spam", "is_spam": true }
```

## Quick Start

### Prerequisites
- Docker & Docker Compose

### Run with Docker Compose
```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier
docker compose up
```

| Service | URL |
|---------|-----|
| API | http://127.0.0.1:8000 |
| API Docs (Swagger) | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

### Test the API
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "CLICK HERE to win FREE money now!!!"}'
```

```json
{"label": "spam", "is_spam": true}
```

### Run Tests
```bash
pip install pytest
pytest tests/ -v
```

### Load Testing
```bash
pip install locust
locust --host=http://127.0.0.1:8000
```
Open http://localhost:8089 to run the load test from the browser.

## MLOps Pipeline

### 1. Data Versioning (DVC)
Raw email data from [Apache SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/) is tracked with DVC — keeping Git lightweight while maintaining full data reproducibility.

```bash
dvc pull   # fetch raw data
dvc push   # push new data versions
```

### 2. Experiment Tracking (MLflow)
Every training run is logged with MLflow — hyperparameters, metrics, and serialized model artifacts.

| Metric | Value |
|--------|-------|
| Precision | ~96% |
| Recall | ~94% |

### 3. Model Serving (FastAPI + Redis)
A REST API exposes the trained model for real-time inference. Redis caches prediction results to reduce load and latency — with graceful degradation if Redis is unavailable.

```
POST /predict  →  { "text": "..." }  →  { "label": "spam/ham", "is_spam": true/false }
GET  /         →  { "status": "ok" }
GET  /docs     →  Swagger UI
```

### 4. Containerization (Docker Compose)
Three isolated services communicate over Docker's internal network:

```
[FastAPI :8000] ──→ [MLflow :5000]
      ↕
[Redis :6379]
      ↑
 your browser
```

### 5. CI/CD (GitHub Actions)
Every push triggers an automated pipeline:
1. Run unit tests (pytest)
2. Build Docker images (FastAPI + MLflow)

If tests fail, the Docker build is skipped.

### 6. Load Testing (Locust)
Tested at **405 concurrent users** with:
- **~196 RPS** (requests per second)
- **5ms** median response time
- **0% failure rate**

## Project Stages

| Stage | Branch | Description |
|-------|--------|-------------|
| Setup | `main` | Project structure, virtual environment |
| Data versioning | `feature/dvc-setup` | DVC init, remote storage, data tracking |
| Experiment tracking | `feature/mlflow-tracking` | MLflow integration, model logging |
| Model serving | `feature/fastapi-serving` | FastAPI endpoint, pipeline serving |
| Containerization | `feature/docker` | Dockerfile, docker-compose, multi-service setup |
| CI/CD | `feature/ci-cd` | GitHub Actions, pytest, Docker build check |
| Caching | `feature/redis-caching` | Redis caching with graceful degradation |

## Dataset

[Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- `easy_ham` — ~2500 legitimate emails
- `spam` — ~500 spam emails
