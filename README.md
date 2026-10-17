# Email Spam Classifier 📧

A production-grade spam classification system built as a full MLOps portfolio project. The goal was not just to train a model, but to build the complete engineering infrastructure around it — data versioning, experiment tracking, model serving, and containerized deployment.

## Tech Stack

| Layer | Tool |
|-------|------|
| ML & Preprocessing | scikit-learn, NLTK, BeautifulSoup4 |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
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
├── Dockerfile                # FastAPI container
├── Dockerfile.mlflow         # MLflow server container
├── docker-compose.yml        # Orchestrates both services
├── requirements.txt          # Production dependencies
└── data/raw.dvc              # DVC pointer to raw data
```

## How It Works

```
Raw Email Text
      ↓
EmailToWordCountTransformer   (strip headers, lowercase, replace URLs/numbers, stemming)
      ↓
WordCounterToVectorTransformer  (top-1000 vocabulary → sparse matrix)
      ↓
LogisticRegression
      ↓
{ "label": "spam", "is_spam": true }
```

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- MLflow tracking database (`mlflow.db`) in project root

### Run with Docker Compose
```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier
docker compose up
```

The API will be available at `http://127.0.0.1:8000`

### Test the API
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "CLICK HERE to win FREE money now!!!"}'
```

Expected response:
```json
{"label": "spam", "is_spam": true}
```

### API Documentation
Interactive Swagger UI available at: `http://127.0.0.1:8000/docs`

## MLOps Pipeline

### 1. Data Versioning (DVC)
Raw email data from [Apache SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/) is tracked with DVC, keeping the Git repository lightweight while maintaining full data reproducibility.

```bash
dvc pull   # fetch raw data from remote storage
dvc push   # push new data versions
```

### 2. Experiment Tracking (MLflow)
Every training run is logged with MLflow — hyperparameters, metrics (precision/recall), and the serialized model artifact.

- **Precision:** ~96%
- **Recall:** ~94%

### 3. Model Serving (FastAPI)
A REST API exposes the trained model for real-time inference. The pipeline and model are loaded once at startup from MLflow.

```
POST /predict  →  { "text": "..." }  →  { "label": "spam/ham", "is_spam": true/false }
GET  /         →  { "status": "ok" }
GET  /docs     →  Swagger UI
```

### 4. Containerization (Docker Compose)
Two isolated services communicate over Docker's internal network:

```
[FastAPI container :8000] ──→ [MLflow container :5000]
         ↑
    your browser
```

## Project Stages

| Stage | Branch | Description |
|-------|--------|-------------|
| Setup | `main` | Project structure, .gitignore, virtual environment |
| Data versioning | `feature/dvc-setup` | DVC init, remote storage, data tracking |
| Experiment tracking | `feature/mlflow-tracking` | MLflow integration, model & pipeline logging |
| Model serving | `feature/fastapi-serving` | FastAPI endpoint, preprocessing pipeline serving |
| Containerization | `feature/docker` | Dockerfile, docker-compose, multi-container setup |

## Dataset

[Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- `easy_ham`: ~2500 legitimate emails
- `spam`: ~500 spam emails
