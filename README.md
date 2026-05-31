# IntelliProfile (IntelliX)

**ML-powered multi-language code performance profiler** — paste code, get instant complexity metrics, ML predictions, and optimization recommendations.

[![CI](https://github.com/DeepxD-code/IntelliX/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepxD-code/IntelliX/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-89%20passing-brightgreen)

## Features

- **3 languages**: Python (AST-based), C++, Java
- **4 ML classifiers**: Random Forest, Gradient Boosting, SVM, Neural Network (auto-selects best via GridSearchCV)
- **Metrics**: execution time, memory usage, loop depth, nesting depth, function calls, conditionals, complexity score (+ cyclomatic complexity for Java)
- **Prediction**: EFFICIENT / MODERATE / NEEDS_OPTIMIZATION with confidence & class probabilities
- **Config-driven**: all thresholds, model params, dataset size via `config.yaml`
- **Model versioning**: ModelRegistry with version tracking and rollback support
- **Structured logging**: JSON-formatted logs for production monitoring
- **Web UI**: Flask frontend with code editor, live results, probability bars
- **REST API**: Swagger-documented endpoints at `/api/docs`
- **Docker support**: single-command deployment

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Web UI     │────▶│  Flask API   │────▶│  CodeProfiler    │
│  index.html │     │  app.py      │     │  (unified entry) │
└─────────────┘     └──────────────┘     └───────┬──────────┘
                         │                       │
                         ▼                       ▼
                  ┌──────────────┐     ┌──────────────────┐
                  │  SQLite DB   │     │  Language-Specific│
                  │  history     │     │  Profilers        │
                  └──────────────┘     │  (AST / heuristic)│
                                       └───────┬──────────┘
                                               │
                                               ▼
                                        ┌──────────────────┐
                                        │  MLPipeline      │
                                        │  4 classifiers   │
                                        │  + GridSearchCV  │
                                        └───────┬──────────┘
                                               │
                                        ┌──────▼───────┐
                                        │  ModelRegistry│
                                        │  versioned    │
                                        │  model_v*.pkl │
                                        └──────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Test accuracy | 93-100% (varies by classifier) |
| 5-fold CV accuracy | 91-96% |
| Profile latency | < 50ms (typical) |
| Batch throughput | 200+ profiles/sec |
| Training time (500 samples, GridSearch) | ~10-30s |
| Model size | ~2-10 MB |

## Quick Start

### Local

```bash
pip install -r requirements.txt
python profiler_main.py     # Train model + CLI demo
python app.py               # Start web UI at http://localhost:5000
```

### Docker

```bash
docker compose build
docker compose up -d
# -> http://localhost:5000
```

## Project Structure

```
IntelliX/
├── app.py                 # Flask web app (7 API endpoints + Swagger)
├── profiler_main.py       # Core: AST analyzer, ML pipeline, profiler, config
├── config.yaml            # YAML configuration (thresholds, ML, logging)
├── test_profiler.py       # Unit tests (89 tests)
├── test_app.py            # API integration tests
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image
├── docker-compose.yml     # Orchestration
├── Makefile               # make test / make run / make docker-build
├── pyproject.toml         # Ruff linter config
├── .github/workflows/ci.yml   # GitHub Actions CI
├── templates/
│   └── index.html         # Web UI template
├── static/
│   └── style.css          # Web UI styles
├── models/                # Saved ML models (gitignored)
├── logs/                  # JSON log output (gitignored)
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check + model info |
| POST | `/api/profile` | Profile single code snippet |
| POST | `/api/batch-profile` | Profile multiple snippets |
| GET | `/api/model-info` | Model details & features |
| GET | `/api/history` | Recent profile history (SQLite) |
| GET | `/api/docs` | Swagger API documentation |

### Profile Request

```json
POST /api/profile
{
  "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
  "language": "python"
}
```

### Profile Response

```json
{
  "metrics": {
    "execution_time_ms": 5.23,
    "memory_usage_kb": 640,
    "loop_depth": 0,
    "complexity_score": 3.42,
    ...
  },
  "ml_prediction": {
    "label": "EFFICIENT",
    "confidence": 0.97,
    "probabilities": {
      "EFFICIENT": 0.97,
      "MODERATE": 0.02,
      "NEEDS_OPTIMIZATION": 0.01
    }
  },
  "recommendations": [
    "Code appears well-structured and efficient.",
    "Consider adding type hints for better maintainability."
  ],
  "analysis_time_ms": 1.23
}
```

## ML Pipeline

- **4 classifiers**: Random Forest, Gradient Boosting, SVM, MLP (Neural Network)
- **Hyperparameter tuning**: GridSearchCV with 3-fold cross-validation
- **Training data**: 500 synthetic samples (balanced across 3 complexity classes)
- **7 features**: execution_time_ms, memory_usage_kb, loop_depth, max_nesting_depth, function_calls, conditionals, complexity_score
- **Classes**: EFFICIENT, MODERATE, NEEDS_OPTIMIZATION
- **Persistence**: versioned model files (`model_v<timestamp>.joblib + scaler + metadata`)
- **ModelRegistry**: list versions, get latest, rollback support

## Configuration

All settings in `config.yaml`:

```yaml
ml_model:
  classifier_type: "random_forest"   # or gradient_boosting, svm, neural_network
  hyperparameter_tuning: true
  cross_validation_folds: 5

dataset:
  n_samples: 500

thresholds:
  execution_time_ms: 100.0
  loop_depth: 3

server:
  host: "0.0.0.0"
  port: 5000
```

Override with `CONFIG_PATH` env var.

## Testing

```bash
# All tests
make test

# Unit tests only
python test_profiler.py

# API integration tests only
python test_app.py

# Lint
make lint
```

**89 unit tests** covering:
- Python AST analyzer (15 visitor methods)
- Dataset generation
- ML pipeline (4 classifiers + GridSearchCV + save/load/registry)
- ConfigManager (validation, dot-notation, edge cases)
- ModelRegistry (list versions, get latest)
- Recommendation engine
- Code profiler (all 3 languages + edge cases)
- 1500 bulk predictions (500 per language)

**12 API tests** covering:
- Health check
- Single profile (Python, C++, Java, unsupported languages)
- Batch profile (mixed success/failure)
- Model info
- History (SQLite persistence)
- Index page

## License

MIT
