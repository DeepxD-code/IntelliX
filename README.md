# IntelliProfile (Intellix)

**ML-powered multi-language code performance profiler** — paste code, get instant complexity metrics, ML predictions, and optimization recommendations.

## Features

- **3 languages**: Python (AST-based), C++, Java
- **ML model**: Random Forest (200 trees, 12 depth) trained on synthetic labeled dataset
- **Metrics**: execution time, memory usage, loop depth, nesting depth, function calls, conditionals, complexity score (+ cyclomatic complexity for Java)
- **Prediction**: EFFICIENT / MODERATE / NEEDS_OPTIMIZATION with confidence & class probabilities
- **Recommendations**: contextual optimization suggestions per prediction
- **Web UI**: Flask frontend with code editor, live results, and batch profiling
- **REST API**: `/api/profile`, `/api/batch-profile`, `/api/health`, `/api/model-info`
- **Model persistence**: auto-saves to `models/`, loads latest on restart

## Quick Start

```bash
# Install dependencies
pip install scikit-learn joblib pandas numpy flask

# Train model & run CLI demo
python profiler_main.py

# Start web UI
python app.py
# -> http://localhost:5000
```

## Project Structure

```
IntelliX/
├── app.py                 # Flask web app (frontend + API)
├── profiler_main.py       # Core: AST analyzer, ML pipeline, profiler
├── test_profiler.py       # Comprehensive test suite (75 tests)
├── cpp_profiler.cpp.txt   # C++ profiler module (legacy)
├── java_profiler.java     # Java profiler module (legacy)
├── templates/
│   └── index.html         # Web UI template
├── static/
│   └── style.css          # Web UI styles
├── models/                # Saved ML models (gitignored)
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
| GET | `/api/history` | Recent profile results |

### Profile Request

```json
POST /api/profile
{
  "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
  "language": "python"
}
```

## ML Model

- **Type**: RandomForestClassifier (n_estimators=200, max_depth=12)
- **Training data**: 150 synthetic samples (50 per class)
- **Features**: execution_time_ms, memory_usage_kb, loop_depth, max_nesting_depth, function_calls, conditionals, complexity_score
- **Classes**: EFFICIENT, MODERATE, NEEDS_OPTIMIZATION
- **Accuracy**: 96-100% (test), 94-96% (5-fold CV)

## Testing

```bash
python test_profiler.py
# Runs 75 tests covering:
# - Python AST analyzer (15 visitor methods)
# - Dataset generation (150 samples)
# - ML pipeline (train/save/load/predict)
# - Recommendations engine
# - Code profiler (all 3 languages)
# - Flask API endpoints
# - 1500 bulk predictions (500 per language)
```
