# IntelliProfile — Setup Guide

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/DeepxD-code/IntelliX.git
cd IntelliX

# 2. Install dependencies
pip install scikit-learn joblib pandas numpy flask
```

## Usage

### CLI Mode (Train + Demo)

```bash
python profiler_main.py
```

This will:
1. Generate a synthetic dataset (150 samples)
2. Train a Random Forest classifier
3. Save the model to `models/`
4. Profile sample Python, C++, and Java code
5. Export results to `profile_results.json`

### Web UI Mode

```bash
python app.py
```

Open http://localhost:5000 in your browser.

- Select a language (Python/C++/Java)
- Paste code or load a sample
- Click "Profile Code" to analyze
- View metrics, ML prediction, probabilities, and recommendations

### API Mode

The Flask app exposes a REST API at http://localhost:5000.

```bash
# Health check
curl http://localhost:5000/api/health

# Profile code
curl -X POST http://localhost:5000/api/profile \
  -H "Content-Type: application/json" \
  -d '{"code": "def f(): return 1", "language": "python"}'

# Batch profile
curl -X POST http://localhost:5000/api/batch-profile \
  -H "Content-Type: application/json" \
  -d '{"snippets": [{"code": "x=1", "language": "python"}, {"code": "int x;", "language": "cpp"}]}'
```

### Testing

```bash
python test_profiler.py
```

Runs 75 unit tests + 1500 bulk predictions across all 3 languages.

## Project Map

```
IntelliX/
├── app.py                 # Flask server (routes, API, frontend)
├── profiler_main.py       # Core engine (AST, ML, profiler)
├── test_profiler.py       # Test suite
├── cpp_profiler.cpp.txt   # C++ module (legacy reference)
├── java_profiler.java     # Java module (legacy reference)
├── templates/index.html   # Web UI
├── static/style.css       # Web UI styles
├── models/                # Trained models (auto-generated)
├── profile_results.json   # Last run results (auto-generated)
├── .gitignore
├── README.md
└── SETUP.md
```

## Troubleshooting

**"No saved models found"**: Run `python profiler_main.py` first to train and save a model, or the web app will auto-train on first startup.

**Port 5000 in use**: Set a custom port: `$env:PORT=5001; python app.py`

**Unicode errors on Windows**: Run with UTF-8 mode: `python -X utf8 app.py`
