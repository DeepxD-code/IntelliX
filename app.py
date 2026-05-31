"""IntelliProfile — Flask Web Frontend + REST API"""

import hashlib
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

from flasgger import Swagger, swag_from
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiler_main import (
    COMPLEXITY_LABELS,
    FEATURE_COLUMNS,
    CodeProfiler,
    ConfigManager,
    MLPipeline,
    generate_dataset,
    setup_logging,
)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024
app.config['SWAGGER'] = {
    'title': 'IntelliProfile API',
    'description': 'ML-powered code profiler for Python, C++, and Java',
    'version': '1.0.0',
    'uiversion': 3,
}
Swagger(app)

app.start_time = time.time()

config = ConfigManager()
setup_logging(config)
logger = logging.getLogger(__name__)

profiler = None
ml_pipeline = None
model_info = {}

DB_PATH = os.path.join(os.path.dirname(__file__), 'profile_history.db')


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                language TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                complexity_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                execution_time_ms REAL,
                memory_usage_kb INTEGER,
                complexity_score REAL,
                analysis_time_ms REAL
            )
        """)
        g.db.commit()
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_backend():
    global profiler, ml_pipeline, model_info
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        ml_pipeline = MLPipeline.load_latest()
        logger.info("Loaded existing model")
        model_info = {
            'accuracy': ml_pipeline.accuracy,
            'f1_score': ml_pipeline.f1,
            'classifier': ml_pipeline.classifier_type,
        }
    except (FileNotFoundError, Exception) as e:
        logger.info(f"No saved model found ({e}). Training new model...")
        n = config.get('dataset.n_samples', 500)
        ctype = config.get('ml_model.classifier_type', 'random_forest')
        df = generate_dataset(n, seed=config.get('dataset.seed', 42))
        ml_pipeline = MLPipeline(classifier_type=ctype)
        results = ml_pipeline.train(df, tune=config.get('ml_model.hyperparameter_tuning', True))
        version = ml_pipeline.save()
        model_info = {
            'accuracy': results['test_accuracy'],
            'f1_score': results['test_f1_weighted'],
            'cv_accuracy': results['cv_accuracy_mean'],
            'classifier': ctype,
            'version': version,
        }
        logger.info(f"Trained {ctype}, acc={results['test_accuracy']:.4f}")

    profiler = CodeProfiler(ml_pipeline, config)


init_backend()


@app.route('/')
def index():
    return render_template('portfolio.html', model_info={
        'accuracy': model_info.get('accuracy', 0),
        'f1_score': model_info.get('f1_score', 0),
        'classifier': model_info.get('classifier', 'random_forest'),
        'features': FEATURE_COLUMNS,
        'classes': COMPLEXITY_LABELS,
    })


@app.route('/api/health', methods=['GET'])
@swag_from({
    'tags': ['System'],
    'responses': {
        200: {
            'description': 'Health check',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'model_accuracy': {'type': 'number'},
                    'uptime_seconds': {'type': 'number'},
                    'languages': {'type': 'array', 'items': {'type': 'string'}},
                }
            }
        }
    }
})
def api_health():
    return jsonify({
        'status': 'healthy',
        'model_accuracy': model_info.get('accuracy', 0),
        'model_f1': model_info.get('f1_score', 0),
        'uptime_seconds': round(time.time() - app.start_time, 2),
        'languages': ['python', 'cpp', 'java'],
    })


@app.route('/api/profile', methods=['POST'])
@swag_from({
    'tags': ['Profiling'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'required': ['code'],
                'properties': {
                    'code': {'type': 'string', 'description': 'Source code to profile'},
                    'language': {'type': 'string', 'enum': ['python', 'cpp', 'java'], 'default': 'python'},
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Profile result with metrics, ML prediction, and recommendations'},
        400: {'description': 'Invalid input'},
    }
})
def api_profile():
    data = request.get_json(force=True)
    code = data.get('code', '').strip()
    language = data.get('language', 'python').strip().lower()

    if not code:
        return jsonify({'error': 'Code cannot be empty'}), 400
    if language not in ['python', 'cpp', 'java']:
        return jsonify({'error': f'Unsupported language: {language}'}), 400
    if len(code) > 100000:
        return jsonify({'error': 'Code exceeds 100KB limit'}), 400

    try:
        start_t = time.time()
        result = profiler.analyze(code, language)
        elapsed = round((time.time() - start_t) * 1000, 2)

        if 'error' in result:
            return jsonify({'error': result['error']}), 400

        result['analysis_time_ms'] = elapsed
        result['timestamp'] = datetime.now().isoformat()

        pred = result.get('ml_prediction', {})
        label = pred.get('label', '')
        confidence = pred.get('confidence', 0)
        metrics = result.get('metrics', {})
        code_hash = hashlib.md5(code.encode()).hexdigest()[:12]

        db = get_db()
        db.execute(
            "INSERT INTO profiles (timestamp, language, code_hash, complexity_label, confidence, execution_time_ms, memory_usage_kb, complexity_score, analysis_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (result['timestamp'], language, code_hash, label, confidence,
             metrics.get('execution_time_ms'), metrics.get('memory_usage_kb'),
             metrics.get('complexity_score'), elapsed))
        db.commit()

        return jsonify(result)

    except SyntaxError as e:
        return jsonify({'error': f'Syntax error: {e}'}), 400
    except Exception as e:
        logger.error(f"Profile error: {e}", exc_info=True)
        return jsonify({'error': f'Internal error: {e}'}), 500


@app.route('/api/batch-profile', methods=['POST'])
@swag_from({
    'tags': ['Profiling'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'required': ['snippets'],
                'properties': {
                    'snippets': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'code': {'type': 'string'},
                                'language': {'type': 'string', 'enum': ['python', 'cpp', 'java']},
                            }
                        }
                    }
                }
            }
        }
    ],
    'responses': {
        200: {'description': 'Batch profile results'},
    }
})
def api_batch_profile():
    data = request.get_json(force=True)
    snippets = data.get('snippets', [])

    if not snippets:
        return jsonify({'error': 'No snippets provided'}), 400
    if len(snippets) > 50:
        return jsonify({'error': 'Maximum 50 snippets per batch'}), 400

    results = []
    errors = []
    for i, snippet in enumerate(snippets):
        code = snippet.get('code', '').strip()
        language = snippet.get('language', 'python').strip().lower()
        if not code:
            errors.append({'index': i, 'error': 'Empty code'})
            continue
        if language not in ['python', 'cpp', 'java']:
            errors.append({'index': i, 'error': f'Unsupported language: {language}'})
            continue
        try:
            result = profiler.analyze(code, language)
            result['index'] = i
            if 'error' in result:
                errors.append({'index': i, 'error': result['error']})
            else:
                results.append(result)
        except Exception as e:
            errors.append({'index': i, 'error': str(e)})

    return jsonify({
        'total': len(snippets),
        'successful': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors,
    })


@app.route('/api/model-info', methods=['GET'])
@swag_from({
    'tags': ['System'],
    'responses': {
        200: {
            'description': 'Model metadata',
            'schema': {
                'type': 'object',
                'properties': {
                    'model_type': {'type': 'string'},
                    'features': {'type': 'array', 'items': {'type': 'string'}},
                    'classes': {'type': 'array', 'items': {'type': 'string'}},
                    'accuracy': {'type': 'number'},
                }
            }
        }
    }
})
def api_model_info():
    return jsonify({
        'model_type': model_info.get('classifier', 'random_forest'),
        'features': FEATURE_COLUMNS,
        'classes': COMPLEXITY_LABELS,
        'config_classifier': config.get('ml_model.classifier_type', 'random_forest'),
        'config_tuning': config.get('ml_model.hyperparameter_tuning', True),
        'config_cv_folds': config.get('ml_model.cross_validation_folds', 5),
        **model_info,
    })


@app.route('/api/history', methods=['GET'])
@swag_from({
    'tags': ['History'],
    'parameters': [
        {
            'name': 'limit',
            'in': 'query',
            'type': 'integer',
            'default': 50,
            'description': 'Number of recent profiles to return',
        }
    ],
    'responses': {
        200: {
            'description': 'Profile history from SQLite',
            'schema': {
                'type': 'object',
                'properties': {
                    'total': {'type': 'integer'},
                    'results': {'type': 'array'},
                }
            }
        }
    }
})
def api_history():
    limit = request.args.get('limit', 50, type=int)
    limit = min(max(limit, 1), 500)
    db = get_db()
    cursor = db.execute(
        "SELECT timestamp, language, complexity_label, confidence, execution_time_ms, memory_usage_kb, complexity_score, analysis_time_ms FROM profiles ORDER BY id DESC LIMIT ?",
        (limit,))
    rows = cursor.fetchall()
    columns = ['timestamp', 'language', 'complexity_label', 'confidence',
               'execution_time_ms', 'memory_usage_kb', 'complexity_score', 'analysis_time_ms']
    return jsonify({
        'total': len(rows),
        'results': [dict(zip(columns, row)) for row in rows],
    })


@app.route('/api/docs')
def api_docs():
    return render_template('swagger.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    logger.info(f"Starting IntelliProfile on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
