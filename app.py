"""IntelliProfile — Flask Web Frontend + REST API"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiler_main import (
    generate_dataset, MLPipeline, CodeProfiler,
    COMPLEXITY_LABELS, FEATURE_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024
app.start_time = time.time()

profiler = None
ml_pipeline = None
model_info = {}


def init_backend():
    global profiler, ml_pipeline, model_info

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        ml_pipeline = MLPipeline.load_latest()
        logger.info(f"Loaded existing model v{ml_pipeline.save()}")
        model_info = {
            'accuracy': ml_pipeline.accuracy,
            'f1_score': ml_pipeline.f1,
        }
    except (FileNotFoundError, Exception) as e:
        logger.info(f"No saved model found ({e}). Training new model...")
        df = generate_dataset(200)
        ml_pipeline = MLPipeline()
        results = ml_pipeline.train(df)
        version = ml_pipeline.save()
        model_info = {
            'accuracy': results['test_accuracy'],
            'f1_score': results['test_f1_weighted'],
            'cv_accuracy': results['cv_accuracy_mean'],
            'version': version,
        }
        logger.info(f"Trained new model v{version}, accuracy={results['test_accuracy']:.4f}")

    profiler = CodeProfiler(ml_pipeline)


init_backend()


@app.route('/')
def index():
    return render_template('index.html',
                           languages=['python', 'cpp', 'java'],
                           model_info=model_info)


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'model_accuracy': model_info.get('accuracy', 0),
        'model_f1': model_info.get('f1_score', 0),
        'uptime_seconds': round(time.time() - app.start_time, 2),
        'languages': ['python', 'cpp', 'java'],
    })


@app.route('/api/profile', methods=['POST'])
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

        return jsonify(result)

    except SyntaxError as e:
        return jsonify({'error': f'Syntax error: {e}'}), 400
    except Exception as e:
        logger.error(f"Profile error: {e}", exc_info=True)
        return jsonify({'error': f'Internal error: {e}'}), 500


@app.route('/api/batch-profile', methods=['POST'])
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
def api_model_info():
    return jsonify({
        'model_type': 'RandomForest',
        'n_estimators': 200,
        'max_depth': 12,
        'features': FEATURE_COLUMNS,
        'classes': COMPLEXITY_LABELS,
        **model_info,
    })


@app.route('/api/history', methods=['GET'])
def api_history():
    results_file = os.path.join(os.path.dirname(__file__), 'profile_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        return jsonify({'total': len(data), 'results': data})
    return jsonify({'total': 0, 'results': []})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    logger.info(f"Starting IntelliProfile on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
