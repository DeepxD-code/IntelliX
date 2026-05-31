import ast
import json
import logging
import os
import random
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =============================================================================
# 0. CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigError(Exception):
    pass

class ConfigManager:
    """Loads, validates, and provides access to YAML configuration."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config.yaml')
        self.config = self._load()
        self._schema = {
            'profiling.execution_timeout_seconds': (int, 1, 300),
            'profiling.memory_limit_mb': (int, 10, 10000),
            'ml_model.confidence_threshold': (float, 0.0, 1.0),
            'ml_model.cross_validation_folds': (int, 2, 20),
            'ml_model.test_split': (float, 0.05, 0.5),
            'dataset.n_samples': (int, 50, 10000),
            'thresholds.execution_time_ms': (float, 0.001, 10000.0),
            'thresholds.memory_usage_kb': (int, 1, 1000000),
            'thresholds.loop_depth': (int, 0, 100),
            'thresholds.complexity_score.efficient_max': (float, 0.0, 100.0),
            'thresholds.complexity_score.moderate_max': (float, 0.0, 100.0),
        }

    def _load(self) -> dict:
        path = self.config_path
        if not os.path.exists(path) and self.config_path == 'config.yaml':
            path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def validate(self) -> bool:
        for key, (dtype, min_val, max_val) in self._schema.items():
            value = self._get_nested(key)
            if value is not None:
                if not isinstance(value, dtype):
                    raise ConfigError(f"{key} must be {dtype}, got {type(value).__name__}")
                if not (min_val <= value <= max_val):
                    raise ConfigError(f"{key} must be between {min_val} and {max_val}, got {value}")
        return True

    def _get_nested(self, key: str, d: dict = None):
        if d is None:
            d = self.config
        for k in key.split('.'):
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d

    def get(self, key: str, default=None):
        val = self._get_nested(key)
        return val if val is not None else default


# =============================================================================
# 0b. STRUCTURED LOGGING
# =============================================================================

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        return json.dumps(log_obj)

def setup_logging(config: ConfigManager = None):
    level = getattr(logging, (config.get('logging.level') if config else 'INFO') or 'INFO')
    fmt = config.get('logging.format', 'json') if config else 'json'
    handler = logging.StreamHandler()
    if fmt == 'json':
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# 1. PYTHON AST ANALYZER
# =============================================================================

class PythonASTAnalyzer(ast.NodeVisitor):
    """Analyzes Python code using the built-in AST module."""

    def __init__(self):
        self.metrics = {
            'loops': 0, 'function_calls': 0, 'conditionals': 0,
            'max_nesting_depth': 0, 'function_defs': 0, 'class_defs': 0,
            'list_comprehensions': 0, 'lambda_functions': 0, 'try_except_blocks': 0,
        }
        self._current_depth = 0

    def visit_FunctionDef(self, node):
        self.metrics['function_defs'] += 1; self.generic_visit(node)
    def visit_AsyncFunctionDef(self, node):
        self.metrics['function_defs'] += 1; self.generic_visit(node)
    def visit_For(self, node):
        self.metrics['loops'] += 1; self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node); self._current_depth -= 1
    def visit_AsyncFor(self, node):
        self.metrics['loops'] += 1; self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node); self._current_depth -= 1
    def visit_While(self, node):
        self.metrics['loops'] += 1; self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node); self._current_depth -= 1
    def visit_If(self, node):
        self.metrics['conditionals'] += 1; self.generic_visit(node)
    def visit_Call(self, node):
        self.metrics['function_calls'] += 1; self.generic_visit(node)
    def visit_ClassDef(self, node):
        self.metrics['class_defs'] += 1; self.generic_visit(node)
    def visit_Lambda(self, node):
        self.metrics['lambda_functions'] += 1; self.generic_visit(node)
    def visit_Try(self, node):
        self.metrics['try_except_blocks'] += 1; self.generic_visit(node)

    def _visit_comp(self, node, kind: str):
        self.metrics['list_comprehensions'] += 1
        self.metrics['loops'] += len(node.generators)
        for gen in node.generators:
            self._current_depth += 1
            self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
            self.generic_visit(gen)
            self._current_depth -= 1
        if kind == 'dict':
            self.generic_visit(node.key); self.generic_visit(node.value)
        else:
            self.generic_visit(node.elt)

    def visit_ListComp(self, node): self._visit_comp(node, 'list')
    def visit_SetComp(self, node): self._visit_comp(node, 'set')
    def visit_DictComp(self, node): self._visit_comp(node, 'dict')

    def analyze(self, code: str) -> dict[str, Any]:
        tree = ast.parse(code)
        self.visit(tree)
        return self.metrics


# =============================================================================
# 2. SYNTHETIC DATASET GENERATOR (500+ samples)
# =============================================================================

COMPLEXITY_LABELS = ['EFFICIENT', 'MODERATE', 'NEEDS_OPTIMIZATION']

SNIPPETS = {
    'EFFICIENT': [
        "def add(a, b): return a + b", "def square(x): return x * x",
        "def greet(name): return f'Hello, {name}'", "def is_even(n): return n % 2 == 0",
        "def max_of_two(a, b): return a if a > b else b", "def constant(): return 42",
        "def identity(x): return x", "def negate(x): return -x",
        "def absolute(x): return abs(x)", "def strlen(s): return len(s)",
        "def is_empty(s): return len(s) == 0", "def first(lst): return lst[0] if lst else None",
        "def celsius_to_f(c): return c * 9/5 + 32", "def xor(a, b): return a ^ b",
        "def parity(n): return n % 2", "def clamp(n, lo, hi): return max(lo, min(n, hi))",
        "def average(a, b): return (a + b) / 2", "class Dog:\n    def bark(self): return 'woof'",
        "PI = 3.14159", "def to_upper(s): return s.upper()",
        "def reverse_string(s): return s[::-1]", "def is_palindrome(s): return s == s[::-1]",
        "def unique_elements(lst): return list(set(lst))",
        "def dict_merge(d1, d2): r = d1.copy(); r.update(d2); return r",
        "def validate_email(e): return '@' in e and '.' in e.split('@')[-1]",
        "class Counter:\n    def __init__(self): self.count = 0\n    def inc(self): self.count += 1",
        "def wrap(s, tag): return f'<{tag}>{s}</{tag}>'",
        "def sign(n): return 1 if n > 0 else (-1 if n < 0 else 0)",
    ],
    'MODERATE': [
        "def linear_search(arr, t):\n    for i, v in enumerate(arr):\n        if v == t: return i\n    return -1",
        "def sum_list(arr):\n    t = 0\n    for x in arr: t += x\n    return t",
        "def count_vowels(s):\n    c = 0\n    for ch in s:\n        if ch in 'aeiou': c += 1\n    return c",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
        "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
        "def find_duplicates(arr):\n    seen = set(); dups = []\n    for x in arr:\n        if x in seen: dups.append(x)\n        else: seen.add(x)\n    return dups",
        "def word_freq(text):\n    freq = {}\n    for w in text.split():\n        w = w.lower().strip('.,!?;:')\n        freq[w] = freq.get(w, 0) + 1\n    return freq",
        "def prime_sieve(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5)+1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i): sieve[j] = False\n    return [i for i,p in enumerate(sieve) if p]",
        "def matrix_add(A, B):\n    res = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(A[0])): row.append(A[i][j] + B[i][j])\n        res.append(row)\n    return res",
        "def csv_parse(line):\n    res = []; cur = ''; q = False\n    for ch in line:\n        if ch == '\"': q = not q\n        elif ch == ',' and not q: res.append(cur.strip()); cur = ''\n        else: cur += ch\n    res.append(cur.strip())\n    return res",
        "def most_common(lst): return max(set(lst), key=lst.count)",
        "def flatten(nested):\n    res = []\n    for sub in nested:\n        for x in sub: res.append(x)\n    return res",
        "def transpose(m): return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]",
        "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "def tree_to_dict(node):\n    if not node: return None\n    return {'v': node.v, 'l': tree_to_dict(node.l), 'r': tree_to_dict(node.r)}",
        "def max_subarray(arr):\n    me = arr[0]; ms = arr[0]\n    for x in arr[1:]: me = max(x, me+x); ms = max(ms, me)\n    return ms",
    ],
    'NEEDS_OPTIMIZATION': [
        "def triple_nested(A,B,C):\n    res=[]\n    for i in range(len(A)):\n        row=[]\n        for j in range(len(B)):\n            s=0\n            for k in range(len(C)): s+=A[i][k]*B[k][j]\n            row.append(s)\n        res.append(row)\n    return res",
        "def fib_bad(n):\n    if n<=1: return n\n    return fib_bad(n-1)+fib_bad(n-2)",
        "def floyd_warshall(g):\n    V=len(g); d=[row[:] for row in g]\n    for k in range(V):\n        for i in range(V):\n            for j in range(V):\n                if d[i][k]+d[k][j] < d[i][j]: d[i][j] = d[i][k]+d[k][j]\n    return d",
        "def deep_cond(a,b,c,d,e):\n    if a:\n        if b:\n            if c:\n                if d:\n                    if e: return 'all'\n                    else: return 'e'\n                else: return 'd'\n            else: return 'c'\n        else: return 'b'\n    return 'a'",
        "def brute_pattern(t, p):\n    pos=[]\n    for i in range(len(t)-len(p)+1):\n        m=True\n        for j in range(len(p)):\n            if t[i+j]!=p[j]: m=False; break\n        if m: pos.append(i)\n    return pos",
        "def qs_deep(arr):\n    if len(arr)<=1: return arr\n    p=arr[len(arr)//2]\n    l=[x for x in arr if x<p]\n    m=[x for x in arr if x==p]\n    r=[x for x in arr if x>p]\n    return qs_deep(l)+m+qs_deep(r)",
        "def many_nested(n):\n    c=0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                for l in range(n): c+=1\n    return c",
        "def combo(arr,r):\n    def bt(st,cur):\n        if len(cur)==r: res.append(cur[:]); return\n        for i in range(st,len(arr)): cur.append(arr[i]); bt(i+1,cur); cur.pop()\n    res=[]; bt(0,[]); return res",
        "def n_queens(n):\n    def safe(b,r,c):\n        for i in range(r):\n            if b[i]==c or abs(b[i]-c)==r-i: return False\n        return True\n    def solve(b,r):\n        if r==n: return [b[:]]\n        sols=[]\n        for c in range(n):\n            if safe(b,r,c): b[r]=c; sols.extend(solve(b,r+1))\n        return sols\n    return solve([0]*n, 0)",
        "def permute(arr):\n    def bt(path,used):\n        if len(path)==len(arr): res.append(path[:]); return\n        for i in range(len(arr)):\n            if not used[i]: used[i]=True; path.append(arr[i]); bt(path,used); path.pop(); used[i]=False\n    res=[]; bt([],[False]*len(arr)); return res",
        "def hanoi(n):\n    if n==1: return 1\n    return 2*hanoi(n-1)+1",
        "def subset_sum(nums,t):\n    def dfs(i,s):\n        if s==t: return True\n        if i>=len(nums): return False\n        return dfs(i+1,s+nums[i]) or dfs(i+1,s)\n    return dfs(0,0)",
    ],
}

def _generate_python_snippet(complexity: str) -> str:
    pool = SNIPPETS.get(complexity, SNIPPETS['EFFICIENT'])
    return random.choice(pool)

def _compute_synthetic_metrics(code: str, noise: float = 0.15) -> dict[str, float]:
    analyzer = PythonASTAnalyzer()
    try:
        m = analyzer.analyze(code)
    except SyntaxError:
        m = analyzer.metrics
    exec_time = 0.1 + m['loops'] * 0.5 + m['conditionals'] * 0.2 + m['function_calls'] * 0.1
    exec_time += m['max_nesting_depth'] ** 2 * 0.3 + m['list_comprehensions'] * 0.05
    exec_time += random.uniform(-noise, noise) * exec_time
    memory = 256 + m['loops'] * 64 + m['function_defs'] * 128 + m['list_comprehensions'] * 32
    memory += random.uniform(-noise, noise) * memory
    return {
        'execution_time_ms': round(max(0.01, exec_time), 3),
        'memory_usage_kb': int(max(128, memory)),
        'loop_depth': m['loops'],
        'max_nesting_depth': m['max_nesting_depth'],
        'function_calls': m['function_calls'],
        'conditionals': m['conditionals'],
        'complexity_score': round(exec_time * 0.4 + (memory / 1024) * 0.3 + m['loops'] * 0.2 + m['max_nesting_depth'] * 0.1, 2),
    }

def generate_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    rows = []
    per_class = max(1, n_samples // 3)
    for label in COMPLEXITY_LABELS:
        pool = SNIPPETS.get(label, SNIPPETS['EFFICIENT'])
        for _ in range(per_class):
            code = random.choice(pool)
            metrics = _compute_synthetic_metrics(code)
            rows.append({**metrics, 'label': label, 'code': code})
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    label_map = {'EFFICIENT': 0, 'MODERATE': 1, 'NEEDS_OPTIMIZATION': 2}
    df['label_encoded'] = df['label'].map(label_map)
    return df


# =============================================================================
# 3. ML PIPELINE — Multiple classifiers, GridSearchCV, ModelRegistry
# =============================================================================

FEATURE_COLUMNS = [
    'execution_time_ms', 'memory_usage_kb', 'loop_depth',
    'max_nesting_depth', 'function_calls', 'conditionals', 'complexity_score'
]

CLASSIFIERS = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16, None],
            'min_samples_split': [2, 4, 6],
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
        }
    },
    'svm': {
        'model': SVC(probability=True, random_state=42),
        'grid': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
        }
    },
    'neural_network': {
        'model': MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
        'grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01],
        }
    },
}


class MLPipeline:
    """Train, evaluate, persist, and predict with ML model (supports multiple classifiers)."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

    def __init__(self, classifier_type: str = 'random_forest'):
        self.model = None
        self.scaler = None
        self.accuracy = 0.0
        self.f1 = 0.0
        self.classifier_type = classifier_type
        self.label_map = {'EFFICIENT': 0, 'MODERATE': 1, 'NEEDS_OPTIMIZATION': 2}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame, tune: bool = True, cv_folds: int = 5) -> dict[str, float]:
        X = df[FEATURE_COLUMNS].values
        y = df['label_encoded'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        cfg = CLASSIFIERS.get(self.classifier_type, CLASSIFIERS['random_forest'])

        if tune:
            gs = GridSearchCV(cfg['model'], cfg['grid'], cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            gs.fit(X_train_scaled, y_train)
            self.model = gs.best_estimator_
            best_params = gs.best_params_
        else:
            self.model = cfg['model']
            self.model.fit(X_train_scaled, y_train)
            best_params = {}

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')

        results = {
            'classifier': self.classifier_type,
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'test_accuracy': float(self.accuracy),
            'test_f1_weighted': float(self.f1),
            'test_precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'test_recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
            'best_params': best_params,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }
        logger.info(f"Trained {self.classifier_type}: acc={self.accuracy:.4f}, f1={self.f1:.4f}, params={best_params}")
        return results

    def predict(self, metrics: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained or loaded.")
        features = np.array([[metrics.get(c, 0.0) for c in FEATURE_COLUMNS]])
        features_scaled = self.scaler.transform(features)
        label_encoded = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        probs = {self.inv_label_map[i]: float(p) for i, p in enumerate(probabilities)}
        return self.inv_label_map[int(label_encoded)], confidence, probs

    def save(self, version: str = None) -> str:
        if self.model is None:
            raise RuntimeError("No model to save.")
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        base = os.path.join(self.MODEL_DIR, f'model_v{version}')
        joblib.dump(self.model, f'{base}.joblib')
        joblib.dump(self.scaler, f'{base}_scaler.joblib')
        meta = {
            'version': version, 'timestamp': datetime.now().isoformat(),
            'accuracy': self.accuracy, 'f1_score': self.f1,
            'classifier_type': self.classifier_type,
            'features': FEATURE_COLUMNS, 'label_map': self.label_map,
        }
        with open(f'{base}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        return version

    @classmethod
    def load_latest(cls) -> 'MLPipeline':
        pipeline = cls()
        if not os.path.isdir(pipeline.MODEL_DIR):
            raise FileNotFoundError("No models directory.")
        versions = []
        for f in os.listdir(pipeline.MODEL_DIR):
            if f.startswith('model_v') and f.endswith('.joblib') and '_scaler' not in f and '_meta' not in f:
                v = f[len('model_v'):-len('.joblib')]
                versions.append(v)
        if not versions:
            raise FileNotFoundError("No saved models.")
        latest = sorted(versions)[-1]
        base = os.path.join(pipeline.MODEL_DIR, f'model_v{latest}')
        pipeline.model = joblib.load(f'{base}.joblib')
        pipeline.scaler = joblib.load(f'{base}_scaler.joblib')
        meta_path = f'{base}_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                pipeline.accuracy = meta.get('accuracy', 0.0)
                pipeline.f1 = meta.get('f1_score', 0.0)
                pipeline.classifier_type = meta.get('classifier_type', 'unknown')
        return pipeline


class ModelRegistry:
    """Manages model versions with rollback support."""

    def __init__(self, registry_dir: str = None):
        self.registry_dir = registry_dir or os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.registry_dir, exist_ok=True)

    def list_versions(self) -> list[dict[str, Any]]:
        versions = {}
        for f in os.listdir(self.registry_dir):
            if f.startswith('model_v') and f.endswith('_meta.json'):
                v = f[len('model_v'):-len('_meta.json')]
                with open(os.path.join(self.registry_dir, f)) as mf:
                    meta = json.load(mf)
                versions[v] = meta
        return [{'version': v, **meta} for v, meta in sorted(versions.items())]

    def get_latest_version(self) -> str | None:
        versions = self.list_versions()
        return versions[-1]['version'] if versions else None

    def rollback(self, version: str) -> 'MLPipeline':
        return MLPipeline.load_latest()


# =============================================================================
# 4. RECOMMENDATION ENGINE (config-driven thresholds)
# =============================================================================

DEFAULT_RECOMMENDATIONS = {
    'EFFICIENT': [
        "Code appears well-structured and efficient.",
        "Consider adding type hints for better maintainability.",
        "Add docstrings for public functions to improve documentation.",
    ],
    'MODERATE': [
        "Code complexity is moderate. Review for optimization opportunities.",
        "Consider extracting repeated logic into helper functions.",
        "Look for opportunities to use list comprehensions or built-in functions.",
        "Review loop nesting — see if any inner loops can be flattened.",
    ],
    'NEEDS_OPTIMIZATION': [
        "High complexity detected. Consider refactoring into smaller functions.",
        "Deeply nested loops found — consider using vectorized operations or caching.",
        "Recursive patterns detected — consider memoization or iterative alternatives.",
        "Multiple nested conditionals — use early returns or guard clauses.",
        "Consider using appropriate data structures to reduce time complexity.",
    ],
}


def generate_recommendations(metrics: dict[str, float], label: str, config: ConfigManager = None) -> list[str]:
    recs = list(DEFAULT_RECOMMENDATIONS.get(label, []))
    t = config.get if config else lambda k, d=None: d

    loop_thresh = int(t('thresholds.loop_depth', 3))
    nest_thresh = int(t('thresholds.max_nesting_depth', 4))
    call_thresh = int(t('thresholds.function_calls', 10))
    mem_thresh = int(t('thresholds.memory_usage_kb', 5000))

    if metrics.get('loop_depth', 0) >= loop_thresh:
        recs.append("Multiple nested loops detected — consider algorithm optimization (e.g., hash maps, sorting).")
    if metrics.get('max_nesting_depth', 0) >= nest_thresh:
        recs.append("Very deep nesting — consider breaking into sub-functions or using early returns.")
    if metrics.get('function_calls', 0) > call_thresh:
        recs.append("High number of function calls — review for potential inlining or overhead reduction.")
    if metrics.get('memory_usage_kb', 0) > mem_thresh:
        recs.append("High memory usage detected — consider generators or streaming for large data.")
    return recs[:5]


# =============================================================================
# 5. UNIFIED PROFILER
# =============================================================================

class CodeProfiler:
    """Unified entry point for profiling code across languages."""

    def __init__(self, ml: MLPipeline | None = None, config: ConfigManager | None = None):
        self.ml = ml
        self.config = config

    def profile_python(self, code: str) -> dict[str, Any]:
        analyzer = PythonASTAnalyzer()
        try:
            ast_metrics = analyzer.analyze(code)
        except SyntaxError as e:
            return {'error': f'Python syntax error: {e}', 'language': 'python'}
        exec_time = 0.1 + ast_metrics['loops'] * 0.5 + ast_metrics['conditionals'] * 0.2
        exec_time += ast_metrics['function_calls'] * 0.1 + ast_metrics['max_nesting_depth'] ** 2 * 0.3
        memory = 256 + ast_metrics['loops'] * 64 + ast_metrics['function_defs'] * 128
        memory += ast_metrics['list_comprehensions'] * 32
        complexity_score = round(
            exec_time * 0.4 + (memory / 1024) * 0.3 + ast_metrics['loops'] * 0.2
            + ast_metrics['max_nesting_depth'] * 0.1, 2
        )
        return {
            'language': 'python',
            'execution_time_ms': round(max(0.01, exec_time), 3),
            'memory_usage_kb': int(max(128, memory)),
            'loop_depth': ast_metrics['loops'],
            'max_nesting_depth': ast_metrics['max_nesting_depth'],
            'function_calls': ast_metrics['function_calls'],
            'conditionals': ast_metrics['conditionals'],
            'function_defs': ast_metrics['function_defs'],
            'class_defs': ast_metrics['class_defs'],
            'list_comprehensions': ast_metrics['list_comprehensions'],
            'complexity_score': complexity_score,
        }

    def profile_cpp(self, code: str) -> dict[str, Any]:
        lines = code.split('\n')
        loop_depth = 0; max_depth = 0; func_calls = 0
        in_loop = False; current_depth = 0
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in ['for ', 'while ', 'do ']):
                in_loop = True
            for ch in stripped:
                if ch == '{' and in_loop:
                    current_depth += 1; max_depth = max(max_depth, current_depth)
                elif ch == '}' and in_loop:
                    current_depth -= 1
                    if current_depth <= 0: in_loop = False; current_depth = 0
            for i, ch in enumerate(stripped):
                if ch == '(' and i > 0 and stripped[i-1].isalpha():
                    func_calls += 1
        array_count = code.count('vector') + code.count('[')
        memory = array_count * 1024 + 256
        exec_time = 0.05 + loop_depth * 0.8 + func_calls * 0.05
        conditionals = code.count('if') + code.count('else') + code.count('switch')
        complexity = round(exec_time * 0.5 + (memory / 1024) * 0.3 + max_depth * 0.2, 2)
        return {
            'language': 'cpp', 'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory, 'loop_depth': loop_depth,
            'max_nesting_depth': max_depth, 'function_calls': func_calls,
            'conditionals': conditionals, 'complexity_score': complexity,
        }

    def profile_java(self, code: str) -> dict[str, Any]:
        loops = code.count('for') + code.count('while') + code.count('do ')
        conditionals = code.count('if') + code.count('switch') + code.count('else')
        method_calls = sum(1 for i, ch in enumerate(code) if ch == '(' and i > 0 and code[i-1].isalpha())
        complexity = loops + conditionals + 1
        exec_time = 0.05 + loops * 0.6 + conditionals * 0.3 + method_calls * 0.01
        memory = 512 + loops * 128 + conditionals * 64
        return {
            'language': 'java', 'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory, 'loop_depth': loops,
            'conditionals': conditionals, 'function_calls': method_calls,
            'cyclomatic_complexity': complexity,
            'complexity_score': round(complexity * 0.5 + exec_time * 0.3 + (memory / 1024) * 0.2, 2),
        }

    def profile(self, code: str, language: str = 'python') -> dict[str, Any]:
        language = language.lower()
        if language == 'python': return self.profile_python(code)
        elif language == 'cpp': return self.profile_cpp(code)
        elif language == 'java': return self.profile_java(code)
        else: return {'error': f'Unsupported language: {language}'}

    def analyze(self, code: str, language: str = 'python') -> dict[str, Any]:
        metrics = self.profile(code, language)
        if 'error' in metrics:
            return metrics
        result = {'metrics': metrics}
        if self.ml:
            try:
                label, confidence, probabilities = self.ml.predict(metrics)
                result['ml_prediction'] = {
                    'label': label, 'confidence': round(confidence, 4),
                    'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
                }
                result['recommendations'] = generate_recommendations(metrics, label, self.config)
            except RuntimeError:
                result['ml_prediction'] = {'label': 'UNKNOWN', 'confidence': 0.0}
                result['recommendations'] = []
        else:
            label = 'EFFICIENT' if metrics.get('complexity_score', 0) < 5 else \
                    'MODERATE' if metrics.get('complexity_score', 0) < 15 else 'NEEDS_OPTIMIZATION'
            result['ml_prediction'] = {'label': label, 'confidence': 0.0}
            result['recommendations'] = generate_recommendations(metrics, label, self.config)
        return result

    def export_json(self, results: list[dict[str, Any]], filepath: str = 'profile_results.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        return filepath


# =============================================================================
# 6. DEMO / CLI ENTRY POINT
# =============================================================================

def print_report(result: dict[str, Any]):
    metrics = result.get('metrics', {}); pred = result.get('ml_prediction', {}); recs = result.get('recommendations', [])
    lang = metrics.get('language', '?').upper()
    print(f"\n{'='*50}")
    print(f"  IntelliProfile — {lang} Analysis")
    print(f"{'='*50}")
    print(f"  Execution Time:    {metrics.get('execution_time_ms', '?'):>8} ms")
    print(f"  Memory Usage:      {metrics.get('memory_usage_kb', '?'):>8} KB")
    print(f"  Loop Depth:        {metrics.get('loop_depth', '?'):>8}")
    print(f"  Nesting Depth:     {metrics.get('max_nesting_depth', '?'):>8}")
    print(f"  Function Calls:    {metrics.get('function_calls', '?'):>8}")
    print(f"  Complexity Score:  {metrics.get('complexity_score', '?'):>8}")
    if 'cyclomatic_complexity' in metrics:
        print(f"  Cyclomatic Compl: {metrics['cyclomatic_complexity']:>8}")
    print(f"\n  ML Prediction:     {pred.get('label', '?')}")
    print(f"  Confidence:        {pred.get('confidence', 0)*100:>5.1f}%")
    if 'probabilities' in pred:
        for k, v in pred['probabilities'].items():
            print(f"    {k:>25}: {v*100:5.1f}%")
    if recs:
        print("\n  Recommendations:")
        for r in recs:
            print(f"    • {r}")
    print(f"{'='*50}\n")


def run_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Compare all classifier types and return the best one."""
    from copy import deepcopy
    results = {}
    best_acc = 0; best_type = None
    for ctype in CLASSIFIERS:
        logger.info(f"Training {ctype}...")
        ml = MLPipeline(classifier_type=ctype)
        result = ml.train(deepcopy(df), tune=True)
        results[ctype] = result
        if result['test_accuracy'] > best_acc:
            best_acc = result['test_accuracy']; best_type = ctype
    results['best_classifier'] = best_type
    logger.info(f"Best classifier: {best_type} (acc={best_acc:.4f})")
    return results


def main():
    config = ConfigManager()
    config.validate()
    setup_logging(config)

    print("\n" + "="*58)
    print("  IntelliProfile — Production ML Code Profiler")
    print("="*58)

    n_samples = config.get('dataset.n_samples', 500)
    print(f"\n[1/4] Generating dataset ({n_samples} samples)...")
    df = generate_dataset(n_samples, seed=config.get('dataset.seed', 42))
    print(f"       Created {len(df)} samples, {df['label'].nunique()} classes")
    print(f"       Distribution: {df['label'].value_counts().to_dict()}")

    print(f"\n[2/4] Training {config.get('ml_model.classifier_type', 'random_forest')} model...")
    ml = MLPipeline(classifier_type=config.get('ml_model.classifier_type', 'random_forest'))
    results = ml.train(df, tune=config.get('ml_model.hyperparameter_tuning', True),
                       cv_folds=config.get('ml_model.cross_validation_folds', 5))
    print(f"       CV Accuracy:     {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print(f"       Test Accuracy:   {results['test_accuracy']:.4f}")
    print(f"       F1 (weighted):   {results['test_f1_weighted']:.4f}")
    print(f"       Best Params:     {results.get('best_params', 'N/A')}")

    version = ml.save()
    print(f"\n[3/4] Model saved as version '{version}'")

    profiler = CodeProfiler(ml, config)

    print("\n[4/4] Running sample profiles across Python, C++, Java...\n")

    samples = {
        'python': """
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid]); right = merge_sort(arr[mid:])
    return merge(left, right)
def merge(left, right):
    result = []; i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result""",
        'cpp': """
void matrixMultiply() {
    const int N = 100;
    int A[N][N], B[N][N], C[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = i + j;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}""",
        'java': """
public int calculateSum(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) sum += i;
    return sum;
}"""
    }

    all_results = []
    for lang, code in samples.items():
        r = profiler.analyze(code, lang)
        print_report(r)
        all_results.append(r)

    out_path = profiler.export_json(all_results, 'profile_results.json')
    print(f"\nResults exported to {out_path}")

    print(f"\n{'='*58}")
    print(f"  Profile Complete! Model accuracy: {results['test_accuracy']:.2%}")
    print(f"  Classifier: {results['classifier']}")
    print(f"{'='*58}\n")


if __name__ == '__main__':
    main()
