import ast
import os
import sys
import json
import random
import math
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# =============================================================================
# 1. PYTHON AST ANALYZER
# =============================================================================

class PythonASTAnalyzer(ast.NodeVisitor):
    """Analyzes Python code using the built-in AST module."""

    def __init__(self):
        self.metrics = {
            'loops': 0,
            'function_calls': 0,
            'conditionals': 0,
            'max_nesting_depth': 0,
            'function_defs': 0,
            'class_defs': 0,
            'list_comprehensions': 0,
            'lambda_functions': 0,
            'try_except_blocks': 0,
        }
        self._current_depth = 0

    def visit_FunctionDef(self, node):
        self.metrics['function_defs'] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.metrics['function_defs'] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.metrics['loops'] += 1
        self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_AsyncFor(self, node):
        self.metrics['loops'] += 1
        self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_While(self, node):
        self.metrics['loops'] += 1
        self._current_depth += 1
        self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_If(self, node):
        self.metrics['conditionals'] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.metrics['function_calls'] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.metrics['class_defs'] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.metrics['list_comprehensions'] += 1
        self.metrics['loops'] += len(node.generators)
        for gen in node.generators:
            self._current_depth += 1
            self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
            self.generic_visit(gen)
            self._current_depth -= 1
        self.generic_visit(node.elt)

    def visit_SetComp(self, node):
        self.metrics['list_comprehensions'] += 1
        self.metrics['loops'] += len(node.generators)
        for gen in node.generators:
            self._current_depth += 1
            self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
            self.generic_visit(gen)
            self._current_depth -= 1
        self.generic_visit(node.elt)

    def visit_DictComp(self, node):
        self.metrics['list_comprehensions'] += 1
        self.metrics['loops'] += len(node.generators)
        for gen in node.generators:
            self._current_depth += 1
            self.metrics['max_nesting_depth'] = max(self.metrics['max_nesting_depth'], self._current_depth)
            self.generic_visit(gen)
            self._current_depth -= 1
        self.generic_visit(node.key)
        self.generic_visit(node.value)

    def visit_Lambda(self, node):
        self.metrics['lambda_functions'] += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.metrics['try_except_blocks'] += 1
        self.generic_visit(node)

    def analyze(self, code: str) -> Dict[str, Any]:
        tree = ast.parse(code)
        self.visit(tree)
        return self.metrics


# =============================================================================
# 2. SYNTHETIC DATASET GENERATOR
# =============================================================================

COMPLEXITY_LABELS = ['EFFICIENT', 'MODERATE', 'NEEDS_OPTIMIZATION']


def _generate_python_snippet(complexity: str) -> str:
    """Generate a synthetic Python code snippet at the given complexity level."""
    if complexity == 'EFFICIENT':
        snippets = [
            "def add(a, b):\n    return a + b",
            "def square(x):\n    return x * x",
            "def greet(name):\n    return f'Hello, {name}'",
            "def is_even(n):\n    return n % 2 == 0",
            "def max_of_two(a, b):\n    return a if a > b else b",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def linear_search(arr, target):\n    for i, val in enumerate(arr):\n        if val == target:\n            return i\n    return -1",
            "def sum_list(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total",
            "def count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for ch in s:\n        if ch in vowels:\n            count += 1\n    return count",
            "def reverse_string(s):\n    return s[::-1]",
            "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def unique_elements(lst):\n    return list(set(lst))",
            "def dict_merge(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result",
            "def validate_email(email):\n    return '@' in email and '.' in email.split('@')[-1]",
            "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n    def get_count(self):\n        return self.count",
        ]
    elif complexity == 'MODERATE':
        snippets = [
            "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
            "def matrix_add(A, B):\n    result = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(A[0])):\n            row.append(A[i][j] + B[i][j])\n        result.append(row)\n    return result",
            "def find_duplicates(arr):\n    seen = set()\n    duplicates = []\n    for x in arr:\n        if x in seen:\n            duplicates.append(x)\n        else:\n            seen.add(x)\n    return duplicates",
            "def word_frequency(text):\n    freq = {}\n    words = text.split()\n    for w in words:\n        w = w.lower().strip('.,!?;:')\n        if w in freq:\n            freq[w] += 1\n        else:\n            freq[w] = 1\n    return freq",
            "def prime_sieve(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if sieve[i]:\n            for j in range(i * i, n + 1, i):\n                sieve[j] = False\n    return [i for i, is_prime in enumerate(sieve) if is_prime]",
            "def csv_parse_line(line):\n    result = []\n    current = ''\n    in_quotes = False\n    for ch in line:\n        if ch == '\"':\n            in_quotes = not in_quotes\n        elif ch == ',' and not in_quotes:\n            result.append(current.strip())\n            current = ''\n        else:\n            current += ch\n    result.append(current.strip())\n    return result",
            "def tree_to_dict(node):\n    if not node:\n        return None\n    return {'value': node.value, 'left': tree_to_dict(node.left), 'right': tree_to_dict(node.right)}",
            "def matrix_transpose(matrix):\n    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]",
            "def n_queens(n):\n    def is_safe(board, row, col):\n        for i in range(row):\n            if board[i] == col or abs(board[i] - col) == row - i:\n                return False\n        return True\n    def solve(board, row):\n        if row == n:\n            return [board[:]]\n        solutions = []\n        for col in range(n):\n            if is_safe(board, row, col):\n                board[row] = col\n                solutions.extend(solve(board, row + 1))\n        return solutions\n    return solve([0] * n, 0)",
        ]
    else:
        snippets = [
            "def triple_nested(A, B, C):\n    result = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(B)):\n            total = 0\n            for k in range(len(C)):\n                total += A[i][k] * B[k][j] + C[i][j][k]\n            row.append(total)\n        result.append(row)\n    return result",
            "def fibonacci_recursive_bad(n):\n    if n <= 1:\n        return n\n    return fibonacci_recursive_bad(n - 1) + fibonacci_recursive_bad(n - 2)",
            "def deeply_nested_conditionals(a, b, c, d, e):\n    if a:\n        if b:\n            if c:\n                if d:\n                    if e:\n                        return 'all true'\n                    else:\n                        return 'e false'\n                else:\n                    return 'd false'\n            else:\n                return 'c false'\n        else:\n            return 'b false'\n    return 'a false'",
            "def brute_force_pattern(text, pattern):\n    positions = []\n    for i in range(len(text) - len(pattern) + 1):\n        match = True\n        for j in range(len(pattern)):\n            if text[i + j] != pattern[j]:\n                match = False\n                break\n        if match:\n            positions.append(i)\n    return positions",
            "def floyd_warshall(graph):\n    V = len(graph)\n    dist = [row[:] for row in graph]\n    for k in range(V):\n        for i in range(V):\n            for j in range(V):\n                if dist[i][k] + dist[k][j] < dist[i][j]:\n                    dist[i][j] = dist[i][k] + dist[k][j]\n    return dist",
            "def generate_combinations(arr, r):\n    def backtrack(start, current):\n        if len(current) == r:\n            result.append(current[:])\n            return\n        for i in range(start, len(arr)):\n            current.append(arr[i])\n            backtrack(i + 1, current)\n            current.pop()\n    result = []\n    backtrack(0, [])\n    return result",
            "def quicksort_deep(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort_deep(left) + middle + quicksort_deep(right)",
            "def many_nested_loops(n):\n    count = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                for l in range(n):\n                    count += 1\n    return count",
            "def recursive_backtracking(n):\n    def bt(choices, path, remaining):\n        if remaining == 0:\n            results.append(path[:])\n            return\n        for c in choices:\n            if c <= remaining:\n                path.append(c)\n                bt(choices, path, remaining - c)\n                path.pop()\n    results = []\n    bt([1, 2, 3], [], n)\n    return results",
        ]
    return random.choice(snippets)


def _compute_synthetic_metrics(code: str) -> Dict[str, float]:
    """Compute metrics for a Python snippet using AST analysis."""
    analyzer = PythonASTAnalyzer()
    try:
        m = analyzer.analyze(code)
    except SyntaxError:
        m = analyzer.metrics

    exec_time = 0.1 + m['loops'] * 0.5 + m['conditionals'] * 0.2 + m['function_calls'] * 0.1
    exec_time += m['max_nesting_depth'] ** 2 * 0.3 + m['list_comprehensions'] * 0.05
    exec_time += random.uniform(-0.2, 0.2) * exec_time

    memory = 256 + m['loops'] * 64 + m['function_defs'] * 128 + m['list_comprehensions'] * 32
    memory += random.uniform(-0.1, 0.1) * memory

    return {
        'execution_time_ms': round(max(0.01, exec_time), 3),
        'memory_usage_kb': int(max(128, memory)),
        'loop_depth': m['loops'],
        'max_nesting_depth': m['max_nesting_depth'],
        'function_calls': m['function_calls'],
        'conditionals': m['conditionals'],
        'complexity_score': round(exec_time * 0.4 + (memory / 1024) * 0.3 + m['loops'] * 0.2 + m['max_nesting_depth'] * 0.1, 2),
    }


def generate_dataset(n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate a labeled dataset of synthetic code snippets."""
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    per_class = n_samples // 3

    for label in COMPLEXITY_LABELS:
        for _ in range(per_class):
            code = _generate_python_snippet(label)
            metrics = _compute_synthetic_metrics(code)
            rows.append({**metrics, 'label': label, 'code': code})

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    label_map = {'EFFICIENT': 0, 'MODERATE': 1, 'NEEDS_OPTIMIZATION': 2}
    df['label_encoded'] = df['label'].map(label_map)
    return df


# =============================================================================
# 3. ML PIPELINE
# =============================================================================

FEATURE_COLUMNS = [
    'execution_time_ms', 'memory_usage_kb', 'loop_depth',
    'max_nesting_depth', 'function_calls', 'conditionals', 'complexity_score'
]


class MLPipeline:
    """Train, evaluate, persist, and predict with a Random Forest classifier."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.accuracy: float = 0.0
        self.f1: float = 0.0
        self.label_map = {'EFFICIENT': 0, 'MODERATE': 1, 'NEEDS_OPTIMIZATION': 2}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        X = df[FEATURE_COLUMNS].values
        y = df['label_encoded'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=4,
            random_state=42, class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')

        results = {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'test_accuracy': float(self.accuracy),
            'test_f1_weighted': float(self.f1),
            'test_precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
            'test_recall_weighted': float(recall_score(y_test, y_pred, average='weighted')),
        }
        return results

    def predict(self, metrics: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

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
        path = os.path.join(self.MODEL_DIR, f'model_v{version}.joblib')
        scaler_path = os.path.join(self.MODEL_DIR, f'scaler_v{version}.joblib')
        meta_path = os.path.join(self.MODEL_DIR, f'model_v{version}_meta.json')

        joblib.dump(self.model, path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.accuracy,
            'f1_score': self.f1,
            'features': FEATURE_COLUMNS,
            'label_map': self.label_map,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return version

    @classmethod
    def load_latest(cls) -> 'MLPipeline':
        pipeline = cls()
        if not os.path.isdir(pipeline.MODEL_DIR):
            raise FileNotFoundError("No models directory found.")
        versions = []
        for f in os.listdir(pipeline.MODEL_DIR):
            if f.startswith('model_v') and f.endswith('.joblib'):
                v = f[len('model_v'):-len('.joblib')]
                versions.append(v)
        if not versions:
            raise FileNotFoundError("No saved models found.")
        latest = sorted(versions)[-1]

        pipeline.model = joblib.load(os.path.join(pipeline.MODEL_DIR, f'model_v{latest}.joblib'))
        pipeline.scaler = joblib.load(os.path.join(pipeline.MODEL_DIR, f'scaler_v{latest}.joblib'))

        meta_path = os.path.join(pipeline.MODEL_DIR, f'model_v{latest}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                pipeline.accuracy = meta.get('accuracy', 0.0)
                pipeline.f1 = meta.get('f1_score', 0.0)

        return pipeline


# =============================================================================
# 4. RECOMMENDATION ENGINE
# =============================================================================

RECOMMENDATIONS = {
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


def generate_recommendations(metrics: Dict[str, float], label: str) -> List[str]:
    recs = list(RECOMMENDATIONS.get(label, []))

    if metrics.get('loop_depth', 0) >= 3:
        recs.append("Multiple nested loops detected — consider algorithm optimization (e.g., hash maps, sorting).")
    if metrics.get('max_nesting_depth', 0) >= 4:
        recs.append("Very deep nesting — consider breaking into sub-functions or using early returns.")
    if metrics.get('function_calls', 0) > 10:
        recs.append("High number of function calls — review for potential inlining or overhead reduction.")
    if metrics.get('memory_usage_kb', 0) > 5000:
        recs.append("High memory usage detected — consider generators or streaming for large data.")

    return recs[:5]


# =============================================================================
# 5. UNIFIED PROFILER
# =============================================================================

class CodeProfiler:
    """Unified entry point for profiling code across languages."""

    def __init__(self, ml: Optional[MLPipeline] = None):
        self.ml = ml

    def profile_python(self, code: str) -> Dict[str, Any]:
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

    def profile_cpp(self, code: str) -> Dict[str, Any]:
        lines = code.split('\n')
        loop_depth = 0
        max_depth = 0
        func_calls = 0
        in_loop = False
        current_depth = 0

        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in ['for ', 'while ', 'do ']):
                in_loop = True

            for ch in stripped:
                if ch == '{' and in_loop:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif ch == '}' and in_loop:
                    current_depth -= 1
                    if current_depth <= 0:
                        in_loop = False
                        current_depth = 0

            for i, ch in enumerate(stripped):
                if ch == '(' and i > 0 and stripped[i-1].isalpha():
                    func_calls += 1

        array_count = code.count('vector') + code.count('[')
        memory = array_count * 1024 + 256
        exec_time = 0.05 + loop_depth * 0.8 + func_calls * 0.05
        complexity = round(exec_time * 0.5 + (memory / 1024) * 0.3 + max_depth * 0.2, 2)

        return {
            'language': 'cpp',
            'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory,
            'loop_depth': loop_depth,
            'max_nesting_depth': max_depth,
            'function_calls': func_calls,
            'complexity_score': complexity,
        }

    def profile_java(self, code: str) -> Dict[str, Any]:
        loops = code.count('for') + code.count('while') + code.count('do ')
        conditionals = code.count('if') + code.count('switch')
        method_calls = code.count('(')
        complexity = loops + conditionals + 1
        exec_time = 0.05 + loops * 0.6 + conditionals * 0.3 + method_calls * 0.01
        memory = 512 + loops * 128 + conditionals * 64

        return {
            'language': 'java',
            'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory,
            'loop_depth': loops,
            'conditionals': conditionals,
            'function_calls': method_calls,
            'cyclomatic_complexity': complexity,
            'complexity_score': round(complexity * 0.5 + exec_time * 0.3 + (memory / 1024) * 0.2, 2),
        }

    def profile(self, code: str, language: str = 'python') -> Dict[str, Any]:
        language = language.lower()
        if language == 'python':
            return self.profile_python(code)
        elif language == 'cpp':
            return self.profile_cpp(code)
        elif language == 'java':
            return self.profile_java(code)
        else:
            return {'error': f'Unsupported language: {language}'}

    def analyze(self, code: str, language: str = 'python') -> Dict[str, Any]:
        metrics = self.profile(code, language)
        if 'error' in metrics:
            return metrics

        result = {'metrics': metrics}

        if self.ml:
            try:
                label, confidence, probabilities = self.ml.predict(metrics)
                result['ml_prediction'] = {
                    'label': label,
                    'confidence': round(confidence, 4),
                    'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
                }
                result['recommendations'] = generate_recommendations(metrics, label)
            except RuntimeError:
                result['ml_prediction'] = {'label': 'UNKNOWN', 'confidence': 0.0}
                result['recommendations'] = []
        else:
            label = 'EFFICIENT' if metrics.get('complexity_score', 0) < 5 else \
                    'MODERATE' if metrics.get('complexity_score', 0) < 15 else 'NEEDS_OPTIMIZATION'
            result['ml_prediction'] = {'label': label, 'confidence': 0.0}
            result['recommendations'] = generate_recommendations(metrics, label)

        return result

    def export_json(self, results: List[Dict[str, Any]], filepath: str = 'profile_results.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        return filepath


# =============================================================================
# 6. DEMO / CLI ENTRY POINT
# =============================================================================

def print_report(result: Dict[str, Any]):
    metrics = result.get('metrics', {})
    pred = result.get('ml_prediction', {})
    recs = result.get('recommendations', [])

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
        print(f"  Probabilities:")
        for k, v in pred['probabilities'].items():
            print(f"    {k:>25}: {v*100:5.1f}%")

    if recs:
        print(f"\n  Recommendations:")
        for r in recs:
            print(f"    • {r}")
    print(f"{'='*50}\n")


def main():
    print("\n" + "="*58)
    print("  IntelliProfile — Production ML Code Profiler")
    print("="*58)

    os.makedirs('models', exist_ok=True)

    print("\n[1/4] Generating dataset (150 samples)...")
    df = generate_dataset(150)
    print(f"       Created {len(df)} samples across {df['label'].nunique()} classes")
    print(f"       Class distribution: {df['label'].value_counts().to_dict()}")

    print("\n[2/4] Training Random Forest model...")
    ml = MLPipeline()
    results = ml.train(df)
    print(f"       CV Accuracy:     {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print(f"       Test Accuracy:   {results['test_accuracy']:.4f}")
    print(f"       F1 (weighted):   {results['test_f1_weighted']:.4f}")
    print(f"       Precision:       {results['test_precision_weighted']:.4f}")
    print(f"       Recall:          {results['test_recall_weighted']:.4f}")

    version = ml.save()
    print(f"\n[3/4] Model saved as version '{version}'")

    profiler = CodeProfiler(ml)

    print("\n[4/4] Running sample profiles across all 3 languages...\n")

    python_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
    print_report(profiler.analyze(python_code, 'python'))

    cpp_code = """
void matrixMultiply() {
    const int N = 100;
    int A[N][N], B[N][N], C[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
"""
    print_report(profiler.analyze(cpp_code, 'cpp'))

    java_code = """
public int calculateSum(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += i;
    }
    return sum;
}
"""
    print_report(profiler.analyze(java_code, 'java'))

    all_results = [
        profiler.analyze(python_code, 'python'),
        profiler.analyze(cpp_code, 'cpp'),
        profiler.analyze(java_code, 'java'),
    ]
    out_path = profiler.export_json(all_results, 'profile_results.json')
    print(f"\nResults exported to {out_path}")

    print(f"\n{'='*58}")
    print(f"  Profile Complete! Model accuracy: {results['test_accuracy']:.2%}")
    print(f"{'='*58}\n")


if __name__ == '__main__':
    main()
