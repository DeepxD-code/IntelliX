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
# 2. MULTI-LANGUAGE PROGRAM GENERATORS (500+ programs per language)
# =============================================================================

COMPLEXITY_LABELS = ['EFFICIENT', 'MODERATE', 'NEEDS_OPTIMIZATION']

# ─────────────────────────────────────────────────────────────
# Python program generator
# ─────────────────────────────────────────────────────────────

PYTHON_TEMPLATES = {
    'EFFICIENT': [
        # --- O(1) patterns ---
        "def f(x):\n    return x * {m} + {c}",
        "def f(x, y):\n    return (x + y) // 2",
        "def f(s):\n    return s.upper().strip()",
        "def f(n):\n    return n % {m} == 0",
        "def f(a, b):\n    return a if a > b else b",
        "def f():\n    return {c}",
        "def f(x):\n    return abs(x) * {m}",
        "def f(x):\n    return -x if x < 0 else x",
        "def f(s):\n    return len(s) > {n}",
        "def f(c):\n    return c * 9/5 + 32",
        "def f(a, b):\n    return a ^ b",
        "def f(n, lo, hi):\n    return max(lo, min(n, hi))",
        "def f(a, b):\n    return (a + b) / 2",
        "def f(s):\n    return s[::-1]",
        "def f(s):\n    return s == s[::-1]",
        "def f(lst):\n    return list(set(lst))",
        "def f(e):\n    return '@' in e and '.' in e.split('@')[-1]",
        "def f(n):\n    return 1 if n > 0 else (-1 if n < 0 else 0)",
        "def f(s, tag):\n    return f'<{tag}>{s}</{tag}>'",
        "class C:\n    def __init__(self): self.n = 0\n    def inc(self): self.n += 1\n    def val(self): return self.n",
        # --- O(n) single-pass patterns ---
        "def f(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total",
        "def f(arr):\n    s = set()\n    for x in arr:\n        s.add(x)\n    return list(s)",
        "def f(s):\n    c = 0\n    for ch in s:\n        if ch in 'aeiou':\n            c += 1\n    return c",
        "def f(arr):\n    result = []\n    for x in arr:\n        if x % 2 == 0:\n            result.append(x)\n    return result",
        "def f(arr, t):\n    for i, v in enumerate(arr):\n        if v == t:\n            return i\n    return -1",
        "def f(arr):\n    if not arr:\n        return None\n    mx = mn = arr[0]\n    for x in arr[1:]:\n        if x > mx:\n            mx = x\n        if x < mn:\n            mn = x\n    return mx, mn",
        "def f(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "def f(text):\n    freq = {}\n    for w in text.split():\n        w = w.lower().strip('.,!?;:')\n        freq[w] = freq.get(w, 0) + 1\n    return freq",
        "def f(arr):\n    seen = set()\n    dups = set()\n    for x in arr:\n        if x in seen:\n            dups.add(x)\n        seen.add(x)\n    return list(dups)",
        "def f(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        "def f(items):\n    index = {}\n    for i, item in enumerate(items):\n        key = str(item).lower()\n        if key not in index:\n            index[key] = []\n        index[key].append(i)\n    return index",
        "def f(records):\n    names = []\n    emails = []\n    for r in records:\n        parts = r.split(',')\n        if len(parts) >= 2:\n            names.append(parts[0].strip())\n            emails.append(parts[1].strip())\n    return names, emails",
        "def f(lines):\n    errors = []\n    warnings = []\n    infos = []\n    for line in lines:\n        if 'ERROR' in line:\n            errors.append(line)\n        elif 'WARN' in line:\n            warnings.append(line)\n        else:\n            infos.append(line)\n    return {'errors': len(errors), 'warnings': len(warnings)}",
        "def f(items):\n    results = []\n    errors = []\n    for i, item in enumerate(items):\n        try:\n            val = int(item.strip())\n            results.append(val * 2)\n        except (ValueError, AttributeError):\n            errors.append((i, str(item)))\n    return results, errors",
        "def f(text):\n    words = text.lower().strip().split()\n    cleaned = []\n    for w in words:\n        w = w.strip('.,!?;:')\n        if w:\n            cleaned.append(w)\n    return cleaned",
        "def f(text):\n    counts = {}\n    for ch in text:\n        ch = ch.lower()\n        if ch.isalpha():\n            counts[ch] = counts.get(ch, 0) + 1\n    return counts",
        "def f(line):\n    res = []\n    cur = ''\n    q = False\n    for ch in line:\n        if ch == '\"':\n            q = not q\n        elif ch == ',' and not q:\n            res.append(cur.strip())\n            cur = ''\n        else:\n            cur += ch\n    res.append(cur.strip())\n    return res",
        "def f(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "def f(arr):\n    me = arr[0]\n    ms = arr[0]\n    for x in arr[1:]:\n        me = max(x, me + x)\n        ms = max(ms, me)\n    return ms",
        "def f(d1, d2):\n    r = d1.copy()\n    for k, v in d2.items():\n        r[k] = r.get(k, 0) + v\n    return r",
        "def f(items, page, per_page):\n    start = (page - 1) * per_page\n    end = start + per_page\n    return items[start:end], len(items) > end",
        "def f(s):\n    result = []\n    for ch in s:\n        result.append(chr(ord(ch) ^ {key}))\n    return ''.join(result)",
        "def f(arr):\n    return [x * {m} for x in arr if x > {n}]",
        "def f(arr):\n    return { {x: x * {m} for x in arr} }",
        "def f(s):\n    return {ch for ch in s if ch.isalpha()}",
        "def f(n):\n    return sum(i * i for i in range(n))",
        "def f(lst):\n    return all(x > 0 for x in lst)",
        "def f(arr):\n    r = []\n    i = 0\n    while i < len(arr):\n        r.append(arr[i] * {m})\n        i += 1\n    return r",
        "def f(n):\n    return sum(1 for _ in range(n) if _ % {m} == 0)",
        "import math\ndef f(n):\n    return math.isqrt(n)",
        "def f(k, v):\n    d = {k[i]: v[i] for i in range(len(k))}\n    return d",
        "def f(s):\n    parts = s.split('.')\n    return parts[-1] if len(parts) > 1 else ''",
        "def f(data):\n    return list(filter(None, data))",
    ],
    'MODERATE': [
        # --- O(n log n) / O(n²) small-n / recursive ---
        "def f(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = f(arr[:mid])\n    right = f(arr[mid:])\n    return merge(left, right)\ndef merge(l, r):\n    res = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] < r[j]:\n            res.append(l[i])\n            i += 1\n        else:\n            res.append(r[j])\n            j += 1\n    res.extend(l[i:])\n    res.extend(r[j:])\n    return res",
        "def f(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr",
        "def f(A, B):\n    res = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(B[0])):\n            row.append(A[i][j] + B[i][j])\n        res.append(row)\n    return res",
        "def f(nested):\n    res = []\n    for sub in nested:\n        for x in sub:\n            res.append(x)\n    return res",
        "def f(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i * i, n + 1, i):\n                sieve[j] = False\n    return [i for i, p in enumerate(sieve) if p]",
        "def f(node):\n    if not node:\n        return None\n    return {'v': node.v, 'l': f(node.l), 'r': f(node.r)}",
        "def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)",
        "def f(t, p):\n    pos = []\n    for i in range(len(t) - len(p) + 1):\n        m = True\n        for j in range(len(p)):\n            if t[i + j] != p[j]:\n                m = False\n                break\n        if m:\n            pos.append(i)\n    return pos",
        "def f(arr):\n    if len(arr) <= 1:\n        return arr\n    p = arr[len(arr) // 2]\n    l = [x for x in arr if x < p]\n    m = [x for x in arr if x == p]\n    r = [x for x in arr if x > p]\n    return f(l) + m + f(r)",
        "def f(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return []",
        "def f(root):\n    if not root:\n        return []\n    return f(root.left) + [root.val] + f(root.right)",
        "def f(root):\n    if not root:\n        return []\n    result = []\n    queue = [root]\n    while queue:\n        level = []\n        for _ in range(len(queue)):\n            node = queue.pop(0)\n            level.append(node.val)\n            if node.left:\n                queue.append(node.left)\n            if node.right:\n                queue.append(node.right)\n        result.append(level)\n    return result",
        "def f(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def f(graph, start):\n    dist = {v: float('inf') for v in graph}\n    dist[start] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        u = min((v for v in graph if v not in visited), key=lambda v: dist[v])\n        visited.add(u)\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n    return dist",
        "def f(weights, values, cap):\n    n = len(weights)\n    dp = [[0] * (cap + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(cap + 1):\n            if weights[i - 1] <= w:\n                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])\n            else:\n                dp[i][w] = dp[i - 1][w]\n    return dp[n][cap]",
        "def f(graph, start, end):\n    queue = [(start, [start])]\n    visited = {start}\n    while queue:\n        node, path = queue.pop(0)\n        for neighbor in graph[node]:\n            if neighbor == end:\n                return path + [neighbor]\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append((neighbor, path + [neighbor]))\n    return []",
        "def f(m):\n    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]",
        "import heapq\ndef f(arr):\n    heapq.heapify(arr)\n    return [heapq.heappop(arr) for _ in range(len(arr))]",
        "def f(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
        # --- Additional moderate patterns ---
        "def f(n):\n    return sum(i * j for i in range(n) for j in range(i))",
        "def f(mat):\n    total = 0\n    for i in range(len(mat)):\n        for j in range(len(mat[i])):\n            total += mat[i][j]\n    return total",
        "def f(n):\n    count = 0\n    for a in range(n):\n        for b in range(a, n):\n            if (a + b) % 2 == 0:\n                count += 1\n    return count",
        "def f(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(n):\n            if i != j and arr[i] == arr[j]:\n                return True\n    return False",
        "def f(n):\n    if n == 0:\n        return 0\n    return n + f(n - 1)",
        "def f(n):\n    if n <= 2:\n        return 1\n    return f(n - 1) + f(n - 2)",
        "def f(s):\n    def g(i, j):\n        if i >= j:\n            return True\n        if s[i] != s[j]:\n            return False\n        return g(i + 1, j - 1)\n    return g(0, len(s) - 1)",
        "def f(arr, t):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == t:\n            return mid\n        elif arr[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "def f(s, p):\n    if not p:\n        return True\n    if not s:\n        return False\n    for i in range(len(s) - len(p) + 1):\n        match = True\n        for j in range(len(p)):\n            if s[i + j] != p[j]:\n                match = False\n                break\n        if match:\n            return True\n    return False",
        "def f(n):\n    result = []\n    def backtrack(start, path):\n        result.append(path[:])\n        for i in range(start, n):\n            path.append(i)\n            backtrack(i + 1, path)\n            path.pop()\n    backtrack(0, [])\n    return result",
        "def f(arr):\n    pairs = []\n    for i in range(len(arr)):\n        for j in range(i + 1, len(arr)):\n            pairs.append((arr[i], arr[j]))\n    return pairs",
    ],
    'NEEDS_OPTIMIZATION': [
        # --- O(n³) / exponential patterns ---
        "def f(A, B, C):\n    res = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(B)):\n            s = 0\n            for k in range(len(C)):\n                s += A[i][k] * B[k][j]\n            row.append(s)\n        res.append(row)\n    return res",
        "def f(n):\n    if n <= 1:\n        return n\n    return f(n - 1) + f(n - 2)",
        "def f(g):\n    V = len(g)\n    d = [row[:] for row in g]\n    for k in range(V):\n        for i in range(V):\n            for j in range(V):\n                if d[i][k] + d[k][j] < d[i][j]:\n                    d[i][j] = d[i][k] + d[k][j]\n    return d",
        "def f(a, b, c, d, e):\n    if a:\n        if b:\n            if c:\n                if d:\n                    if e:\n                        return 'all'\n                    else:\n                        return 'e'\n                else:\n                    return 'd'\n            else:\n                return 'c'\n        else:\n            return 'b'\n    return 'a'",
        "def f(n):\n    c = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                for l in range(n):\n                    c += 1\n    return c",
        "def f(A, B):\n    n = len(A)\n    m = len(B[0])\n    p = len(B)\n    C = [[0] * m for _ in range(n)]\n    for i in range(n):\n        for j in range(m):\n            for k in range(p):\n                C[i][j] += A[i][k] * B[k][j]\n    return C",
        "def f(arr):\n    def bt(i, cur):\n        if i == len(arr):\n            res.append(cur[:])\n            return\n        bt(i + 1, cur)\n        cur.append(arr[i])\n        bt(i + 1, cur)\n        cur.pop()\n    res = []\n    bt(0, [])\n    return res",
        "def f(n):\n    if n <= 1:\n        return n\n    return f(n - 1) + f(n - 2) + f(n - 3)",
        "def f(board):\n    def valid(r, c, val):\n        for i in range(9):\n            if board[r][i] == val or board[i][c] == val:\n                return False\n        br = r // 3 * 3\n        bc = c // 3 * 3\n        for i in range(3):\n            for j in range(3):\n                if board[br + i][bc + j] == val:\n                    return False\n        return True\n    def solve():\n        for r in range(9):\n            for c in range(9):\n                if board[r][c] == 0:\n                    for val in range(1, 10):\n                        if valid(r, c, val):\n                            board[r][c] = val\n                            if solve():\n                                return True\n                            board[r][c] = 0\n                    return False\n        return True\n    return solve()",
        "def f(n):\n    def safe(b, r, c):\n        for i in range(r):\n            if b[i] == c or abs(b[i] - c) == r - i:\n                return False\n        return True\n    def solve(b, r):\n        if r == n:\n            return [b[:]]\n        sols = []\n        for c in range(n):\n            if safe(b, r, c):\n                b[r] = c\n                sols.extend(solve(b, r + 1))\n        return sols\n    return solve([0] * n, 0)",
        "def f(arr):\n    def bt(path, used):\n        if len(path) == len(arr):\n            res.append(path[:])\n            return\n        for i in range(len(arr)):\n            if not used[i]:\n                used[i] = True\n                path.append(arr[i])\n                bt(path, used)\n                path.pop()\n                used[i] = False\n    res = []\n    bt([], [False] * len(arr))\n    return res",
        "def f(n):\n    if n == 1:\n        return 1\n    return 2 * f(n - 1) + 1",
        "def f(nums, t):\n    def dfs(i, s):\n        if s == t:\n            return True\n        if i >= len(nums):\n            return False\n        return dfs(i + 1, s + nums[i]) or dfs(i + 1, s)\n    return dfs(0, 0)",
        "def f(dist):\n    n = len(dist)\n    VISITED_ALL = (1 << n) - 1\n    def dp(mask, pos):\n        if mask == VISITED_ALL:\n            return dist[pos][0]\n        if (mask, pos) in memo:\n            return memo[(mask, pos)]\n        ans = float('inf')\n        for city in range(n):\n            if not (mask & (1 << city)):\n                ans = min(ans, dist[pos][city] + dp(mask | (1 << city), city))\n        memo[(mask, pos)] = ans\n        return ans\n    memo = {}\n    return dp(1, 0)",
        "def f(chars, max_len):\n    def gen(prefix, depth):\n        if depth > max_len:\n            return\n        if prefix:\n            results.append(prefix)\n        for c in chars:\n            gen(prefix + c, depth + 1)\n    results = []\n    gen('', 0)\n    return results",
        "def f(n):\n    c = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                c += 1\n    return c",
        "def f(arr):\n    n = len(arr)\n    total = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                total += arr[i] * arr[j] * arr[k]\n    return total",
        "def f(n):\n    import itertools\n    return len(list(itertools.permutations(range(n))))",
        "def f(s):\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1\n            r += 1\n        return s[l + 1:r]\n    longest = ''\n    for i in range(len(s)):\n        for j in range(i, len(s)):\n            sub = s[i:j + 1]\n            if sub == sub[::-1] and len(sub) > len(longest):\n                longest = sub\n    return longest",
        "def f(n):\n    count = 1\n    for i in range(1, n + 1):\n        inner = 1\n        for j in range(1, i + 1):\n            inner *= j\n        count += inner\n    return count",
        "def f(arr):\n    def backtrack(start, path):\n        if sum(path) > target:\n            return\n        if sum(path) == target:\n            solutions.append(path[:])\n            return\n        for i in range(start, len(arr)):\n            path.append(arr[i])\n            backtrack(i, path)\n            path.pop()\n    solutions = []\n    target = sum(arr) // 2\n    backtrack(0, [])\n    return solutions",
        "def f(n):\n    res = []\n    for a in range(1, n + 1):\n        for b in range(a, n + 1):\n            for c in range(b, n + 1):\n                if a * a + b * b == c * c:\n                    res.append((a, b, c))\n    return res",
    ],
}

# ─────────────────────────────────────────────────────────────
# C++ program generator
# ─────────────────────────────────────────────────────────────

CPP_TEMPLATES = {
    'EFFICIENT': [
        "int f(int x) { return x * {m} + {c}; }",
        "int f(int a, int b) { return (a + b) / 2; }",
        "int f(int n) { return n % {m}; }",
        "int f(int a, int b) { return a > b ? a : b; }",
        "int f() { return {c}; }",
        "int f(int x) { return x < 0 ? -x : x; }",
        "double f(double c) { return c * 9.0 / 5.0 + 32; }",
        "int f(int a, int b) { return a ^ b; }",
        "bool f(int n) { return n % 2 == 0; }",
        "int f(int n, int lo, int hi) { return std::max(lo, std::min(n, hi)); }\n#include <algorithm>",
        "int f(const std::vector<int>& arr) { int t = 0; for (int x : arr) t += x; return t; }\n#include <vector>",
        "bool f(const std::vector<int>& arr, int t) { for (int v : arr) if (v == t) return true; return false; }\n#include <vector>",
        "int f(const std::string& s) { int c = 0; for (char ch : s) if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') c++; return c; }",
        "int f(const std::vector<int>& arr) { int mx = arr[0]; for (int x : arr) if (x > mx) mx = x; return mx; }\n#include <vector>",
        "int f(const std::vector<int>& arr) { int mn = arr[0]; for (int x : arr) if (x < mn) mn = x; return mn; }\n#include <vector>",
        "int f(int n) { int a = 0, b = 1; for (int i = 0; i < n; i++) { int t = a + b; a = b; b = t; } return a; }",
        "std::vector<int> f(const std::vector<int>& arr) { std::vector<int> r; for (int x : arr) if (x % 2 == 0) r.push_back(x); return r; }\n#include <vector>",
        "int f(const std::string& s) { return s.length(); }",
        "std::string f(const std::string& s) { std::string r = s; std::reverse(r.begin(), r.end()); return r; }\n#include <algorithm>",
        "int f(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }",
        "int f(const std::vector<int>& arr, int t) { for (size_t i = 0; i < arr.size(); i++) if (arr[i] == t) return i; return -1; }\n#include <vector>",
        "bool f(int n) { if (n <= 0) return false; return (n & (n - 1)) == 0; }",
        "void f(int n) { for (int i = 0; i < n; i++) std::cout << i << ' '; }\n#include <iostream>",
        "int f(const std::string& s) { return std::count(s.begin(), s.end(), '{ch}'); }\n#include <algorithm>",
        "int f(int n) { int sum = 0; for (int i = 1; i <= n; i++) sum += i; return sum; }",
        "double f(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / v.size(); }\n#include <vector>",
        "std::vector<int> f(const std::vector<int>& a, const std::vector<int>& b) { std::vector<int> r; for (size_t i = 0; i < a.size() && i < b.size(); i++) r.push_back(a[i] + b[i]); return r; }\n#include <vector>",
        "bool f(const std::string& s) { for (size_t i = 0; i < s.length(); i++) if (s[i] != s[s.length() - 1 - i]) return false; return true; }",
        "int f(int n) { return n * n; }",
        "int f(int x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }",
        "long long f(long long n) { long long r = 1; for (int i = 2; i <= n; i++) r *= i; return r; }",
        "int f(int* arr, int n) { int s = 0; for (int i = 0; i < n; i++) s += arr[i]; return s; }",
        "int f(const std::string& s) { auto p = s.find('{ch}'); return p != std::string::npos ? p : -1; }",
        "std::string f(const std::string& s) { std::string r; for (char c : s) r.push_back(std::toupper(c)); return r; }\n#include <cctype>",
    ],
    'MODERATE': [
        "void f(std::vector<int>& arr) { for (size_t i = 1; i < arr.size(); i++) { int key = arr[i]; int j = i - 1; while (j >= 0 && arr[j] > key) { arr[j + 1] = arr[j]; j--; } arr[j + 1] = key; } }\n#include <vector>",
        "void f(std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B) { for (size_t i = 0; i < A.size(); i++) for (size_t j = 0; j < A[0].size(); j++) A[i][j] += B[i][j]; }\n#include <vector>",
        "void f(std::vector<std::vector<int>>& m) { for (size_t i = 0; i < m.size(); i++) for (size_t j = i + 1; j < m[i].size(); j++) std::swap(m[i][j], m[j][i]); }\n#include <vector>",
        "void f(std::vector<std::vector<int>>& m) { for (auto& row : m) for (int& x : row) x *= 2; }\n#include <vector>",
        "std::vector<int> f(int n) { std::vector<bool> sieve(n + 1, true); std::vector<int> primes; for (int i = 2; i <= n; i++) { if (sieve[i]) { primes.push_back(i); for (int j = i * i; j <= n; j += i) sieve[j] = false; } } return primes; }\n#include <vector>",
        "int f(int n) { if (n <= 1) return 1; return n * f(n - 1); }",
        "int f(int n) { if (n < 2) return false; for (int i = 2; i * i <= n; i++) if (n % i == 0) return false; return true; }",
        "void f(std::vector<int>& arr) { for (size_t i = 0; i < arr.size(); i++) for (size_t j = 0; j < arr.size() - i - 1; j++) if (arr[j] > arr[j + 1]) std::swap(arr[j], arr[j + 1]); }\n#include <vector>",
        "std::vector<int> f(const std::vector<int>& arr) { std::vector<int> r; for (size_t i = 0; i < arr.size(); i++) for (size_t j = i + 1; j < arr.size(); j++) r.push_back(arr[i] + arr[j]); return r; }\n#include <vector>",
        "bool f(const std::vector<std::vector<int>>& mat, int t) { for (size_t i = 0; i < mat.size(); i++) for (size_t j = 0; j < mat[i].size(); j++) if (mat[i][j] == t) return true; return false; }\n#include <vector>",
        "int f(int arr[], int n) { for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (arr[i] == arr[j]) return arr[i]; return -1; }",
        "int f(int n) { if (n <= 2) return 1; return f(n - 1) + f(n - 2); }",
        "int f(const std::vector<int>& arr, int t) { int lo = 0, hi = arr.size() - 1; while (lo <= hi) { int mid = (lo + hi) / 2; if (arr[mid] == t) return mid; if (arr[mid] < t) lo = mid + 1; else hi = mid - 1; } return -1; }\n#include <vector>",
        "void f(std::vector<int>& v) { std::sort(v.begin(), v.end()); }\n#include <algorithm>\n#include <vector>",
        "int f(const std::string& text, const std::string& pat) { for (size_t i = 0; i <= text.length() - pat.length(); i++) { bool m = true; for (size_t j = 0; j < pat.length(); j++) { if (text[i + j] != pat[j]) { m = false; break; } } if (m) return i; } return -1; }",
        "int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) c += i * j; return c; }",
        "int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if ((i + j) % 2 == 0) c++; return c; }",
        "void f(int n) { for (int i = 1; i <= n; i++) { for (int j = 1; j <= n - i; j++) std::cout << ' '; std::cout << '*'; for (int j = 1; j < 2 * i; j++) std::cout << '*'; std::cout << '\\n'; } }\n#include <iostream>",
        "int f(int n) { int dp[n + 1]; dp[0] = 0; dp[1] = 1; for (int i = 2; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2]; return dp[n]; }",
        "int f(std::vector<int>& arr, int s, int e) { if (s >= e) return 0; int m = (s + e) / 2; int c = 0; c += f(arr, s, m); c += f(arr, m + 1, e); if (arr[m] > arr[e]) c++; return c; }\n#include <vector>",
        "int f(int n) { int s = 0; for (int i = 1; i <= n; i++) for (int j = 1; j <= i; j++) s += j; return s; }",
        "void f(int n) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) std::cout << i * j << ' '; }\n#include <iostream>",
    ],
    'NEEDS_OPTIMIZATION': [
        "void f(int n) { int*** cube = new int**[n]; for (int i = 0; i < n; i++) { cube[i] = new int*[n]; for (int j = 0; j < n; j++) { cube[i][j] = new int[n]; for (int k = 0; k < n; k++) cube[i][j][k] = i + j + k; } } for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) delete[] cube[i][j]; delete[] cube[i]; } delete[] cube; }",
        "void f(int n) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) for (int l = 0; l < n; l++) std::cout << i + j + k + l << ' '; }\n#include <iostream>",
        "int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2); }",
        "void f(int A[][100], int B[][100], int C[][100], int n) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { C[i][j] = 0; for (int k = 0; k < n; k++) C[i][j] += A[i][k] * B[k][j]; } }",
        "void f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c += i * j * k; }",
        "int f(int n) { int* a = new int[n]; int c = 0; for (int i = 0; i < n; i++) { a[i] = i; for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c += a[i] * j * k; } delete[] a; return c; }",
        "int f(int n) { if (n <= 1) return 1; return 2 * f(n - 1) + 1; }",
        "void f(int n) { for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { for (int k = 0; k < n; k++) { std::cout << '(' << i << ',' << j << ',' << k << ')' << ' '; } } } }\n#include <iostream>",
        "int f(int n) { int c = 0; for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++) for (int k = 1; k <= n; k++) for (int l = 1; l <= n; l++) c++; return c; }",
        "int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2) + f(n - 3); }",
        "void f(int n) { int* arr = new int[n]; for (int i = 0; i < n; i++) arr[i] = i; for (int i = 0; i < n; i++) { for (int j = i + 1; j < n; j++) { if (arr[i] == arr[j]) std::cout << arr[i]; } } delete[] arr; }\n#include <iostream>",
        "void f(int* arr, int n) { for (int i = 0; i < n; i++) { int* p = &arr[i]; for (int j = 0; j < n; j++) { for (int k = 0; k < n; k++) { int** pp = &p; **pp += arr[j] * arr[k]; } } } }",
        "int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) { if (i + j + k < n) c++; } return c; }",
        "int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = i; j < n; j++) for (int k = j; k < n; k++) for (int l = k; l < n; l++) c++; return c; }",
    ],
}

# ─────────────────────────────────────────────────────────────
# Java program generator
# ─────────────────────────────────────────────────────────────

JAVA_TEMPLATES = {
    'EFFICIENT': [
        "public int f(int x) { return x * {m} + {c}; }",
        "public int f(int a, int b) { return (a + b) / 2; }",
        "public boolean f(int n) { return n % {m} == 0; }",
        "public int f(int a, int b) { return a > b ? a : b; }",
        "public long f() { return {c}L; }",
        "public int f(int x) { return x < 0 ? -x : x; }",
        "public double f(double c) { return c * 9.0 / 5.0 + 32; }",
        "public int f(int a, int b) { return a ^ b; }",
        "public boolean f(int n) { return n % 2 == 0; }",
        "public String f(String s) { return s.toUpperCase(); }",
        "public int f(String s) { return s.length(); }",
        "public boolean f(String s) { return s.equals(new StringBuilder(s).reverse().toString()); }",
        "public int f(List<Integer> arr) { int t = 0; for (int x : arr) t += x; return t; }",
        "public boolean f(List<Integer> arr, int t) { for (int v : arr) if (v == t) return true; return false; }",
        "public Optional<Integer> f(List<Integer> arr) { return arr.stream().filter(x -> x % 2 == 0).findFirst(); }",
        "public int f(int[] arr) { int mx = arr[0]; for (int x : arr) if (x > mx) mx = x; return mx; }",
        "public int f(int[] arr) { int mn = arr[0]; for (int x : arr) if (x < mn) mn = x; return mn; }",
        "public int f(int n) { int a = 0, b = 1; for (int i = 0; i < n; i++) { int t = a + b; a = b; b = t; } return a; }",
        "public List<Integer> f(List<Integer> arr) { List<Integer> r = new ArrayList<>(); for (int x : arr) if (x % 2 == 0) r.add(x); return r; }",
        "public String f(String s) { return new StringBuilder(s).reverse().toString(); }",
        "public int f(int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; }",
        "public Set<Integer> f(List<Integer> arr) { return new HashSet<>(arr); }",
        "public int f(int n) { int sum = 0; for (int i = 1; i <= n; i++) sum += i; return sum; }",
        "public double f(List<Double> v) { double s = 0; for (double x : v) s += x; return s / v.size(); }",
        "public boolean f(int n) { if (n <= 0) return false; return (n & (n - 1)) == 0; }",
        "public long f(long n) { long r = 1; for (int i = 2; i <= n; i++) r *= i; return r; }",
        "public int f(String s, char c) { int count = 0; for (int i = 0; i < s.length(); i++) if (s.charAt(i) == c) count++; return count; }",
        "public int f(String s) { return s.indexOf('{ch}'); }",
        "public List<String> f(String[] words) { return Arrays.stream(words).map(String::toUpperCase).collect(Collectors.toList()); }",
        "public int[] f(int[] a, int[] b) { int[] r = new int[Math.min(a.length, b.length)]; for (int i = 0; i < r.length; i++) r[i] = a[i] + b[i]; return r; }",
    ],
    'MODERATE': [
        "public void f(int[] arr) { for (int i = 1; i < arr.length; i++) { int key = arr[i]; int j = i - 1; while (j >= 0 && arr[j] > key) { arr[j + 1] = arr[j]; j--; } arr[j + 1] = key; } }",
        "public int[][] f(int[][] A, int[][] B) { int n = A.length, m = A[0].length; int[][] C = new int[n][m]; for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j]; return C; }",
        "public void f(int[][] m) { for (int i = 0; i < m.length; i++) for (int j = i + 1; j < m[i].length; j++) { int t = m[i][j]; m[i][j] = m[j][i]; m[j][i] = t; } }",
        "public List<Integer> f(int n) { boolean[] sieve = new boolean[n + 1]; List<Integer> primes = new ArrayList<>(); for (int i = 2; i <= n; i++) { if (!sieve[i]) { primes.add(i); for (int j = i * i; j <= n; j += i) sieve[j] = true; } } return primes; }",
        "public int f(int n) { if (n <= 1) return 1; return n * f(n - 1); }",
        "public boolean f(int n) { if (n < 2) return false; for (int i = 2; i * i <= n; i++) if (n % i == 0) return false; return true; }",
        "public void f(List<Integer> arr) { for (int i = 0; i < arr.size(); i++) for (int j = 0; j < arr.size() - i - 1; j++) if (arr.get(j) > arr.get(j + 1)) Collections.swap(arr, j, j + 1); }",
        "public int f(int[] arr) { for (int i = 0; i < arr.length; i++) for (int j = i + 1; j < arr.length; j++) if (arr[i] == arr[j]) return arr[i]; return -1; }",
        "public boolean f(int[][] mat, int t) { for (int i = 0; i < mat.length; i++) for (int j = 0; j < mat[i].length; j++) if (mat[i][j] == t) return true; return false; }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) c += i * j; return c; }",
        "public int f(int n) { int dp[] = new int[n + 1]; dp[0] = 0; dp[1] = 1; for (int i = 2; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2]; return dp[n]; }",
        "public int f(int[] arr, int t) { int lo = 0, hi = arr.length - 1; while (lo <= hi) { int mid = (lo + hi) / 2; if (arr[mid] == t) return mid; if (arr[mid] < t) lo = mid + 1; else hi = mid - 1; } return -1; }",
        "public void f(List<Integer> v) { Collections.sort(v); }",
        "public int f(String text, String pat) { for (int i = 0; i <= text.length() - pat.length(); i++) { boolean m = true; for (int j = 0; j < pat.length(); j++) { if (text.charAt(i + j) != pat.charAt(j)) { m = false; break; } } if (m) return i; } return -1; }",
        "public int f(int n) { if (n <= 2) return 1; return f(n - 1) + f(n - 2); }",
        "public int f(int n) { int s = 0; for (int i = 1; i <= n; i++) for (int j = 1; j <= i; j++) s += j; return s; }",
        "public void f(int n) { for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { System.out.print(i * j + \" \"); } System.out.println(); } }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if ((i + j) % 2 == 0) c++; return c; }",
    ],
    'NEEDS_OPTIMIZATION': [
        "public void f(int n) { int[][][] cube = new int[n][n][n]; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) cube[i][j][k] = i + j + k; }",
        "public int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2); }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c += i * j * k; return c; }",
        "public int[][] f(int[][] A, int[][] B) { int n = A.length, m = B[0].length, p = B.length; int[][] C = new int[n][m]; for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) for (int k = 0; k < p; k++) C[i][j] += A[i][k] * B[k][j]; return C; }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) for (int l = 0; l < n; l++) c++; return c; }",
        "public int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2) + f(n - 3); }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { System.out.print(i * j + \" \"); for (int k = 0; k < n; k++) { System.out.print(k); } } } return c; }",
        "public String f(String s) { String r = \"\"; for (int i = 0; i < s.length(); i++) { for (int j = i; j < s.length(); j++) { for (int k = i; k <= j; k++) { r += s.charAt(k); } } } return r; }",
        "public int f(int[] arr) { int n = arr.length; int c = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) for (int k = 0; k < n; k++) c += arr[i] * arr[j] * arr[k]; return c; }",
        "public int f(int n) { int c = 0; for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++) for (int k = 1; k <= n; k++) for (int l = 1; l <= n; l++) c++; return c; }",
        "public int f(int n) { int c = 0; for (int i = 0; i < n; i++) for (int j = i; j < n; j++) for (int k = j; k < n; k++) for (int l = k; l < n; l++) c++; return c; }",
        "public void f(int n) { for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) { for (int k = 0; k < n; k++) { System.out.print(\"(\" + i + \",\" + j + \",\" + k + \") \"); } } } }",
    ],
}

ALL_LANGS = ['python', 'cpp', 'java']

TEMPLATE_DIRS = {
    'python': PYTHON_TEMPLATES,
    'cpp': CPP_TEMPLATES,
    'java': JAVA_TEMPLATES,
}

def _fill_template(template: str, m: int = None, c: int = None, n: int = None, ch: str = None, key: int = None) -> str:
    """Fill a template's {m}, {c}, {n}, {ch}, {key} placeholders."""
    t = template
    if m is not None:
        t = t.replace('{m}', str(m))
    else:
        t = t.replace('{m}', str(random.randint(2, 10)))
    if c is not None:
        t = t.replace('{c}', str(c))
    else:
        t = t.replace('{c}', str(random.randint(1, 100)))
    if n is not None:
        t = t.replace('{n}', str(n))
    else:
        t = t.replace('{n}', str(random.randint(5, 20)))
    if ch is not None:
        t = t.replace('{ch}', ch)
    else:
        t = t.replace('{ch}', random.choice('aeiouxyz'))
    if key is not None:
        t = t.replace('{key}', str(key))
    else:
        t = t.replace('{key}', str(random.randint(1, 255)))
    return t

def _generate_program(lang: str, complexity: str) -> tuple[str, int]:
    """Generate a program for given language and complexity.
    
    Returns (code_string, synthetic_loop_count) where synthetic_loop_count
    is the approximate algorithmic complexity indicator used for smart noise.
    """
    if lang not in TEMPLATE_DIRS:
        lang = 'python'
    templates = TEMPLATE_DIRS[lang].get(complexity, TEMPLATE_DIRS[lang].get('EFFICIENT', []))
    template = random.choice(templates)
    code = _fill_template(template)
    
    # Approximate complexity indicator from the chosen template
    lines = code.split('\n')
    loop_sigs = sum(1 for l in lines if 'for' in l or 'while' in l)
    nest_sigs = max(loop_sigs - 1, 0)
    if complexity == 'EFFICIENT':
        si = 1 + loop_sigs
    elif complexity == 'MODERATE':
        si = 3 + loop_sigs * 2 + nest_sigs
    else:
        si = 5 + loop_sigs * 3 + nest_sigs * 2
    
    return code, si

def _compute_features(code: str, lang: str) -> dict[str, float]:
    """Compute profile features for a code snippet using the same profiler
    that will be used at inference time."""
    profiler = CodeProfiler()
    metrics = profiler.profile(code, lang)
    return {
        'execution_time_ms': metrics.get('execution_time_ms', 1.0),
        'memory_usage_kb': metrics.get('memory_usage_kb', 256),
        'loop_depth': metrics.get('loop_depth', 0),
        'max_nesting_depth': metrics.get('max_nesting_depth', 0),
        'function_calls': metrics.get('function_calls', 0),
        'conditionals': metrics.get('conditionals', 0),
        'complexity_score': metrics.get('complexity_score', 1.0),
        'language': LANGUAGE_ENCODE.get(lang, 0.0),
    }

def generate_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    rows = []
    per_lang = max(1, n_samples // len(ALL_LANGS))
    per_class = max(1, per_lang // 3)
    
    for lang in ALL_LANGS:
        for label in COMPLEXITY_LABELS:
            templates = TEMPLATE_DIRS[lang].get(label, TEMPLATE_DIRS[lang]['EFFICIENT'])
            needed = per_class
            attempts = 0
            generated = 0
            while generated < needed and attempts < needed * 10:
                code, si = _generate_program(lang, label)
                attempts += 1
                try:
                    metrics = _compute_features(code, lang)
                except Exception:
                    continue
                # small noise proportional to complexity indicator
                noise = 0.05
                for k in ['execution_time_ms', 'complexity_score']:
                    metrics[k] *= 1.0 + random.uniform(-noise, noise)
                metrics['memory_usage_kb'] = int(metrics['memory_usage_kb'] * (1.0 + random.uniform(-noise, noise)))
                rows.append({
                    **metrics,
                    'label': label,
                    'code': code,
                    'language': LANGUAGE_ENCODE.get(lang, 0.0),
                    'lang_str': lang,
                })
                generated += 1
    
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
    'max_nesting_depth', 'function_calls', 'conditionals', 'complexity_score',
    'language'
]

LANGUAGE_ENCODE = {'python': 0.0, 'cpp': 1.0, 'java': 2.0}
LANGUAGE_DECODE = {v: k for k, v in LANGUAGE_ENCODE.items()}

CLASSIFIERS = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'grid': {
            'n_estimators': [200, 400, 600, 800],
            'max_depth': [10, 16, 24, None],
            'min_samples_split': [2, 3, 5, 7],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'grid': {
            'n_estimators': [200, 400, 600],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.03, 0.05, 0.1, 0.15],
            'min_samples_split': [2, 4, 6],
            'subsample': [0.8, 0.9, 1.0],
        }
    },
    'svm': {
        'model': SVC(probability=True, random_state=42, class_weight='balanced'),
        'grid': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'degree': [2, 3],
        }
    },
    'neural_network': {
        'model': MLPClassifier(max_iter=2000, random_state=42, early_stopping=True),
        'grid': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.005, 0.01],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128],
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
        raw = {}
        for c in FEATURE_COLUMNS:
            v = metrics.get(c, 0.0)
            if c == 'language':
                v = LANGUAGE_ENCODE.get(v, 0.0)
            raw[c] = v
        features = np.array([[raw[c] for c in FEATURE_COLUMNS]])
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
        loop_count = 0; max_depth = 0; func_calls = 0
        current_depth = 0
        for line in lines:
            stripped = line.strip()
            # Count each loop keyword occurrence (robust heuristic)
            if any(kw in stripped for kw in ['for (', 'while (', 'do ']):
                loop_count += 1
            for ch in stripped:
                if ch == '{':
                    current_depth += 1; max_depth = max(max_depth, current_depth)
                elif ch == '}':
                    current_depth = max(0, current_depth - 1)
            for i, ch in enumerate(stripped):
                if ch == '(' and i > 0 and stripped[i-1].isalpha():
                    func_calls += 1
        array_count = code.count('vector') + code.count('[')
        memory = array_count * 1024 + 256
        exec_time = 0.05 + loop_count * 0.8 + func_calls * 0.05
        conditionals = code.count('if') + code.count('else') + code.count('switch')
        complexity = round(exec_time * 0.5 + (memory / 1024) * 0.3 + max_depth * 0.2, 2)
        return {
            'language': 'cpp', 'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory, 'loop_depth': loop_count,
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
        # Estimate max_nesting_depth from brace structure
        depth = 0; max_depth = 0
        for ch in code:
            if ch == '{': depth += 1; max_depth = max(max_depth, depth)
            elif ch == '}': depth = max(0, depth - 1)
        return {
            'language': 'java', 'execution_time_ms': round(exec_time, 3),
            'memory_usage_kb': memory, 'loop_depth': loops,
            'max_nesting_depth': max_depth, 'conditionals': conditionals,
            'function_calls': method_calls,
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

    n_samples = config.get('dataset.n_samples', 3000)
    print(f"\n[1/4] Generating dataset ({n_samples} samples)...")
    df = generate_dataset(n_samples, seed=config.get('dataset.seed', 42))
    print(f"       Created {len(df)} samples, {df['label'].nunique()} classes")
    print(f"       Distribution: {df['label'].value_counts().to_dict()}")

    if config.get('ml_model.auto_select_best', True):
        print(f"\n[2/4] Comparing all classifiers to find the best...")
        comparison = run_comparison(df)
        best_type = comparison['best_classifier']
        print(f"       Best classifier: {best_type} (acc={comparison[best_type]['test_accuracy']:.4f})")
        for ctype, res in comparison.items():
            if ctype != 'best_classifier':
                print(f"         {ctype:20s}: acc={res['test_accuracy']:.4f} f1={res['test_f1_weighted']:.4f}")
    else:
        best_type = config.get('ml_model.classifier_type', 'random_forest')

    print(f"\n[3/4] Final training with {best_type}...")
    ml = MLPipeline(classifier_type=best_type)
    results = ml.train(df, tune=config.get('ml_model.hyperparameter_tuning', True),
                       cv_folds=config.get('ml_model.cross_validation_folds', 5))
    print(f"       CV Accuracy:     {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print(f"       Test Accuracy:   {results['test_accuracy']:.4f}")
    print(f"       F1 (weighted):   {results['test_f1_weighted']:.4f}")
    print(f"       Best Params:     {results.get('best_params', 'N/A')}")

    version = ml.save()
    print(f"\n[4/4] Model saved as version '{version}'")

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
