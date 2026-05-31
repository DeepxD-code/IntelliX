"""Integration tests for app.py Flask API endpoints."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import DB_PATH, app


def setup_module():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def add(self, name, status, detail=None):
        if status == 'PASS':
            self.passed += 1
            print(f"    V {name}")
        else:
            self.failed += 1
            self.failures.append((name, detail or ''))
            print(f"    X {name}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n  API Tests: {total} total, {self.passed} passed, {self.failed} failed")
        for name, detail in self.failures:
            print(f"    FAIL: {name}")
            if detail:
                print(f"      -> {detail}")
        return total == self.passed


def test_health_endpoint(t):
    with app.test_client() as c:
        resp = c.get('/api/health')
        assert resp.status_code == 200
        data = resp.get_json()
        t.add("GET /api/health returns 200", "PASS")
        if data.get('status') == 'healthy':
            t.add("health.status == healthy", "PASS")
        else:
            t.add("health.status == healthy", "FAIL", str(data))
        if 'model_accuracy' in data:
            t.add("health has model_accuracy", "PASS")
        else:
            t.add("health has model_accuracy", "FAIL")


def test_profile_endpoint(t):
    with app.test_client() as c:
        resp = c.post('/api/profile', json={})
        t.add("POST /api/profile empty body -> 400", "PASS" if resp.status_code == 400 else "FAIL")

        resp = c.post('/api/profile', json={'code': ''})
        t.add("POST /api/profile empty code -> 400", "PASS" if resp.status_code == 400 else "FAIL")

        resp = c.post('/api/profile', json={'code': 'def add(a, b): return a + b', 'language': 'python'})
        if resp.status_code == 200:
            data = resp.get_json()
            t.add("POST /api/profile python 200", "PASS")
            if 'metrics' in data and 'ml_prediction' in data:
                t.add("profile has metrics + ml_prediction", "PASS")
            else:
                t.add("profile has metrics + ml_prediction", "FAIL", str(list(data.keys())))
            if 'recommendations' in data:
                t.add("profile has recommendations", "PASS")
            else:
                t.add("profile has recommendations", "FAIL")
        else:
            t.add("POST /api/profile python 200", "FAIL", f"status={resp.status_code}")

        resp = c.post('/api/profile', json={'code': 'int main() { return 0; }', 'language': 'cpp'})
        if resp.status_code == 200:
            t.add("POST /api/profile cpp 200", "PASS")
        else:
            t.add("POST /api/profile cpp 200", "FAIL", f"status={resp.status_code}")

        resp = c.post('/api/profile', json={'code': 'class Foo { public int bar() { return 1; } }', 'language': 'java'})
        if resp.status_code == 200:
            t.add("POST /api/profile java 200", "PASS")
        else:
            t.add("POST /api/profile java 200", "FAIL", f"status={resp.status_code}")

        resp = c.post('/api/profile', json={'code': 'x=1', 'language': 'ruby'})
        if resp.status_code == 400:
            t.add("POST /api/profile unsupported lang -> 400", "PASS")
        else:
            t.add("POST /api/profile unsupported lang -> 400", "FAIL", f"status={resp.status_code}")


def test_batch_profile_endpoint(t):
    with app.test_client() as c:
        resp = c.post('/api/batch-profile', json={})
        t.add("POST /api/batch-profile empty -> 400", "PASS" if resp.status_code == 400 else "FAIL")

        resp = c.post('/api/batch-profile', json={'snippets': [
            {'code': 'x=1', 'language': 'python'},
            {'code': 'int main() { return 0; }', 'language': 'cpp'},
            {'code': 'class A { void m() { } }', 'language': 'java'},
            {'code': '', 'language': 'python'},
        ]})
        if resp.status_code == 200:
            data = resp.get_json()
            if data.get('total') == 4 and data.get('successful') == 3 and data.get('failed') == 1:
                t.add("batch-profile 4 snippets (3 ok, 1 empty)", "PASS")
            else:
                t.add("batch-profile 4 snippets (3 ok, 1 empty)", "FAIL", str(data))
        else:
            t.add("batch-profile 4 snippets (3 ok, 1 empty)", "FAIL", f"status={resp.status_code}")


def test_model_info_endpoint(t):
    with app.test_client() as c:
        resp = c.get('/api/model-info')
        assert resp.status_code == 200
        data = resp.get_json()
        if 'model_type' in data and 'features' in data and 'classes' in data:
            t.add("GET /api/model-info has all fields", "PASS")
        else:
            t.add("GET /api/model-info has all fields", "FAIL", str(list(data.keys())))
        if 'accuracy' in data:
            t.add("model-info has accuracy", "PASS")
        else:
            t.add("model-info has accuracy", "FAIL")


def test_history_endpoint(t):
    with app.test_client() as c:
        resp = c.get('/api/history')
        assert resp.status_code == 200
        data = resp.get_json()
        if 'total' in data and 'results' in data:
            t.add("GET /api/history has total + results", "PASS")
        else:
            t.add("GET /api/history has total + results", "FAIL", str(list(data.keys())))

        # Profile something first so history isn't empty
        c.post('/api/profile', json={'code': 'def f(): return 1', 'language': 'python'})
        resp = c.get('/api/history?limit=10')
        data = resp.get_json()
        if data.get('total', 0) >= 1:
            t.add("History has recorded profiles", "PASS")
        else:
            t.add("History has recorded profiles", "FAIL", f"total={data.get('total')}")


def test_index_page(t):
    with app.test_client() as c:
        resp = c.get('/')
        if resp.status_code == 200:
            t.add("GET / serves index.html", "PASS")
        else:
            t.add("GET / serves index.html", "FAIL", f"status={resp.status_code}")


def run_api_tests():
    t = TestResult()
    print("\n--- API Integration Tests ---\n")
    test_health_endpoint(t)
    print()
    test_profile_endpoint(t)
    print()
    test_batch_profile_endpoint(t)
    print()
    test_model_info_endpoint(t)
    print()
    test_history_endpoint(t)
    print()
    test_index_page(t)
    print()
    ok = t.summary()
    return ok


if __name__ == '__main__':
    ok = run_api_tests()
    sys.exit(0 if ok else 1)
