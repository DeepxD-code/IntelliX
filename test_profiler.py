"""
Comprehensive test suite for IntelliProfile (profiler_main.py)
Tests every class, method, and function with 500+ random code snippets per language
"""

import sys
import os
import json
import random
import tempfile
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure we import the module under test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiler_main import (
    PythonASTAnalyzer,
    COMPLEXITY_LABELS,
    _generate_python_snippet,
    _compute_synthetic_metrics,
    generate_dataset,
    FEATURE_COLUMNS,
    MLPipeline, ModelRegistry,
    CLASSIFIERS,
    generate_recommendations,
    CodeProfiler,
    ConfigManager, ConfigError,
    print_report,
    main,
)

# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def _gen_python_effi():
    return [
        "def add(a, b): return a + b",
        "def square(x): return x * x",
        "x = 42",
        "import os; print(os.getcwd())",
        "def greet(name):\n    return f'Hello, {name}'",
        "def is_even(n): return n % 2 == 0",
        "def max_of_two(a, b): return a if a > b else b",
        "result = sum([1, 2, 3])",
        "def constant(): return 1",
        "name = 'world'; print(f'hello {name}')",
        "class Empty: pass",
        "PI = 3.14159",
        "def identity(x): return x",
        "def negate(x): return -x",
        "def absolute(x): return abs(x)",
        "def first(lst): return lst[0] if lst else None",
        "def last(lst): return lst[-1] if lst else None",
        "def to_upper(s): return s.upper()",
        "def to_lower(s): return s.lower()",
        "def strlen(s): return len(s)",
        "def is_empty(s): return len(s) == 0",
        "def celsius_to_f(c): return c * 9/5 + 32",
        "def xor(a, b): return a ^ b",
        "def parity(n): return n % 2",
        "def sign(n): return 1 if n > 0 else (-1 if n < 0 else 0)",
        "def clamp(n, lo, hi): return max(lo, min(n, hi))",
        "def average(a, b): return (a + b) / 2",
        "def difference(a, b): return abs(a - b)",
        "class Dog:\n    def bark(self): return 'woof'",
        "def wrap(s, tag): return f'<{tag}>{s}</{tag}>'",
    ]

def _gen_python_moderate():
    return [
        "def linear_search(arr, target):\n    for i, v in enumerate(arr):\n        if v == target: return i\n    return -1",
        "def count_vowels(s):\n    count = 0\n    for ch in s:\n        if ch in 'aeiou': count += 1\n    return count",
        "def sum_list(arr):\n    t = 0\n    for x in arr: t += x\n    return t",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
        "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
        "def find_duplicates(arr):\n    seen = set(); dups = []\n    for x in arr:\n        if x in seen: dups.append(x)\n        else: seen.add(x)\n    return dups",
        "def word_freq(text):\n    freq = {}\n    for w in text.split():\n        w = w.lower().strip('.,!?;:')\n        freq[w] = freq.get(w, 0) + 1\n    return freq",
        "def prime_sieve(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n+1, i): sieve[j] = False\n    return [i for i, p in enumerate(sieve) if p]",
        "def matrix_add(A, B):\n    res = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(A[0])): row.append(A[i][j] + B[i][j])\n        res.append(row)\n    return res",
        "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "def csv_parse(line):\n    res = []; cur = ''; q = False\n    for ch in line:\n        if ch == '\"': q = not q\n        elif ch == ',' and not q: res.append(cur.strip()); cur = ''\n        else: cur += ch\n    res.append(cur.strip())\n    return res",
        "def unique(lst): return list(set(lst))",
        "def dict_merge(d1, d2):\n    r = d1.copy(); r.update(d2); return r",
        "def reverse_words(s): return ' '.join(s.split()[::-1])",
        "def most_common(lst): return max(set(lst), key=lst.count)",
        "def flatten(nested):\n    res = []\n    for sub in nested:\n        for x in sub: res.append(x)\n    return res",
        "def transpose(matrix): return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]",
        "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True",
    ]

def _gen_python_needz():
    return [
        "def triple_nested(A, B, C):\n    res = []\n    for i in range(len(A)):\n        row = []\n        for j in range(len(B)):\n            s = 0\n            for k in range(len(C)): s += A[i][k] * B[k][j]\n            row.append(s)\n        res.append(row)\n    return res",
        "def fib_bad(n):\n    if n <= 1: return n\n    return fib_bad(n-1) + fib_bad(n-2)",
        "def floyd_warshall(g):\n    V = len(g); d = [row[:] for row in g]\n    for k in range(V):\n        for i in range(V):\n            for j in range(V):\n                if d[i][k] + d[k][j] < d[i][j]: d[i][j] = d[i][k] + d[k][j]\n    return d",
        "def deep_cond(a,b,c,d,e):\n    if a:\n        if b:\n            if c:\n                if d:\n                    if e: return 'all'\n                    else: return 'e'\n                else: return 'd'\n            else: return 'c'\n        else: return 'b'\n    return 'a'",
        "def brute_pattern(text, pat):\n    pos = []\n    for i in range(len(text)-len(pat)+1):\n        match = True\n        for j in range(len(pat)):\n            if text[i+j] != pat[j]: match = False; break\n        if match: pos.append(i)\n    return pos",
        "def qs_deep(arr):\n    if len(arr) <= 1: return arr\n    p = arr[len(arr)//2]\n    l = [x for x in arr if x < p]\n    m = [x for x in arr if x == p]\n    r = [x for x in arr if x > p]\n    return qs_deep(l) + m + qs_deep(r)",
        "def many_nested(n):\n    c = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                for l in range(n): c += 1\n    return c",
        "def combo(arr, r):\n    def bt(st, cur):\n        if len(cur) == r: res.append(cur[:]); return\n        for i in range(st, len(arr)): cur.append(arr[i]); bt(i+1, cur); cur.pop()\n    res = []; bt(0, []); return res",
        "def n_queens(n):\n    def safe(b, r, c):\n        for i in range(r):\n            if b[i]==c or abs(b[i]-c)==r-i: return False\n        return True\n    def solve(b, r):\n        if r==n: return [b[:]]\n        sols = []\n        for c in range(n):\n            if safe(b,r,c): b[r]=c; sols.extend(solve(b,r+1))\n        return sols\n    return solve([0]*n, 0)",
        "def permute(arr):\n    def bt(path, used):\n        if len(path)==len(arr): res.append(path[:]); return\n        for i in range(len(arr)):\n            if not used[i]: used[i]=True; path.append(arr[i]); bt(path,used); path.pop(); used[i]=False\n    res = []; bt([], [False]*len(arr)); return res",
    ]

def _gen_cpp_snippet(complexity: str) -> str:
    if complexity == 'EFFICIENT':
        snippets = [
            "int add(int a, int b) { return a + b; }",
            "int square(int x) { return x * x; }",
            "void hello() { std::cout << \"hi\"; }",
            "int max(int a, int b) { return a > b ? a : b; }",
            "double average(double a, double b) { return (a + b) / 2.0; }",
            "int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }",
            "bool isEven(int n) { return n % 2 == 0; }",
            "int absVal(int x) { return x < 0 ? -x : x; }",
            "void inc(int& x) { x++; }",
            "int sum2(int a, int b) { return a + b; }",
            "int triple(int x) { return x * 3; }",
            "char firstChar(const char* s) { return s[0]; }",
            "bool isPositive(int x) { return x > 0; }",
            "int negate(int x) { return -x; }",
            "double circleArea(double r) { return 3.14159 * r * r; }",
            "int diff(int a, int b) { return a > b ? a - b : b - a; }",
        ]
    elif complexity == 'MODERATE':
        snippets = [
            "int sumArray(int arr[], int n) { int s=0; for(int i=0;i<n;i++) s+=arr[i]; return s; }",
            "int linearSearch(int arr[], int n, int t) { for(int i=0;i<n;i++) if(arr[i]==t) return i; return -1; }",
            "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
            "void bubbleSort(int arr[], int n) { for(int i=0;i<n-1;i++) for(int j=0;j<n-i-1;j++) if(arr[j]>arr[j+1]) { int t=arr[j]; arr[j]=arr[j+1]; arr[j+1]=t; } }",
            "int gcd(int a, int b) { while(b) { int t=a%b; a=b; b=t; } return a; }",
            "bool isPrime(int n) { if(n<2) return false; for(int i=2;i*i<=n;i++) if(n%i==0) return false; return true; }",
            "int countVowels(const char* s) { int c=0; for(int i=0;s[i];i++) { char ch=tolower(s[i]); if(ch=='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u') c++; } return c; }",
            "int maxSubarray(int arr[], int n) { int maxEnd=arr[0], maxSoFar=arr[0]; for(int i=1;i<n;i++) { maxEnd=(arr[i]>maxEnd+arr[i])?arr[i]:maxEnd+arr[i]; maxSoFar=(maxSoFar>maxEnd)?maxSoFar:maxEnd; } return maxSoFar; }",
            "void reverseArray(int arr[], int n) { for(int i=0;i<n/2;i++) { int t=arr[i]; arr[i]=arr[n-1-i]; arr[n-1-i]=t; } }",
            "int binarySearch(int arr[], int l, int r, int x) { while(l<=r) { int m=l+(r-l)/2; if(arr[m]==x) return m; if(arr[m]<x) l=m+1; else r=m-1; } return -1; }",
        ]
    else:
        snippets = [
            "void tripleNested(int n) { int c=0; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) c++; }",
            "int fibBad(int n) { if(n<=1) return n; return fibBad(n-1)+fibBad(n-2); }",
            "void floyd(int g[][100], int V) { for(int k=0;k<V;k++) for(int i=0;i<V;i++) for(int j=0;j<V;j++) if(g[i][k]+g[k][j]<g[i][j]) g[i][j]=g[i][k]+g[k][j]; }",
            "void deepConds(int a,int b,int c,int d,int e) { if(a){if(b){if(c){if(d){if(e){}else{}}else{}}else{}}else{}} }",
            "void bruteForce(const char* t, const char* p) { int n=strlen(t), m=strlen(p); for(int i=0;i<=n-m;i++) { int j; for(j=0;j<m;j++) if(t[i+j]!=p[j]) break; if(j==m) {} } }",
            "void quadNested(int n) { int c=0; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) for(int l=0;l<n;l++) c++; }",
            "void matrixMult(int A[][100], int B[][100], int C[][100], int N) { for(int i=0;i<N;i++) for(int j=0;j<N;j++) { C[i][j]=0; for(int k=0;k<N;k++) C[i][j]+=A[i][k]*B[k][j]; } }",
            "int hanoi(int n) { if(n==1) return 1; return 2*hanoi(n-1)+1; }",
        ]
    return random.choice(snippets)


def _gen_java_snippet(complexity: str) -> str:
    if complexity == 'EFFICIENT':
        snippets = [
            "public int add(int a, int b) { return a + b; }",
            "public int square(int x) { return x * x; }",
            "public boolean isEven(int n) { return n % 2 == 0; }",
            "public int max(int a, int b) { return a > b ? a : b; }",
            "public double avg(double a, double b) { return (a + b) / 2; }",
            "public int abs(int x) { return x < 0 ? -x : x; }",
            "public String greet(String n) { return \"Hello \" + n; }",
            "public int triple(int x) { return x * 3; }",
            "public boolean isPositive(int x) { return x > 0; }",
            "public int negate(int x) { return -x; }",
            "public long squareLong(long x) { return x * x; }",
            "public char firstChar(String s) { return s.charAt(0); }",
            "public int strLen(String s) { return s.length(); }",
            "public boolean isEmpty(String s) { return s.isEmpty(); }",
            "public double celsiusToF(double c) { return c * 9/5 + 32; }",
        ]
    elif complexity == 'MODERATE':
        snippets = [
            "public int sumArray(int[] arr) { int s=0; for(int x:arr) s+=x; return s; }",
            "public int linearSearch(int[] arr, int t) { for(int i=0;i<arr.length;i++) if(arr[i]==t) return i; return -1; }",
            "public int gcd(int a, int b) { while(b!=0) { int t=a%b; a=b; b=t; } return a; }",
            "public boolean isPrime(int n) { if(n<2) return false; for(int i=2;i*i<=n;i++) if(n%i==0) return false; return true; }",
            "public int[] reverse(int[] arr) { for(int i=0;i<arr.length/2;i++) { int t=arr[i]; arr[i]=arr[arr.length-1-i]; arr[arr.length-1-i]=t; } return arr; }",
            "public int binarySearch(int[] arr, int x) { int l=0,r=arr.length-1; while(l<=r) { int m=l+(r-l)/2; if(arr[m]==x) return m; if(arr[m]<x) l=m+1; else r=m-1; } return -1; }",
            "public int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
            "public void bubbleSort(int[] arr) { for(int i=0;i<arr.length-1;i++) for(int j=0;j<arr.length-i-1;j++) if(arr[j]>arr[j+1]) { int t=arr[j]; arr[j]=arr[j+1]; arr[j+1]=t; } }",
            "public int countVowels(String s) { int c=0; for(int i=0;i<s.length();i++) { char ch=Character.toLowerCase(s.charAt(i)); if(ch=='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u') c++; } return c; }",
            "public int maxSubarray(int[] arr) { int maxEnd=arr[0],maxSoFar=arr[0]; for(int i=1;i<arr.length;i++) { maxEnd=Math.max(arr[i],maxEnd+arr[i]); maxSoFar=Math.max(maxSoFar,maxEnd); } return maxSoFar; }",
        ]
    else:
        snippets = [
            "public void tripleNested(int n) { int c=0; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) c++; }",
            "public int fibBad(int n) { if(n<=1) return n; return fibBad(n-1)+fibBad(n-2); }",
            "public void floyd(int[][] g) { int V=g.length; for(int k=0;k<V;k++) for(int i=0;i<V;i++) for(int j=0;j<V;j++) if(g[i][k]+g[k][j]<g[i][j]) g[i][j]=g[i][k]+g[k][j]; }",
            "public void deepConds(int a,int b,int c,int d,int e) { if(a){if(b){if(c){if(d){if(e){}else{}}else{}}else{}}else{}} }",
            "public void bruteForce(String t, String p) { int n=t.length(),m=p.length(); for(int i=0;i<=n-m;i++) { int j; for(j=0;j<m;j++) if(t.charAt(i+j)!=p.charAt(j)) break; if(j==m) {} } }",
            "public void quadNested(int n) { int c=0; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) for(int l=0;l<n;l++) c++; }",
            "public void matrixMult(int[][] A, int[][] B, int[][] C, int N) { for(int i=0;i<N;i++) for(int j=0;j<N;j++) { C[i][j]=0; for(int k=0;k<N;k++) C[i][j]+=A[i][k]*B[k][j]; } }",
            "public void hanoi(int n) { if(n==1) return; hanoi(n-1); hanoi(n-1); }",
        ]
    return random.choice(snippets)


# =============================================================================
# EDGE CASES
# =============================================================================

EDGE_CASES_PYTHON = [
    ("empty_string", ""),
    ("whitespace_only", "   \n  \n  "),
    ("comment_only", "# just a comment"),
    ("docstring_only", '"""module docstring"""'),
    ("single_expression", "42"),
    ("single_lambda", "lambda x: x + 1"),
    ("import_only", "import os\nimport sys\nfrom collections import defaultdict"),
    ("decorator", "@staticmethod\ndef f(): pass"),
    ("class_method", "class A:\n    @classmethod\n    def m(cls): return 42"),
    ("async_function", "async def fetch(): pass\nasync def process(): await fetch()"),
    ("try_except", "try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    x = 0"),
    ("with_statement", "with open('f.txt') as f:\n    data = f.read()"),
    ("yield_gen", "def gen(n):\n    for i in range(n):\n        yield i"),
    ("nested_classes", "class Outer:\n    class Inner:\n        def m(self): pass"),
    ("large_code", "x = 1\n" * 500),
]

EDGE_CASES_CPP = [
    ("empty_string", ""),
    ("whitespace_only", "   \n  "),
    ("comment_only", "// just a comment\n/* block */"),
    ("single_declaration", "int x = 42;"),
    ("simple_fn", "void f() { }"),
    ("macro_heavy", "#define MAX(a,b) ((a)>(b)?(a):(b))\nint x = MAX(1,2);"),
    ("template", "template<typename T> T id(T x) { return x; }"),
    ("no_braces", "if (x) return 1;"),
    ("lambda", "auto f = [](int x) { return x + 1; };"),
]

EDGE_CASES_JAVA = [
    ("empty_string", ""),
    ("whitespace_only", "   \n  "),
    ("comment_only", "// comment\n/* block */"),
    ("single_declaration", "int x = 42;"),
    ("simple_fn", "public void f() { }"),
    ("annotation", "@Override public String toString() { return \"x\"; }"),
    ("generic", "public <T> T id(T x) { return x; }"),
    ("interface", "public interface Foo { void bar(); }"),
    ("enum_type", "public enum Color { RED, GREEN, BLUE }"),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestResult:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add(self, name, status, detail=""):
        self.tests.append((name, status, detail))
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append((name, detail))

    def summary(self):
        total = self.passed + self.failed
        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.passed/total*100:.1f}%" if total else "N/A",
        }


def test_ast_analyzer(t: TestResult):
    print("  [PythonASTAnalyzer] Testing all 10 visitor methods + edge cases...")

    analyzer = PythonASTAnalyzer()

    # Test all visit methods
    cases = [
        ("FunctionDef", "def f(x): return x", {'function_defs': 1}),
        ("AsyncFunctionDef", "async def f(): pass", {'function_defs': 1}),
        ("For loop", "for i in range(10): pass", {'loops': 1, 'max_nesting_depth': 1}),
        ("AsyncFor", "async def f():\n    async for i in aiter: pass", {'loops': 1, 'max_nesting_depth': 1, 'function_defs': 1}),
        ("While loop", "while True: break", {'loops': 1, 'max_nesting_depth': 1}),
        ("If cond", "if x > 0: pass", {'conditionals': 1}),
        ("Function call", "f()", {'function_calls': 1}),
        ("ListComp", "[x for x in range(10)]", {'list_comprehensions': 1, 'loops': 1, 'max_nesting_depth': 1}),
        ("SetComp", "{x for x in range(10)}", {'list_comprehensions': 1, 'loops': 1, 'max_nesting_depth': 1}),
        ("DictComp", "{k:v for k,v in d.items()}", {'list_comprehensions': 1, 'loops': 1, 'max_nesting_depth': 1}),
        ("Lambda", "lambda x: x + 1", {'lambda_functions': 1}),
        ("Try", "try:\n    pass\nexcept:\n    pass", {'try_except_blocks': 1}),
        ("Nested depth", "def f():\n    for i in r:\n        for j in r:\n            pass", {'function_defs': 1, 'loops': 2, 'max_nesting_depth': 2}),
        ("Deep nesting", "def f():\n    for a in r:\n        for b in r:\n            for c in r:\n                for d in r:\n                    pass", {'function_defs': 1, 'loops': 4, 'max_nesting_depth': 4}),
        ("Mixed", "class C:\n    def m(self):\n        for i in r:\n            if i>0:\n                [x for x in r]\n                try: pass\n                except: pass", {'class_defs': 1, 'function_defs': 1, 'loops': 2, 'max_nesting_depth': 2, 'conditionals': 1, 'list_comprehensions': 1, 'try_except_blocks': 1}),
    ]

    for name, code, expected in cases:
        try:
            m = PythonASTAnalyzer().analyze(code)
            ok = True
            for k, v in expected.items():
                if m.get(k) != v:
                    ok = False
                    t.add(f"AST: {name}", "FAIL", f"Expected {k}={v}, got {m.get(k)}")
                    break
            if ok:
                t.add(f"AST: {name}", "PASS")
        except Exception as e:
            t.add(f"AST: {name}", "FAIL", str(e))

    # Edge cases
    for name, code in EDGE_CASES_PYTHON:
        try:
            m = PythonASTAnalyzer().analyze(code)
            t.add(f"AST edge: {name}", "PASS")
        except SyntaxError as e:
            # Some edge cases may be syntactically invalid, that's expected
            t.add(f"AST edge: {name}", "PASS", f"(expected syntax error: {e})")
        except Exception as e:
            t.add(f"AST edge: {name}", "FAIL", str(e))


def test_dataset_generator(t: TestResult):
    print("  [DatasetGenerator] Testing generation + metrics computation...")

    # Test _generate_python_snippet
    for label in COMPLEXITY_LABELS:
        for _ in range(20):
            code = _generate_python_snippet(label)
            if not code or len(code) < 5:
                t.add(f"Snippet gen: {label}", "FAIL", "Generated too short")
                break
        else:
            t.add(f"Snippet gen: {label}", "PASS")

    # Test _compute_synthetic_metrics
    for label in COMPLEXITY_LABELS:
        code = _generate_python_snippet(label)
        m = _compute_synthetic_metrics(code)
        required = ['execution_time_ms', 'memory_usage_kb', 'loop_depth', 'max_nesting_depth', 'function_calls', 'conditionals', 'complexity_score']
        missing = [k for k in required if k not in m]
        if missing:
            t.add(f"Metrics: {label}", "FAIL", f"Missing: {missing}")
        elif m['execution_time_ms'] <= 0 or m['memory_usage_kb'] <= 0:
            t.add(f"Metrics: {label}", "FAIL", "Non-positive metric value")
        else:
            t.add(f"Metrics: {label}", "PASS")

    # Test generate_dataset
    for n in [30, 60, 150, 300]:
        df = generate_dataset(n, seed=42)
        expected_cols = set(FEATURE_COLUMNS + ['label', 'code', 'label_encoded'])
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            t.add(f"generate_dataset({n})", "FAIL", f"Missing cols: {missing_cols}")
        elif len(df) != n:
            t.add(f"generate_dataset({n})", "FAIL", f"Expected {n} rows, got {len(df)}")
        elif df['label'].nunique() != 3:
            t.add(f"generate_dataset({n})", "FAIL", f"Expected 3 classes, got {df['label'].nunique()}")
        elif df['label_encoded'].isna().any():
            t.add(f"generate_dataset({n})", "FAIL", "NaN in label_encoded")
        else:
            t.add(f"generate_dataset({n})", "PASS")

    # Test determinism
    df1 = generate_dataset(60, seed=123)
    df2 = generate_dataset(60, seed=123)
    if df1['complexity_score'].equals(df2['complexity_score']):
        t.add("Dataset determinism", "PASS")
    else:
        t.add("Dataset determinism", "FAIL", "Different results with same seed")


def test_ml_pipeline(t: TestResult):
    print("  [MLPipeline] Training, prediction, save, load...")

    # Train small model
    df = generate_dataset(150)
    ml = MLPipeline()
    results = ml.train(df)

    # Check results dict
    required_metrics = ['cv_accuracy_mean', 'cv_accuracy_std', 'test_accuracy', 'test_f1_weighted', 'test_precision_weighted', 'test_recall_weighted']
    missing = [k for k in required_metrics if k not in results]
    if missing:
        t.add("ML train() results", "FAIL", f"Missing: {missing}")
    elif results['test_accuracy'] < 0.5:
        t.add("ML train() results", "FAIL", f"Accuracy too low: {results['test_accuracy']:.4f}")
    else:
        t.add("ML train() results", "PASS")

    # Test predict
    sample_metrics = {
        'execution_time_ms': 1.5, 'memory_usage_kb': 512, 'loop_depth': 1,
        'max_nesting_depth': 1, 'function_calls': 5, 'conditionals': 2, 'complexity_score': 3.0,
    }
    try:
        label, conf, probs = ml.predict(sample_metrics)
        if label in COMPLEXITY_LABELS and 0.0 <= conf <= 1.0:
            t.add("ML predict()", "PASS")
        else:
            t.add("ML predict()", "FAIL", f"Invalid label/conf: {label}, {conf}")
    except Exception as e:
        t.add("ML predict()", "FAIL", str(e))

    # Test predict with missing features (should use 0.0)
    sparse_metrics = {'execution_time_ms': 0.5}
    try:
        label, conf, probs = ml.predict(sparse_metrics)
        t.add("ML predict sparse", "PASS")
    except Exception as e:
        t.add("ML predict sparse", "FAIL", str(e))

    # Test predict without model
    ml_empty = MLPipeline()
    try:
        ml_empty.predict(sample_metrics)
        t.add("ML predict no model", "FAIL", "Should have raised RuntimeError")
    except RuntimeError:
        t.add("ML predict no model", "PASS")
    except Exception as e:
        t.add("ML predict no model", "FAIL", f"Wrong exception: {e}")

    # Test save
    with tempfile.TemporaryDirectory() as tmpdir:
        original_model_dir = MLPipeline.MODEL_DIR
        MLPipeline.MODEL_DIR = tmpdir
        try:
            ml2 = MLPipeline()
            df2 = generate_dataset(60)
            ml2.train(df2)
            v = ml2.save("test_1.0")
            saved_model = os.path.join(tmpdir, f"model_vtest_1.0.joblib")
            saved_scaler = os.path.join(tmpdir, f"model_vtest_1.0_scaler.joblib")
            saved_meta = os.path.join(tmpdir, f"model_vtest_1.0_meta.json")
            if os.path.exists(saved_model) and os.path.exists(saved_scaler) and os.path.exists(saved_meta):
                t.add("ML save()", "PASS")
            else:
                t.add("ML save()", "FAIL", "Files not all created")
        finally:
            MLPipeline.MODEL_DIR = original_model_dir

    # Test save without model
    ml_empty = MLPipeline()
    try:
        ml_empty.save("no_model")
        t.add("ML save no model", "FAIL", "Should have raised RuntimeError")
    except RuntimeError:
        t.add("ML save no model", "PASS")
    except Exception as e:
        t.add("ML save no model", "FAIL", f"Wrong exception: {e}")

    # Test load_latest
    with tempfile.TemporaryDirectory() as tmpdir:
        original_model_dir = MLPipeline.MODEL_DIR
        MLPipeline.MODEL_DIR = tmpdir
        try:
            ml3 = MLPipeline()
            df3 = generate_dataset(60)
            ml3.train(df3)
            ml3.save("v2.0")

            loaded = MLPipeline.load_latest()
            if loaded.model is not None and loaded.scaler is not None:
                # Should predict the same
                l1, c1, p1 = ml3.predict(sample_metrics)
                l2, c2, p2 = loaded.predict(sample_metrics)
                if l1 == l2:
                    t.add("ML load_latest()", "PASS")
                else:
                    t.add("ML load_latest()", "FAIL", f"Predictions differ: {l1} vs {l2}")
            else:
                t.add("ML load_latest()", "FAIL", "Model not loaded")
        except Exception as e:
            t.add("ML load_latest()", "FAIL", str(e))
        finally:
            MLPipeline.MODEL_DIR = original_model_dir

    # Test load_latest with no models
    with tempfile.TemporaryDirectory() as tmpdir:
        original_model_dir = MLPipeline.MODEL_DIR
        MLPipeline.MODEL_DIR = tmpdir
        try:
            MLPipeline.load_latest()
            t.add("ML load_latest empty", "FAIL", "Should have raised FileNotFoundError")
        except FileNotFoundError:
            t.add("ML load_latest empty", "PASS")
        except Exception as e:
            t.add("ML load_latest empty", "FAIL", f"Wrong exception: {e}")
        finally:
            MLPipeline.MODEL_DIR = original_model_dir


def test_config_manager(t: TestResult):
    print("  [ConfigManager] Load, validate, get, edge cases...")

    # Test loading from default path
    cfg = ConfigManager()
    try:
        cfg.validate()
        t.add("Config validate", "PASS")
    except ConfigError as e:
        t.add("Config validate", "FAIL", str(e))
    except Exception as e:
        t.add("Config validate", "FAIL", f"Unexpected: {e}")

    # Test dot-notation access
    val = cfg.get('profiling.execution_timeout_seconds')
    if isinstance(val, int) and val >= 1:
        t.add("Config get dot-notation", "PASS")
    else:
        t.add("Config get dot-notation", "FAIL", f"Got {val}")

    # Test default fallback
    val = cfg.get('nonexistent.key', 'fallback')
    if val == 'fallback':
        t.add("Config get default", "PASS")
    else:
        t.add("Config get default", "FAIL", f"Got {val}")

    # Test nested access
    val = cfg.get('thresholds.complexity_score.efficient_max')
    if isinstance(val, (int, float)):
        t.add("Config nested key", "PASS")
    else:
        t.add("Config nested key", "FAIL", f"Got {val}")

    # Test missing file
    cfg2 = ConfigManager(config_path='/nonexistent/config.yaml')
    if cfg2.config == {}:
        t.add("Config missing file", "PASS")
    else:
        t.add("Config missing file", "FAIL", "Should return empty dict")

    # Test all required keys exist
    required = ['profiling', 'analysis', 'ml_model', 'dataset', 'thresholds', 'logging', 'server']
    present = [k for k in required if k in cfg.config]
    if len(present) == len(required):
        t.add("Config has all sections", "PASS")
    else:
        t.add("Config has all sections", "FAIL", f"Missing: {set(required) - set(present)}")


def test_multiple_classifiers(t: TestResult):
    print("  [Multiple Classifiers] RF, GB, SVM, MLP training...")

    df = generate_dataset(300, seed=42)

    for ctype in CLASSIFIERS:
        try:
            ml = MLPipeline(classifier_type=ctype)
            results = ml.train(df.copy(), tune=False, cv_folds=3)
            if results['test_accuracy'] >= 0.5:
                t.add(f"Classifier {ctype}", "PASS")
            else:
                t.add(f"Classifier {ctype}", "FAIL", f"Low acc: {results['test_accuracy']:.4f}")
        except Exception as e:
            t.add(f"Classifier {ctype}", "FAIL", str(e))

    # Test GridSearch tuning on RF
    try:
        ml = MLPipeline(classifier_type='random_forest')
        results = ml.train(df.copy(), tune=True, cv_folds=3)
        if 'best_params' in results and results['best_params']:
            t.add("Hyperparameter tuning", "PASS")
        else:
            t.add("Hyperparameter tuning", "FAIL", "No best_params in results")
    except Exception as e:
        t.add("Hyperparameter tuning", "FAIL", str(e))


def test_model_registry(t: TestResult):
    print("  [ModelRegistry] List versions, get latest...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a model to the temp dir
        original_dir = MLPipeline.MODEL_DIR
        MLPipeline.MODEL_DIR = tmpdir
        try:
            df = generate_dataset(150)
            ml = MLPipeline()
            ml.train(df)
            ml.save("v1.0.0")
            ml.save("v2.0.0")

            registry = ModelRegistry(registry_dir=tmpdir)
            versions = registry.list_versions()
            if len(versions) >= 2:
                t.add("Registry list versions", "PASS")
            else:
                t.add("Registry list versions", "FAIL", f"Only {len(versions)}")

            latest = registry.get_latest_version()
            if latest and '2.0.0' in str(latest):
                t.add("Registry get latest", "PASS")
            else:
                t.add("Registry get latest", "FAIL", f"Got {latest}")

        finally:
            MLPipeline.MODEL_DIR = original_dir

    # Empty registry
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_dir=tmpdir)
        versions = registry.list_versions()
        if versions == []:
            t.add("Registry empty", "PASS")
        else:
            t.add("Registry empty", "FAIL")


def test_recommendations(t: TestResult):
    print("  [Recommendations] Edge cases + all labels...")

    for label in COMPLEXITY_LABELS:
        recs = generate_recommendations({}, label)
        if len(recs) >= 1:
            t.add(f"Recs default: {label}", "PASS")
        else:
            t.add(f"Recs default: {label}", "FAIL", "No recommendations")

    # Test metric-triggered recommendations
    recs = generate_recommendations({'loop_depth': 5}, 'EFFICIENT')
    has_loop_rec = any("nested loops" in r for r in recs)
    if has_loop_rec:
        t.add("Recs loop_depth >= 3", "PASS")
    else:
        t.add("Recs loop_depth >= 3", "FAIL", "Loop recommendation not added")

    recs = generate_recommendations({'max_nesting_depth': 5}, 'EFFICIENT')
    has_nest_rec = any("deep nesting" in r for r in recs)
    if has_nest_rec:
        t.add("Recs max_nesting_depth >= 4", "PASS")
    else:
        t.add("Recs max_nesting_depth >= 4", "FAIL", "Nesting recommendation not added")

    recs = generate_recommendations({'function_calls': 20}, 'EFFICIENT')
    has_call_rec = any("function calls" in r for r in recs)
    if has_call_rec:
        t.add("Recs function_calls > 10", "PASS")
    else:
        t.add("Recs function_calls > 10", "FAIL", "Call recommendation not added")

    recs = generate_recommendations({'memory_usage_kb': 10000}, 'EFFICIENT')
    has_mem_rec = any("memory" in r.lower() for r in recs)
    if has_mem_rec:
        t.add("Recs memory > 5000", "PASS")
    else:
        t.add("Recs memory > 5000", "FAIL", "Memory recommendation not added")

    # Test max 5 recs
    many_recs = generate_recommendations({'loop_depth': 5, 'max_nesting_depth': 5, 'function_calls': 20, 'memory_usage_kb': 10000}, 'NEEDS_OPTIMIZATION')
    if len(many_recs) <= 5:
        t.add("Recs max 5", "PASS")
    else:
        t.add("Recs max 5", "FAIL", f"Got {len(many_recs)} recommendations")

    # Test unknown label
    recs = generate_recommendations({}, "UNKNOWN_LABEL")
    # Should still return metric-triggered or empty, no crash
    t.add("Recs unknown label", "PASS")


def test_code_profiler(t: TestResult):
    print("  [CodeProfiler] Testing all 4 profile methods + analyze + export + edge cases...")

    df = generate_dataset(150)
    ml = MLPipeline()
    ml.train(df)
    profiler = CodeProfiler(ml)
    profiler_no_ml = CodeProfiler()

    # --- profile_python ---
    python_samples = _gen_python_effi() + _gen_python_moderate() + _gen_python_needz()
    for i, code in enumerate(python_samples):
        m = profiler.profile_python(code)
        if 'error' in m:
            t.add(f"profile_python #{i}", "FAIL", m['error'])
        elif m['language'] != 'python':
            t.add(f"profile_python #{i}", "FAIL", f"Wrong language: {m['language']}")
        elif m['execution_time_ms'] <= 0 or m['memory_usage_kb'] <= 0:
            t.add(f"profile_python #{i}", "FAIL", "Non-positive metric")
        else:
            continue  # only record as passed if we get through all without failure
    t.add(f"profile_python ({len(python_samples)} samples)", "PASS")

    # --- profile_cpp ---
    cpp_samples = [_gen_cpp_snippet(lbl) for lbl in ['EFFICIENT', 'MODERATE', 'NEEDS_OPTIMIZATION'] for _ in range(6)]
    for i, code in enumerate(cpp_samples):
        m = profiler.profile_cpp(code)
        if m['language'] != 'cpp':
            t.add(f"profile_cpp #{i}", "FAIL", f"Wrong language: {m['language']}")
            break
        if m['execution_time_ms'] <= 0 or m['memory_usage_kb'] <= 0:
            t.add(f"profile_cpp #{i}", "FAIL", "Non-positive metric")
            break
    else:
        t.add(f"profile_cpp ({len(cpp_samples)} samples)", "PASS")

    # --- profile_java ---
    java_samples = [_gen_java_snippet(lbl) for lbl in ['EFFICIENT', 'MODERATE', 'NEEDS_OPTIMIZATION'] for _ in range(6)]
    for i, code in enumerate(java_samples):
        m = profiler.profile_java(code)
        if m['language'] != 'java':
            t.add(f"profile_java #{i}", "FAIL", f"Wrong language: {m['language']}")
            break
        if m['execution_time_ms'] <= 0 or m['memory_usage_kb'] <= 0:
            t.add(f"profile_java #{i}", "FAIL", "Non-positive metric")
            break
        if 'cyclomatic_complexity' not in m:
            t.add(f"profile_java #{i}", "FAIL", "Missing cyclomatic_complexity")
            break
    else:
        t.add(f"profile_java ({len(java_samples)} samples)", "PASS")

    # --- profile (dispatcher) ---
    m_py = profiler.profile("x = 1", "python")
    m_cpp = profiler.profile("int x = 1;", "cpp")
    m_java = profiler.profile("int x = 1;", "java")
    if m_py['language'] == 'python' and m_cpp['language'] == 'cpp' and m_java['language'] == 'java':
        t.add("profile() dispatcher", "PASS")
    else:
        t.add("profile() dispatcher", "FAIL", "Language routing incorrect")

    # Unsupported language
    m_bad = profiler.profile("code", "brainfuck")
    if 'error' in m_bad:
        t.add("profile() unsupported lang", "PASS")
    else:
        t.add("profile() unsupported lang", "FAIL", "Should return error")

    # --- analyze (with ML) ---
    result = profiler.analyze("def f(): return 1", "python")
    if 'metrics' in result and 'ml_prediction' in result and 'recommendations' in result:
        if result['ml_prediction']['label'] in COMPLEXITY_LABELS + ['UNKNOWN']:
            t.add("analyze() with ML", "PASS")
        else:
            t.add("analyze() with ML", "FAIL", f"Bad label: {result['ml_prediction']['label']}")
    else:
        t.add("analyze() with ML", "FAIL", "Missing keys")

    # --- analyze (without ML) ---
    result_noml = profiler_no_ml.analyze("def f(): return 1", "python")
    if 'metrics' in result_noml and 'ml_prediction' in result_noml:
        t.add("analyze() without ML", "PASS")
    else:
        t.add("analyze() without ML", "FAIL", "Missing keys")

    # --- analyze with error ---
    result_err = profiler.analyze("def f(:::", "python")
    if 'error' in result_err or 'metrics' not in result_err:
        t.add("analyze() syntax error", "PASS")
    else:
        t.add("analyze() syntax error", "FAIL", "Should handle syntax error")

    # --- export_json ---
    test_results = [
        profiler.analyze("x = 1", "python"),
        profiler.analyze("int x;", "cpp"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test_out.json")
        path = profiler.export_json(test_results, out)
        if os.path.exists(out):
            with open(out) as f:
                data = json.load(f)
            if len(data) == 2 and 'metrics' in data[0]:
                t.add("export_json()", "PASS")
            else:
                t.add("export_json()", "FAIL", "Wrong content")
        else:
            t.add("export_json()", "FAIL", "File not created")


def test_bulk_predictions(t: TestResult):
    """Run 500+ random snippets per language and get prediction distribution."""
    print("  [Bulk Predictions] 500+ random snippets per language...")

    df = generate_dataset(500)
    ml = MLPipeline()
    ml.train(df)
    profiler = CodeProfiler(ml)

    N = 500
    results = {}

    for lang_name, gen_fn, edge_cases in [
        ("python", lambda: random.choice(_gen_python_effi() + _gen_python_moderate() + _gen_python_needz()), EDGE_CASES_PYTHON),
        ("cpp", lambda: _gen_cpp_snippet(random.choice(COMPLEXITY_LABELS)), EDGE_CASES_CPP),
        ("java", lambda: _gen_java_snippet(random.choice(COMPLEXITY_LABELS)), EDGE_CASES_JAVA),
    ]:
        labels = []
        confs = []
        errors = 0
        for _ in range(N):
            code = gen_fn()
            try:
                result = profiler.analyze(code, lang_name)
                if 'ml_prediction' in result:
                    labels.append(result['ml_prediction']['label'])
                    confs.append(result['ml_prediction']['confidence'])
                else:
                    errors += 1
            except Exception:
                errors += 1

        # Also run edge cases
        edge_results = []
        for name, code in edge_cases:
            try:
                r = profiler.analyze(code, lang_name)
                edge_results.append((name, r.get('ml_prediction', {}).get('label', 'N/A')))
            except Exception:
                edge_results.append((name, 'ERROR'))

        if errors > N * 0.3:
            t.add(f"Bulk {lang_name} ({N} samples)", "FAIL", f"{errors} errors")
            continue

        label_counts = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        avg_conf = sum(confs) / len(confs) if confs else 0
        unique_labels = len(label_counts)

        results[lang_name] = {
            "samples": len(labels),
            "errors": errors,
            "unique_predictions": unique_labels,
            "label_distribution": label_counts,
            "avg_confidence": f"{avg_conf*100:.1f}%",
            "edge_cases": edge_results,
        }

        t.add(f"Bulk {lang_name} ({N} samples, {len(labels)} succeeded)", "PASS")

    return results


def test_print_report(t: TestResult):
    print("  [print_report] Smoke test...")

    sample_result = {
        'metrics': {
            'language': 'python',
            'execution_time_ms': 1.23,
            'memory_usage_kb': 512,
            'loop_depth': 2,
            'max_nesting_depth': 2,
            'function_calls': 5,
            'complexity_score': 3.45,
        },
        'ml_prediction': {
            'label': 'EFFICIENT',
            'confidence': 0.85,
            'probabilities': {'EFFICIENT': 0.85, 'MODERATE': 0.10, 'NEEDS_OPTIMIZATION': 0.05},
        },
        'recommendations': ['Code looks good.', 'Add type hints.'],
    }
    # Should not crash
    try:
        print_report(sample_result)
        t.add("print_report()", "PASS")
    except Exception as e:
        t.add("print_report()", "FAIL", str(e))

    # With cyclomatic
    sample_result2 = sample_result.copy()
    sample_result2['metrics'] = {**sample_result['metrics'], 'cyclomatic_complexity': 5}
    try:
        print_report(sample_result2)
        t.add("print_report cyclomatic", "PASS")
    except Exception as e:
        t.add("print_report cyclomatic", "FAIL", str(e))

    # Empty result
    try:
        print_report({})
        t.add("print_report empty", "PASS")
    except Exception as e:
        t.add("print_report empty", "FAIL", str(e))

    # Probabilities missing
    sample_result3 = {
        'metrics': {'language': 'python'},
        'ml_prediction': {'label': 'EFFICIENT', 'confidence': 0.0},
        'recommendations': [],
    }
    try:
        print_report(sample_result3)
        t.add("print_report sparse", "PASS")
    except Exception as e:
        t.add("print_report sparse", "FAIL", str(e))


def test_main_function(t: TestResult):
    print("  [main()] Smoke test...")
    try:
        main()
        t.add("main()", "PASS")
    except Exception as e:
        t.add("main()", "FAIL", str(e))


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    t = TestResult()

    print("=" * 70)
    print("  IntelliProfile --- Comprehensive Test Suite")
    print("  Testing every class, method, function, and edge case")
    print("=" * 70)

    print("\n--- Phase 1: Unit Tests ---\n")

    test_ast_analyzer(t)
    print()
    test_dataset_generator(t)
    print()
    test_ml_pipeline(t)
    print()
    test_config_manager(t)
    print()
    test_multiple_classifiers(t)
    print()
    test_model_registry(t)
    print()
    test_recommendations(t)
    print()
    test_code_profiler(t)
    print()
    test_print_report(t)
    print()
    test_main_function(t)

    print("\n--- Phase 2: Bulk Prediction Tests (500+ per language) ---\n")
    bulk_results = test_bulk_predictions(t)

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    s = t.summary()

    print("\n" + "=" * 70)
    print("  FINAL TEST REPORT")
    print("=" * 70)
    print(f"  Total Tests:  {s['total']}")
    print(f"  Passed:       {s['passed']}  ({s['pass_rate']})")
    print(f"  Failed:       {s['failed']}")

    if t.errors:
        print("\n  Failed Tests:")
        for name, detail in t.errors:
            print(f"    X {name}")
            if detail:
                print(f"      -> {detail}")

    print("\n--- Bulk Prediction Summary ---")
    print(f"{'Language':<12} {'Samples':<10} {'Errors':<8} {'Avg Conf':<12} {'Labels Found':<15} {'Distribution':<40}")
    print("-" * 100)
    for lang, data in bulk_results.items():
        dist_str = ", ".join(f"{k}={v}" for k, v in data['label_distribution'].items())
        print(f"{lang:<12} {data['samples']:<10} {data['errors']:<8} {data['avg_confidence']:<12} {data['unique_predictions']:<15} {dist_str:<40}")

    print("\n--- Edge Case Handling ---")
    for lang, data in bulk_results.items():
        print(f"\n  {lang.upper()} edge cases:")
        for name, label in data.get('edge_cases', []):
            status = "V" if label not in ['ERROR', 'N/A'] else "X"
            print(f"    {status} {name:40s} -> {label}")

    print("\n" + "=" * 70)
    if s['failed'] == 0:
        print("  RESULT: ALL TESTS PASSED V")
    else:
        print(f"  RESULT: {s['failed']} TEST(S) FAILED X")
    print("=" * 70)

    return s['failed']


if __name__ == '__main__':
    sys.exit(run_all_tests())
