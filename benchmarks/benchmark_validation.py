"""
IntelliX Benchmark Validation Suite
=====================================
Validates that the EFFICIENT / MODERATE / NEEDS_OPTIMIZATION complexity
classes used throughout IntelliX correspond to real, measured runtime
behavior -- not just to which synthetic template bucket a snippet was
written into.

For each language (Python, C++, Java), this runs a small set of
deliberately designed, parameterized benchmark functions at increasing
input sizes, measures REAL execution time using the same kind of
instrumentation the legacy cpp_profiler.txt (std::chrono) and
java_profiler.java (now System.nanoTime) were built around -- but
actually exercises it, on purpose-built code, with a single timed call
per process rather than the JVM/heap-delta approach those files used.
It fits an empirical growth exponent to the measured times and
classifies the *observed* complexity, then compares that against the
*intended* label the benchmark was written to represent.

SCOPE NOTE: This validates the labeling CONCEPT using curated,
hand-written benchmark code that ships with the project. It does not
change how the live API/MCP tool analyzes arbitrary user-submitted code
-- that remains static, parse-based analysis (real tree-sitter parsing
for C++/Java, AST for Python) with nothing compiled or executed,
deliberately, since compiling and running untrusted submitted code is a
real security/sandboxing problem this suite does not attempt to solve
(see IntelliX_Council_Verdict.md). This is a validation tool, not a
change to the training or inference pipeline.

Methodology notes:
  - Each (benchmark, n) pair is measured REPS times; the median is used
    to reduce noise from process/OS scheduling jitter.
  - Java benchmarks run with `-Xint` (interpreter only, JIT disabled).
    Without it, HotSpot's on-stack-replacement compiler kicks in mid-loop
    for the larger inputs and makes them disproportionately fast,
    distorting the growth curve. Forcing pure interpretation removes
    that confound; the goal here is "does relative growth match the
    complexity class", not "how fast is JIT-optimized Java".
  - The quadratic_pairs benchmarks use a volatile, non-foldable
    accumulator (`c += (i ^ j) & 1`) because a naive `c++` nested loop
    has a closed-form value (n^2) that an optimizing C++ compiler will
    fold away at compile time, making the "quadratic" benchmark
    silently run in constant time.
  - Growth exponent is estimated as the average of pairwise
    log(T2/T1) / log(n2/n1) across consecutive (n, time) measurements.
    True exponential growth (naive_fib) produces a very large value
    under this formula (since exponential time vastly outpaces any
    fixed power of n), which is why it's classified correctly without
    needing separate exponential-detection logic.

Usage:
    python benchmark_validation.py
"""

import json
import math
import os
import statistics
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(BENCH_DIR, "cpp")
JAVA_DIR = os.path.join(BENCH_DIR, "java")
REPS = 3

# ---------------------------------------------------------------------------
# Python benchmark functions (timed in-process -- no subprocess overhead
# to exclude, perf_counter wraps exactly the function call).
# ---------------------------------------------------------------------------

def _py_mod_check(n):
    return n % 97 == 0

def _py_sqrt_calc(n):
    return math.isqrt(n)

def _py_linear_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

def _py_sort_n(n):
    arr = list(range(n, 0, -1))
    arr.sort()
    return arr[0]

def _py_quadratic_pairs(n):
    c = 0
    for i in range(n):
        for j in range(n):
            c += 1
    return c

def _py_naive_fib(n):
    if n <= 1:
        return n
    return _py_naive_fib(n - 1) + _py_naive_fib(n - 2)


def _time_python(fn, n):
    t0 = time.perf_counter()
    fn(n)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _time_subprocess(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    for token in out.split():
        if token.startswith("ELAPSED_MS="):
            return float(token.split("=", 1)[1])
    raise RuntimeError(f"Could not parse ELAPSED_MS from output: {out!r}")


def _cpp_runner(binary):
    path = os.path.join(CPP_DIR, binary)
    return lambda n: _time_subprocess([path, str(n)])


def _java_runner(classname):
    return lambda n: _time_subprocess(["java", "-Xint", "-cp", JAVA_DIR, classname, str(n)])


# ---------------------------------------------------------------------------
# Benchmark registry: (name, intended_label, n_values, run_fn)
# run_fn(n) -> elapsed_ms for a single call (caller handles repetition)
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "python": [
        ("mod_check",       "EFFICIENT",          [10_000, 100_000, 1_000_000],          lambda n: _time_python(_py_mod_check, n)),
        ("sqrt_calc",       "EFFICIENT",          [10_000, 100_000, 1_000_000],          lambda n: _time_python(_py_sqrt_calc, n)),
        ("linear_sum",      "MODERATE",           [10_000, 50_000, 200_000],             lambda n: _time_python(_py_linear_sum, n)),
        ("sort_n",          "MODERATE",           [10_000, 50_000, 200_000],             lambda n: _time_python(_py_sort_n, n)),
        ("quadratic_pairs", "NEEDS_OPTIMIZATION", [500, 1_000, 2_000],                   lambda n: _time_python(_py_quadratic_pairs, n)),
        ("naive_fib",       "NEEDS_OPTIMIZATION", [20, 25, 28],                          lambda n: _time_python(_py_naive_fib, n)),
    ],
    "cpp": [
        ("mod_check",       "EFFICIENT",          [1_000_000, 10_000_000, 100_000_000], _cpp_runner("mod_check")),
        ("sqrt_calc",       "EFFICIENT",          [1_000_000, 10_000_000, 100_000_000], _cpp_runner("sqrt_calc")),
        ("linear_sum",      "MODERATE",           [10_000_000, 50_000_000, 200_000_000], _cpp_runner("linear_sum")),
        ("sort_n",          "MODERATE",           [1_000_000, 5_000_000, 20_000_000],   _cpp_runner("sort_n")),
        ("quadratic_pairs", "NEEDS_OPTIMIZATION", [2_000, 4_000, 8_000],                _cpp_runner("quadratic_pairs")),
        ("naive_fib",       "NEEDS_OPTIMIZATION", [30, 35, 38],                          _cpp_runner("naive_fib")),
    ],
    "java": [
        ("mod_check",       "EFFICIENT",          [1_000_000, 10_000_000, 100_000_000], _java_runner("ModCheck")),
        ("sqrt_calc",       "EFFICIENT",          [1_000_000, 10_000_000, 100_000_000], _java_runner("SqrtCalc")),
        ("linear_sum",      "MODERATE",           [10_000_000, 50_000_000, 200_000_000], _java_runner("LinearSum")),
        ("sort_n",          "MODERATE",           [1_000_000, 5_000_000, 20_000_000],   _java_runner("SortN")),
        ("quadratic_pairs", "NEEDS_OPTIMIZATION", [2_000, 4_000, 8_000],                _java_runner("QuadraticPairs")),
        ("naive_fib",       "NEEDS_OPTIMIZATION", [30, 35, 38],                          _java_runner("NaiveFib")),
    ],
}


def classify_exponent(avg_exponent: float) -> str:
    """Map an empirical growth exponent to a real-world complexity label.
    Genuinely exponential growth (e.g. naive recursive fibonacci) produces
    a very large value here -- it isn't a clean power law at all, but the
    mismatch itself reliably pushes the value well past the
    NEEDS_OPTIMIZATION threshold, so no separate exponential-detection
    branch is needed."""
    if avg_exponent < 0.5:
        return "EFFICIENT"
    elif avg_exponent < 1.5:
        return "MODERATE"
    else:
        return "NEEDS_OPTIMIZATION"


def run_benchmark(n_values, run_fn):
    timings = []
    for n in n_values:
        samples = [run_fn(n) for _ in range(REPS)]
        timings.append(statistics.median(samples))
    exponents = []
    for i in range(len(n_values) - 1):
        n1, n2 = n_values[i], n_values[i + 1]
        t1, t2 = max(timings[i], 1e-6), max(timings[i + 1], 1e-6)
        exponents.append(math.log(t2 / t1) / math.log(n2 / n1))
    avg_exponent = sum(exponents) / len(exponents)
    return timings, exponents, avg_exponent


CPP_BENCH_NAMES = ["mod_check", "sqrt_calc", "linear_sum", "sort_n", "quadratic_pairs", "naive_fib"]
JAVA_BENCH_NAMES = ["ModCheck", "SqrtCalc", "LinearSum", "SortN", "QuadraticPairs", "NaiveFib"]


def _ensure_built():
    """Compile any C++/Java benchmark whose binary is missing or older than
    its source, so a fresh clone can just run this script directly. Returns
    (cpp_available, java_available)."""
    cpp_ok = True
    try:
        for name in CPP_BENCH_NAMES:
            src = os.path.join(CPP_DIR, f"{name}.cpp")
            binary = os.path.join(CPP_DIR, name)
            if not os.path.exists(binary) or os.path.getmtime(src) > os.path.getmtime(binary):
                subprocess.run(["g++", "-O2", "-std=c++17", "-o", binary, src], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  [skip] C++ benchmarks unavailable ({e}). Install g++ to include them.")
        cpp_ok = False

    java_ok = True
    try:
        for name in JAVA_BENCH_NAMES:
            src = os.path.join(JAVA_DIR, f"{name}.java")
            classfile = os.path.join(JAVA_DIR, f"{name}.class")
            if not os.path.exists(classfile) or os.path.getmtime(src) > os.path.getmtime(classfile):
                subprocess.run(["javac", src], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  [skip] Java benchmarks unavailable ({e}). Install a JDK (javac) to include them.")
        java_ok = False

    return cpp_ok, java_ok


def main():
    print("=" * 88)
    print("  IntelliX Benchmark Validation Suite")
    print("  Real measured timing vs. intended complexity-class labels")
    print("=" * 88)

    print("\nChecking/compiling benchmark binaries...")
    cpp_ok, java_ok = _ensure_built()

    active_benchmarks = dict(BENCHMARKS)
    if not cpp_ok:
        active_benchmarks.pop("cpp", None)
    if not java_ok:
        active_benchmarks.pop("java", None)

    all_results = []
    mismatches = 0

    for lang, benches in active_benchmarks.items():
        print(f"\n--- {lang.upper()} ---")
        for name, intended, n_values, run_fn in benches:
            timings, exponents, avg_exp = run_benchmark(n_values, run_fn)
            real_label = classify_exponent(avg_exp)
            match = "MATCH" if real_label == intended else "MISMATCH"
            if match == "MISMATCH":
                mismatches += 1
            timing_str = " -> ".join(f"{t:.4f}ms" for t in timings)
            print(f"  {name:18s} n={n_values}")
            print(f"    timings:  {timing_str}")
            print(f"    growth exponent (avg): {avg_exp:6.2f}   measured: {real_label:20s} intended: {intended:20s} [{match}]")
            all_results.append({
                "language": lang,
                "benchmark": name,
                "n_values": n_values,
                "timings_ms": timings,
                "pairwise_exponents": [round(e, 3) for e in exponents],
                "growth_exponent_avg": round(avg_exp, 3),
                "measured_label": real_label,
                "intended_label": intended,
                "result": match,
            })

    total = len(all_results)
    print("\n" + "=" * 88)
    print(f"  RESULT: {total - mismatches}/{total} benchmarks -- measured complexity class matches intended label")
    print("=" * 88)

    out_path = os.path.join(BENCH_DIR, "validation_report.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report written to {out_path}")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
