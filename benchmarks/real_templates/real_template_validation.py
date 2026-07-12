"""
IntelliX Real-Template Validation
====================================
Phase 2 of the benchmark validation effort. The first benchmark suite
(../benchmark_validation.py) used six hand-written CANONICAL examples per
language, deliberately built to be unambiguous -- and got 18/18 agreement.
This script instead samples REAL templates verbatim from
profiler_main.py's PYTHON_TEMPLATES / CPP_TEMPLATES / JAVA_TEMPLATES --
the actual population generate_dataset() draws training data from -- and
asks the harder question: does the project's own EFFICIENT / MODERATE /
NEEDS_OPTIMIZATION labeling scheme hold up against real measured
execution for the code it actually ships?

SAMPLING CAVEAT (read this before citing any number from this report):
The 19 templates here were hand-picked, NOT randomly sampled, specifically
to probe known risk areas: builtin/stdlib calls that hide a real loop from
a text/AST-based heuristic, the project's own definitional choice to file
single-pass O(n) code under "EFFICIENT", and a cross-language sanity
check (does the same algorithm get the same label in all three
languages?). The agreement rate below describes THIS deliberately
adversarial sample, not the dataset as a whole -- it is a demonstration
that specific, systematic disagreement patterns exist and recur across
all three languages, not a measurement of "what fraction of the dataset
is mislabeled".

Also note: the model trained on generate_dataset() agrees with the
INTENDED bucket label on all 19 of these templates with high confidence
(see model_vs_intended.py). That is not informative -- these templates
are literally drawn from the same finite list the model was trained on,
so the model has learned to reconstruct the heuristic-feature-to-bucket
mapping, not to recognize real complexity. The only comparison with
actual information content is INTENDED LABEL vs REAL MEASURED BEHAVIOR,
which is external ground truth the model never saw.

Usage:
    python real_template_validation.py
"""

import json
import math
import os
import sys

sys.path.insert(0, "/home/claude/intellix")

from selected_templates import PYTHON_SAMPLES, CPP_SAMPLES, JAVA_SAMPLES
from python_driver import run_python_sample
from cpp_driver import run_cpp_sample
from java_driver import run_java_sample

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def classify_exponent(avg_exponent: float) -> str:
    if avg_exponent < 0.5:
        return "EFFICIENT"
    elif avg_exponent < 1.5:
        return "MODERATE"
    else:
        return "NEEDS_OPTIMIZATION"


# Known root-cause tags for templates whose result needs more context than
# a bare MATCH/MISMATCH conveys -- filled in as findings are confirmed.
NOTES = {
    "py_set_dedup": "Was a blind spot: PythonASTAnalyzer had no stdlib weight for Python builtins, so list(set(x)) showed loops=0 despite doing two real O(n) passes. FIXED at the feature-extraction level -- PythonASTAnalyzer.visit_Call now has a KNOWN_COMPLEXITY table (set/list/tuple/dict/sorted/sum/max/min/any/all) and correctly reports stdlib_complexity_weight=2.0 for this exact snippet (set() and list() each add 1.0, deliberately NOT chain-capped the way Java's stream calls are, since CPython eagerly materializes each nested call rather than fusing them the way a lazy Java Stream pipeline does -- see the KNOWN_COMPLEXITY docstring in profiler_main.py for the full reasoning). The MISMATCH below persists anyway, for the same reason as cpp_string_reverse/cpp_std_count/java_stream_upper/java_hashset_ctor: this template is filed under EFFICIENT in PYTHON_TEMPLATES by original design, so the model's bucket prediction doesn't change just because the feature it's fed is now more accurate. A training-label problem, not an extraction problem. See HANDOFF.md / README's 'Honest finding'.",
    "py_genexpr_sum": "Was a blind spot: PythonASTAnalyzer had no visit_GeneratorExp, so 'sum(i*i for i in range(n))' showed loops=0. FIXED at the feature-extraction level -- visit_GeneratorExp now correctly reports loops=1, depth=1, comps=1. The MISMATCH below persists anyway: this template is in the EFFICIENT bucket by the project's design choice that O(n) single-pass IS EFFICIENT. Real measurement says MODERATE under the strict exponent threshold -- same definitional gap as py_single_pass_sum.",
    "py_single_pass_sum": "Definitional, not a bug: this exact O(n) single-pass pattern is filed under EFFICIENT by design (see the source comment in PYTHON_TEMPLATES). Real measurement says O(n)=MODERATE under a strict threshold.",
    "py_bubble_sort": "Label moved from MODERATE → NEEDS_OPTIMIZATION (Tier-2 relabeling based on real execution: exponent≈2.0, exceeds the NEEDS_OPTIMIZATION threshold of >1.5). This entry should now MATCH.",
    "cpp_string_reverse": "Was a heuristic blind spot under the old text-pattern analyzer; std::reverse() is real O(n) work with no 'for'/'while' keyword in the source. FIXED at the feature-extraction level -- cpp_analyzer.py's tree-sitter parser now correctly reports stdlib_complexity_weight=1.0 for this call. The MISMATCH below persists anyway: this template is filed under EFFICIENT in CPP_TEMPLATES by original design, so the model still predicts EFFICIENT regardless of the now-correct feature -- a training-label problem, not an extraction problem. See HANDOFF.md / README's 'Honest finding'.",
    "cpp_std_count": "Was a heuristic blind spot (std::count() is O(n), no for/while keyword). FIXED at the feature-extraction level -- now correctly reports stdlib_complexity_weight=1.0. Same residual MISMATCH cause as cpp_string_reverse: a bucket-label problem, not an extraction problem.",
    "cpp_single_pass_sum": "Definitional, not a bug: same O(n)-filed-as-EFFICIENT pattern as Python, same real-measurement disagreement.",
    "cpp_bubble_sort": "Label moved from MODERATE → NEEDS_OPTIMIZATION (Tier-2 relabeling based on real execution: exponent≈2.0). This entry should now MATCH.",
    "cpp_prime_sieve": "Real bug found via execution, then FIXED (see overflow_bug_report.md): outer loop bound was 'i <= n' instead of 'i <= sqrt(n)', so i*i overflowed a 32-bit int and crashed for n above ~46341*46341 -- confirmed via a real SIGSEGV at n=100,000. Fix (bound the sieving loop by sqrt(n) via an overflow-safe long long cast, move prime collection to a separate pass over the full range) verified against known prime counts (pi(100000)=9592) before landing in CPP_TEMPLATES. The historical crash remains documented here and in overflow_bug_report.md; the template itself no longer reproduces it.",
    "cpp_naive_fib": "Control: recursive, can't be folded by the compiler. Confirms exponential growth correctly.",
    "cpp_quad_nested_foldtest": "Compiler folding, not an analyzer problem: g++ -O2 reduces this exact NEEDS_OPTIMIZATION-bucket template's nested-increment loop to a near-constant-time computation, the same phenomenon discovered while building the canonical benchmark suite -- but this time on real shipped code.",
    "java_stream_upper": "Was a heuristic blind spot under the old text-pattern analyzer; Arrays.stream().map() is O(n), no 'for'/'while' substring present. FIXED at the feature-extraction level -- java_analyzer.py's tree-sitter parser now correctly reports stdlib_complexity_weight=1.0 (chain-capped, not double-counted). The MISMATCH below persists anyway: same bucket-label problem as the C++ cases, not an extraction problem.",
    "java_hashset_ctor": "Was a heuristic blind spot (new HashSet<>(arr) copies all n elements, O(n), no for/while substring). FIXED at the feature-extraction level -- now correctly reports stdlib_complexity_weight=1.0 (and correctly does NOT flag the empty-constructor case). Same residual MISMATCH cause: a bucket-label problem, not an extraction problem.",
    "java_single_pass_sum": "Definitional, not a bug: same O(n)-filed-as-EFFICIENT pattern as the other two languages.",
    "java_bubble_sort": "Label moved from MODERATE → NEEDS_OPTIMIZATION (Tier-2 relabeling based on real execution: exponent≈2.0). This entry should now MATCH.",
    "java_prime_sieve": "Same root-cause integer-overflow bug as C++, then FIXED the same way (bound sieving by sqrt(n) via a long cast, separate prime-collection pass) -- confirmed via a real ArrayIndexOutOfBoundsException at n=100,000 before the fix, and verified against pi(100000)=9592 after it. See overflow_bug_report.md.",
    "java_naive_fib": "Control: recursive, can't be folded/inlined away. Confirms exponential growth correctly.",
    "py_prime_sieve": "Control: Python's version IS correctly bounded by sqrt(n) (the only language that wrote this loop bound correctly) -- no overflow risk, real measurement lines up with the MODERATE label.",
    "py_quad_nested": "Control: visible 4x nested loop, not subject to compiler folding in interpreted Python. Confirms NEEDS_OPTIMIZATION correctly.",
}


def main():
    print("=" * 100)
    print("  IntelliX Real-Template Validation -- intended label vs. real measured execution")
    print("=" * 100)
    print("\n  SAMPLING CAVEAT: 19 templates, hand-picked to probe known risk areas, not a")
    print("  random sample. See this file's module docstring before citing any rate below.\n")

    all_results = []

    runners = [
        ("python", PYTHON_SAMPLES, lambda t, fills, kind, ns: run_python_sample(t, kind, ns)),
        ("cpp", CPP_SAMPLES, lambda t, fills, kind, ns, tid=None: None),  # placeholder, replaced below
        ("java", JAVA_SAMPLES, None),
    ]

    # Python
    print("--- PYTHON ---")
    for tid, intended, template, fills, input_kind, n_values in PYTHON_SAMPLES:
        timings, exponents, avg_exp = run_python_sample(template, input_kind, n_values)
        real_label = classify_exponent(avg_exp)
        result = "MATCH" if real_label == intended else "MISMATCH"
        print(f"  {tid:24s} n={n_values}")
        print(f"    timings: {' -> '.join(f'{t:.4f}ms' for t in timings)}")
        print(f"    exponent={avg_exp:6.2f}  intended={intended:20s} real={real_label:20s} [{result}]")
        print(f"    note: {NOTES.get(tid, '')}")
        all_results.append({"language": "python", "id": tid, "intended": intended, "real_label": real_label,
                             "exponent": round(avg_exp, 3), "timings_ms": timings, "result": result,
                             "note": NOTES.get(tid, "")})

    # C++
    print("\n--- C++ ---")
    for tid, intended, template, fills, input_kind, n_values in CPP_SAMPLES:
        timings, exponents, avg_exp = run_cpp_sample(tid, template, fills, input_kind, n_values)
        real_label = classify_exponent(avg_exp)
        result = "MATCH" if real_label == intended else "MISMATCH"
        print(f"  {tid:28s} n={n_values}")
        print(f"    timings: {' -> '.join(f'{t:.4f}ms' for t in timings)}")
        print(f"    exponent={avg_exp:6.2f}  intended={intended:20s} real={real_label:20s} [{result}]")
        print(f"    note: {NOTES.get(tid, '')}")
        all_results.append({"language": "cpp", "id": tid, "intended": intended, "real_label": real_label,
                             "exponent": round(avg_exp, 3), "timings_ms": timings, "result": result,
                             "note": NOTES.get(tid, "")})

    # Java
    print("\n--- JAVA ---")
    for tid, intended, template, fills, input_kind, n_values in JAVA_SAMPLES:
        timings, exponents, avg_exp = run_java_sample(tid, template, fills, input_kind, n_values)
        real_label = classify_exponent(avg_exp)
        result = "MATCH" if real_label == intended else "MISMATCH"
        print(f"  {tid:24s} n={n_values}")
        print(f"    timings: {' -> '.join(f'{t:.4f}ms' for t in timings)}")
        print(f"    exponent={avg_exp:6.2f}  intended={intended:20s} real={real_label:20s} [{result}]")
        print(f"    note: {NOTES.get(tid, '')}")
        all_results.append({"language": "java", "id": tid, "intended": intended, "real_label": real_label,
                             "exponent": round(avg_exp, 3), "timings_ms": timings, "result": result,
                             "note": NOTES.get(tid, "")})

    total = len(all_results)
    matches = sum(1 for r in all_results if r["result"] == "MATCH")
    print("\n" + "=" * 100)
    print(f"  RESULT: {matches}/{total} of this deliberately curated sample matches its intended label")
    print(f"  (this is NOT a random sample -- see caveat above)")
    print("=" * 100)

    with open(os.path.join(OUT_DIR, "real_template_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report written to real_template_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
