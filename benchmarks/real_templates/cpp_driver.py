"""Real-execution driver for the curated C++ template sample.
Wraps each verbatim template body in a generated main() that builds an
input of size n, times exactly the call to f(), and prints ELAPSED_MS."""

import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, "/home/claude/intellix")
from profiler_main import _fill_template

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_gen")
REPS = 3


def _split_includes(template_src: str):
    """Templates store #include lines appended after the function body on
    their own line(s). Pull them out so they can be hoisted to the top of
    a real compilable file."""
    includes, body_lines = [], []
    for line in template_src.split("\n"):
        if line.strip().startswith("#include"):
            includes.append(line.strip())
        else:
            body_lines.append(line)
    return includes, "\n".join(body_lines)


DO_NOT_OPTIMIZE_HELPER = """
template <class Tp>
inline void doNotOptimize(Tp const& value) {
    asm volatile("" : : "m"(value) : "memory");
}
"""


def _returns_void(template_src: str) -> bool:
    return template_src.lstrip().startswith("void ")


def _main_for(input_kind: str, n: int, is_void: bool) -> str:
    if is_void:
        call_string = {
            "string": "f(s); doNotOptimize(s);",
            "list_int": "f(arr); doNotOptimize(arr);",
            "int_direct": "f(static_cast<int>(n));",
        }[input_kind]
    else:
        call_string = {
            "string": "auto r = f(s); doNotOptimize(r);",
            "list_int": "auto r = f(arr); doNotOptimize(r);",
            "int_direct": "auto r = f(static_cast<int>(n)); doNotOptimize(r);",
        }[input_kind]

    setup = {
        "string": f'std::string s(static_cast<size_t>({n}), \'a\');\n    for (size_t i = 0; i < s.size(); i++) s[i] = \'a\' + (i % 26);',
        "list_int": f'std::vector<int> arr(static_cast<size_t>({n}));\n    for (size_t i = 0; i < arr.size(); i++) arr[i] = static_cast<int>(arr.size() - i);',
        "int_direct": f'long long n = {n}LL;',
    }[input_kind]

    return f"""
int main() {{
    {setup}
    auto t0 = std::chrono::high_resolution_clock::now();
    {call_string}
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    std::cout << "ELAPSED_MS=" << ms.count() << std::endl;
    return 0;
}}
"""


def build_and_run(tid: str, template_src: str, fills: dict, input_kind: str, n: int) -> float:
    code = _fill_template(template_src, **fills) if fills else template_src
    includes, body = _split_includes(code)
    standard_includes = ["#include <iostream>", "#include <chrono>", "#include <string>", "#include <vector>"]
    all_includes = sorted(set(standard_includes + includes))
    is_void = _returns_void(body)

    src_path = os.path.join(GEN_DIR, f"{tid}_n{n}.cpp")
    bin_path = os.path.join(GEN_DIR, f"{tid}_n{n}")
    with open(src_path, "w") as fp:
        fp.write("\n".join(all_includes) + "\n\n")
        fp.write(DO_NOT_OPTIMIZE_HELPER + "\n")
        fp.write(body + "\n\n")
        fp.write(_main_for(input_kind, n, is_void))

    subprocess.run(["g++", "-O2", "-std=c++17", "-o", bin_path, src_path], check=True, capture_output=True)
    out = subprocess.run([bin_path], capture_output=True, text=True, timeout=30).stdout
    for token in out.split():
        if token.startswith("ELAPSED_MS="):
            return float(token.split("=", 1)[1])
    raise RuntimeError(f"No ELAPSED_MS in output: {out!r}")


def run_cpp_sample(tid, template_src, fills, input_kind, n_values):
    timings = []
    for n in n_values:
        samples = [build_and_run(tid, template_src, fills, input_kind, n) for _ in range(REPS)]
        timings.append(statistics.median(samples))
    exponents = []
    for i in range(len(n_values) - 1):
        n1, n2 = n_values[i], n_values[i + 1]
        t1, t2 = max(timings[i], 1e-6), max(timings[i + 1], 1e-6)
        exponents.append(math.log(t2 / t1) / math.log(n2 / n1))
    avg_exp = sum(exponents) / len(exponents)
    return timings, exponents, avg_exp


if __name__ == "__main__":
    from selected_templates import CPP_SAMPLES

    for tid, intended, template, fills, input_kind, n_values in CPP_SAMPLES:
        timings, exponents, avg_exp = run_cpp_sample(tid, template, fills, input_kind, n_values)
        timing_str = " -> ".join(f"{t:.4f}ms" for t in timings)
        print(f"{tid:28s} n={n_values}")
        print(f"  timings: {timing_str}")
        print(f"  avg exponent: {avg_exp:.2f}")
