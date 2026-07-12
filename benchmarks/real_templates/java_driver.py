"""Real-execution driver for the curated Java template sample.
Wraps each verbatim template body in a generated class with a main()
that builds an input of size n and times exactly the call to f()."""

import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, "/home/claude/intellix")
from profiler_main import _fill_template

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "java_gen")
REPS = 3

STANDARD_IMPORTS = "import java.util.*;\nimport java.util.stream.*;\n"


def _returns_void(template_src: str) -> bool:
    return template_src.lstrip().startswith("public void ")


def _setup_for(input_kind: str, n: int) -> str:
    if input_kind == "string_array":
        return (
            f"String[] words = new String[{n}];\n"
            f"        for (int i = 0; i < {n}; i++) words[i] = \"word\" + i;"
        )
    elif input_kind == "list_int":
        return (
            f"List<Integer> arr = new ArrayList<>();\n"
            f"        for (int i = 0; i < {n}; i++) arr.add({n} - i);"
        )
    elif input_kind == "int_direct":
        return f"int n = {n};"
    else:
        raise ValueError(input_kind)


def _call_for(input_kind: str, is_void: bool) -> str:
    arg = {"string_array": "words", "list_int": "arr", "int_direct": "n"}[input_kind]
    if is_void:
        return f"Holder.f({arg});"
    return f"Object r = Holder.f({arg});"


def build_and_run(tid: str, template_src: str, fills: dict, input_kind: str, n: int) -> float:
    code = _fill_template(template_src, **fills) if fills else template_src
    is_void = _returns_void(code)
    classname = f"Gen_{tid}_{n}".replace("-", "_")

    # The template is a single method body (e.g. "public List<String> f(...) {...}").
    # Wrap it as a static method inside a holder class so it can be called
    # from main() without re-declaring imports per template.
    method = code.replace("public ", "public static ", 1)

    setup = _setup_for(input_kind, n)
    call = _call_for(input_kind, is_void)

    src = f"""{STANDARD_IMPORTS}
public class {classname} {{
    static class Holder {{
        {method}
    }}
    public static void main(String[] args) {{
        {setup}
        long t0 = System.nanoTime();
        {call}
        long t1 = System.nanoTime();
        System.out.println("ELAPSED_MS=" + (t1 - t0) / 1_000_000.0);
    }}
}}
"""
    src_path = os.path.join(GEN_DIR, f"{classname}.java")
    with open(src_path, "w") as fp:
        fp.write(src)

    subprocess.run(["javac", "-d", GEN_DIR, src_path], check=True, capture_output=True, text=True)
    out = subprocess.run(
        ["java", "-Xint", "-cp", GEN_DIR, classname],
        capture_output=True, text=True, timeout=30,
    ).stdout
    for token in out.split():
        if token.startswith("ELAPSED_MS="):
            return float(token.split("=", 1)[1])
    raise RuntimeError(f"No ELAPSED_MS in output: {out!r}")


def run_java_sample(tid, template_src, fills, input_kind, n_values):
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
    from selected_templates import JAVA_SAMPLES

    for tid, intended, template, fills, input_kind, n_values in JAVA_SAMPLES:
        timings, exponents, avg_exp = run_java_sample(tid, template, fills, input_kind, n_values)
        timing_str = " -> ".join(f"{t:.4f}ms" for t in timings)
        print(f"{tid:24s} n={n_values}")
        print(f"  timings: {timing_str}")
        print(f"  avg exponent: {avg_exp:.2f}")
