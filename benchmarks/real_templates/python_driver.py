"""Real-execution driver for the curated Python template sample."""

import math
import random
import statistics
import time

REPS = 3


def _build_input(input_kind, n, seed=42):
    rng = random.Random(seed)
    if input_kind == "list_int":
        return [rng.randint(-1000, 1000) for _ in range(n)]
    elif input_kind == "string":
        return "".join(rng.choice("abcdefghij") for _ in range(n))
    elif input_kind == "int_direct":
        return n
    else:
        raise ValueError(f"Unknown input_kind: {input_kind}")


def time_template(template_src: str, input_kind: str, n: int) -> float:
    """Exec the template source to define f(), build an input of size n,
    time exactly the call to f() (not input construction)."""
    namespace = {}
    exec(template_src, namespace)
    fn = namespace["f"]
    arg = _build_input(input_kind, n)

    if input_kind == "list_int":
        # Some templates mutate their list argument (e.g. in-place sort).
        # Re-build a fresh copy each rep so later reps aren't measuring a
        # no-op pass over an already-sorted list.
        def run():
            fn(list(arg))
    else:
        def run():
            fn(arg)

    t0 = time.perf_counter()
    run()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def run_python_sample(template_src, input_kind, n_values):
    timings = []
    for n in n_values:
        samples = []
        for _ in range(REPS):
            samples.append(time_template(template_src, input_kind, n))
        timings.append(statistics.median(samples))
    exponents = []
    for i in range(len(n_values) - 1):
        n1, n2 = n_values[i], n_values[i + 1]
        t1, t2 = max(timings[i], 1e-6), max(timings[i + 1], 1e-6)
        exponents.append(math.log(t2 / t1) / math.log(n2 / n1))
    avg_exp = sum(exponents) / len(exponents)
    return timings, exponents, avg_exp


if __name__ == "__main__":
    from selected_templates import PYTHON_SAMPLES

    for tid, intended, template, fills, input_kind, n_values in PYTHON_SAMPLES:
        timings, exponents, avg_exp = run_python_sample(template, input_kind, n_values)
        timing_str = " -> ".join(f"{t:.4f}ms" for t in timings)
        print(f"{tid:24s} n={n_values}")
        print(f"  timings: {timing_str}")
        print(f"  avg exponent: {avg_exp:.2f}")
