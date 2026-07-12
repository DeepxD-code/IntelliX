# IntelliProfile (IntelliX)

**ML-powered multi-language code performance profiler** — paste code, get instant complexity metrics, ML predictions, and optimization recommendations.

[![CI](https://github.com/DeepxD-code/IntelliX/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepxD-code/IntelliX/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-passing-brightgreen)

## Features

- **3 languages**: Python (AST-based), C++ & Java (tree-sitter-based)
- **4 ML classifiers**: Random Forest, Gradient Boosting, SVM, Neural Network (auto-selects best via GridSearchCV)
- **Metrics**: execution time, memory usage, loop depth, nesting depth, function calls, conditionals, complexity score (+ cyclomatic complexity for Java)
- **Prediction**: EFFICIENT / MODERATE / NEEDS_OPTIMIZATION with confidence & class probabilities
- **Config-driven**: all thresholds, model params, dataset size via `config.yaml`
- **Model versioning**: ModelRegistry with version tracking and rollback support
- **Structured logging**: JSON-formatted logs for production monitoring
- **REST API**: Swagger-documented endpoints at `/api/docs`, JSON descriptor at `/`
- **MCP server**: the same profiler exposed as MCP tools for Claude Desktop/Code
- **Opt-in real execution (Python)**: `profiling.enable_execution: true` measures actual subprocess wall-clock time instead of a formula estimate — off by default, see "Python execution sandbox" below
- **Docker support**: single-command deployment, model trained at build time (not on first request)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│ API client  │────▶│  Flask API   │────▶│  CodeProfiler    │
│ (curl/MCP)  │     │  app.py      │     │  (unified entry) │
└─────────────┘     └──────────────┘     └───────┬──────────┘
                         │                       │
                         ▼                       ▼
                  ┌──────────────┐     ┌──────────────────┐
                  │  SQLite DB   │     │  Language-Specific│
                  │  history     │     │  Profilers        │
                  └──────────────┘     │  (AST / parser)   │
                                       └───────┬──────────┘
                                               │
                                               ▼
                                        ┌──────────────────┐
                                        │  MLPipeline      │
                                        │  4 classifiers   │
                                        │  + GridSearchCV  │
                                        └───────┬──────────┘
                                               │
                                        ┌──────▼───────┐
                                        │  ModelRegistry│
                                        │  versioned    │
                                        │  model_v*.pkl │
                                        └──────────────┘
```

**On analysis depth, precisely:** Python uses real AST parsing (`ast` module) — loop/conditional/call counts and nesting depth are exact. C++ and Java now use real parsing too, via [`tree-sitter`](https://tree-sitter.github.io/tree-sitter/) (`cpp_analyzer.py` / `java_analyzer.py`), which replaced the original text/brace-counting heuristic — see "Validation & Benchmarks" below for why that mattered and what changed. On top of each parse, a `KNOWN_COMPLEXITY` lookup recognizes common standard-library/builtin calls that hide real loop-equivalent work behind a single call with no `for`/`while` keyword in the source — `std::sort`/`std::reverse`/`std::count` in C++, `.stream()`/`.contains()`/`new HashSet<>(arr)` in Java, and (added this round) `set()`/`list()`/`sorted()`/`sum()`/`max()`/`min()`/`any()`/`all()`/etc. in Python — and folds a calibrated weight into both the `complexity_score` formula and its own ML feature, `stdlib_complexity_weight`. Python's version is deliberately *not* chain-capped the way Java's is: `list(set(x))` weights both calls independently (2.0 total, not capped to 1.0), because CPython eagerly materializes each nested call — `set()` and `list()` really are two separate O(n) passes, unlike Java's `.stream().map().collect()`, which the JVM lazily fuses into one. The one deliberate exception is a bare generator-expression argument (`sum(x for x in y)`) — that genuinely is a single fused pass, same fusion behavior as a Java Stream, so it isn't double-counted against the loop `visit_GeneratorExp` already reports. See the `KNOWN_COMPLEXITY` docstring in `profiler_main.py` for the full reasoning.

Execution time for C++ and Java remains a formula estimate derived from static counts, **not measured execution** — nothing is compiled or run for either language by the live API, deliberately, since compiling and running untrusted submitted code is a real security problem this project doesn't take on. Python is the one exception: `profiling.enable_execution` in `config.yaml` (off by default) switches `execution_time_ms` from the formula estimate to a real subprocess-measured wall-clock time — see "Python execution sandbox" below. `memory_usage_kb` remains a formula estimate for all three languages regardless of this setting; nothing in this project measures real memory usage. A real parser (or, for Python, real execution) closes blind spots in *feature extraction*, but it does not and cannot fix mislabeled *training data* — see the "Honest finding" callout below for a concrete case where that distinction matters, and a concrete case where fixing the label (not the feature) was the right move.

## Quick Start

### Local

```bash
pip install -r requirements.txt
python profiler_main.py     # Train model + CLI demo
python app.py               # Start REST API at http://localhost:5000 (see /api/docs for Swagger UI)
```

### Docker

```bash
docker compose build
docker compose up -d
# -> http://localhost:5000/api/docs
```

The Docker image trains and bakes a model in at **build time** (`RUN python profiler_main.py` in the Dockerfile), not on first request — a cold container with no persisted model would otherwise block on a multi-minute `GridSearchCV` sweep before answering even a health check.

### MCP Server

```bash
python mcp_server.py
```

Exposes the same `CodeProfiler` as three MCP tools for Claude Desktop/Code:

| Tool | Equivalent to |
|---|---|
| `analyze_code(code, language)` | `POST /api/profile` |
| `batch_analyze_code(snippets)` | `POST /api/batch-profile` |
| `get_analysis_info()` | model metadata **+ what's real-parsed vs. formula-estimated per language** |

Register it by pointing an MCP config at the file:

```json
{
  "mcpServers": {
    "intellix": { "command": "python", "args": ["/absolute/path/to/mcp_server.py"] }
  }
}
```

### Python execution sandbox (opt-in real execution)

By default, `execution_time_ms` for all three languages is a formula estimate derived from static analysis. For Python specifically, there's an opt-in alternative: set `profiling.enable_execution: true` in `config.yaml` and `execution_time_ms` instead comes from actually running the submitted code.

```yaml
profiling:
  enable_execution: true              # default: false
  execution_timeout_seconds: 30       # hard wall-clock cutoff
```

How it works (`execute_python_sandboxed()` in `profiler_main.py`): the submitted code is wrapped in a small timing harness (`time.perf_counter()` before/after, exceptions caught so a crashing snippet still returns *a* number rather than blowing up the request) and run via `subprocess.run()` with a hard timeout. On timeout, a crash, or unparseable output, the function returns `None` and `profile_python()` silently falls back to the formula estimate — a broken or slow snippet never breaks the API response, it just loses the "real measurement" upgrade for that one call. Every response includes a `measured_execution: true/false` field so a caller can tell which path actually produced the number.

**Stated plainly, since it's easy to miss:** the harness's `try/except Exception: pass` means code that *raises* (a `ZeroDivisionError`, an unhandled `KeyError`, anything) still produces a real, valid timing number — the time it took to reach and absorb the exception, not a sentinel for "this failed." A snippet that crashes immediately will report a very small `execution_time_ms`, which is technically true (that's how long it ran before failing) but easy to misread as "fast and efficient" if you don't know this is happening. This was a deliberate tradeoff — the alternative (propagating the exception and returning `None`, i.e. falling back to the formula estimate) was rejected because it would make error-prone code look identical to timeout-prone code in the response, losing the "this ran, briefly, then failed" signal entirely. No caching or memoization is implemented either: with the flag on, every request re-spawns a subprocess, even for a snippet that's been profiled before. Fine for a demo/single-user context; worth adding before enabling this in front of real traffic.

**Security scope, stated plainly:** this is subprocess-level isolation only — the child process runs as the same user with the same filesystem access as the parent. That's an appropriate tradeoff for a trusted-user or local/demo deployment, and it's a real, meaningful upgrade in *accuracy* over the formula estimate. It is **not** a sandbox suitable for running arbitrary untrusted code from the public internet — there's no seccomp filter, no network isolation, no filesystem jail, and no memory/CPU rlimit beyond the wall-clock timeout. For a public multi-tenant deployment, put a real sandboxing layer (gVisor, nsjail, Firecracker) in front of this, or leave the flag off and keep the (safe-by-construction) formula estimate. This is also why C++ and Java don't get an equivalent — compiling and linking untrusted C++, or starting a JVM for untrusted Java, both have a larger and harder-to-bound attack surface than running Python source in a timed subprocess, and neither was in scope for this pass.

## Project Structure

```
IntelliX/
├── app.py                     # Flask web app (API endpoints + Swagger + rate limiting)
├── profiler_main.py           # Core: AST analyzer, ML pipeline, profiler, config
├── mcp_server.py              # MCP tool wrapper around CodeProfiler
├── config.yaml                # YAML configuration (thresholds, ML, logging)
├── test_profiler.py           # Unit tests
├── test_app.py                 # API integration tests
├── requirements.txt           # Pinned Python dependencies
├── Dockerfile                  # Container image (trains model at build time)
├── docker-compose.yml          # Orchestration
├── gunicorn.conf.py            # WSGI server config (see comments on worker count)
├── .dockerignore
├── Makefile                    # make test / make run / make docker-build
├── pyproject.toml               # Ruff linter config
├── benchmarks/                  # Real-measurement validation (see below)
│   ├── benchmark_validation.py
│   ├── cpp/  java/              # canonical benchmark sources
│   └── real_templates/          # audit of the actual synthetic templates
├── .github/workflows/ci.yml     # GitHub Actions CI (tests + benchmark validation)
├── models/                      # Saved ML models (gitignored; baked into Docker image)
├── logs/                        # JSON log output (gitignored)
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API descriptor (name, docs link, health link, current model info) — JSON, not a web page |
| GET | `/api/health` | Health check (reflects actual backend readiness, returns 503 if not ready) |
| POST | `/api/profile` | Profile single code snippet |
| POST | `/api/batch-profile` | Profile multiple snippets |
| GET | `/api/model-info` | Model details & features |
| GET | `/api/history` | Recent profile history (SQLite, WAL mode) |
| GET | `/api/docs` | Swagger API documentation |

All endpoints are rate-limited per `config.yaml`'s `server.max_requests_per_minute` (default 100/min per client). See `gunicorn.conf.py` for why this requires running with a single worker process by default unless `RATELIMIT_STORAGE_URI` is pointed at a shared backend.

### Profile Request

```json
POST /api/profile
{
  "code": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
  "language": "python"
}
```

### Profile Response

```json
{
  "metrics": {
    "execution_time_ms": 5.23,
    "memory_usage_kb": 640,
    "loop_depth": 0,
    "complexity_score": 3.42,
    ...
  },
  "ml_prediction": {
    "label": "EFFICIENT",
    "confidence": 0.97,
    "probabilities": {
      "EFFICIENT": 0.97,
      "MODERATE": 0.02,
      "NEEDS_OPTIMIZATION": 0.01
    }
  },
  "recommendations": [
    "Code appears well-structured and efficient.",
    "Consider adding type hints for better maintainability."
  ],
  "analysis_time_ms": 1.23
}
```

## ML Pipeline

- **4 classifiers**: Random Forest, Gradient Boosting, SVM, MLP (Neural Network)
- **Hyperparameter tuning**: GridSearchCV with 3-fold cross-validation
- **Training data**: synthetic, template-generated samples balanced across 3 complexity classes — see "Validation & Benchmarks" for what this does and doesn't mean
- **11 features**: execution_time_ms, memory_usage_kb, loop_depth, max_nesting_depth, function_calls, conditionals, complexity_score, stdlib_complexity_weight, recursive_branching_risk, element_swap_weight, language
- **Classes**: EFFICIENT, MODERATE, NEEDS_OPTIMIZATION
- **Persistence**: versioned model files (`model_v<timestamp>.joblib + scaler + metadata`)
- **ModelRegistry**: list versions, get latest, and rollback to a *specific* version (this loads that exact version's artifacts, not just whatever happens to be newest)

## Configuration

All settings in `config.yaml`:

```yaml
ml_model:
  classifier_type: "random_forest"   # or gradient_boosting, svm, neural_network
  hyperparameter_tuning: true
  cross_validation_folds: 5

dataset:
  n_samples: 3000

thresholds:
  execution_time_ms: 100.0
  loop_depth: 3

server:
  host: "0.0.0.0"
  port: 5000
  max_requests_per_minute: 100
  max_batch_size: 50
```

Override with `CONFIG_PATH` env var. Override the rate-limit storage backend with `RATELIMIT_STORAGE_URI` (defaults to in-process memory, correct only for a single worker — see `gunicorn.conf.py`). Override worker/thread counts with `WEB_CONCURRENCY` / `WEB_THREADS`.

## Validation & Benchmarks

Unit tests check that the code runs correctly. They don't check whether the EFFICIENT / MODERATE / NEEDS_OPTIMIZATION labels mean anything in the real world. Two benchmark suites do, using real measured execution (`std::chrono`, `System.nanoTime`, `time.perf_counter`) rather than the static, parsed features the live API uses.

### Canonical benchmark suite (`benchmarks/`)

Six hand-written, deliberately unambiguous functions per language (constant-time, linear, quadratic, exponential) at increasing input sizes. Fits an empirical growth exponent to the measured timings and checks it against the intended complexity class.

```bash
python benchmarks/benchmark_validation.py
```

**Result: 18/18** — every canonical example's measured behavior matches its label, in all three languages. Caught two real measurement gotchas along the way (documented in the script itself): an optimizing compiler folding a "quadratic" loop into a constant-time computation, and the JVM JIT compiling a hot loop mid-run and distorting the growth curve. Re-confirmed 18/18 after the tree-sitter parser upgrade below — expected, since this suite measures real execution and never touches the analyzer at all.

### Real-template audit (`benchmarks/real_templates/`)

A harder, slower question: do the labels on the *actual* templates `generate_dataset()` draws from hold up under real execution? 19 templates sampled directly from `PYTHON_TEMPLATES` / `CPP_TEMPLATES` / `JAVA_TEMPLATES` — **not a random sample**, deliberately chosen to probe specific risk areas. Read the script's docstring before citing any number from it.

```bash
python benchmarks/real_templates/real_template_validation.py
```

**Result: 9/19 match**, up from 6/19 in the prior round — every point of movement is accounted for below, and it's a real improvement, not a metric that got easier to pass. **Sample-size caveat, stated plainly:** 19 is a small, deliberately adversarial sample (see below), not a statistically representative one — treat "9/19" as a directional signal about known risk areas, not a precision measurement of overall label accuracy. Confirmed, reproducible findings:

- **Heuristic blind spot — fixed at the feature-extraction level, in all three languages that had one.** Code that hides a real O(n) loop inside a builtin/stdlib call (`set()`, `list()`, a Python generator expression, `std::reverse`, `std::count`, `Arrays.stream()`, `new HashSet<>(arr)`) was invisible to the original text/brace-counting C++/Java analyzer, and to Python's AST analyzer both for generator expressions and for builtin calls like `set()`/`list()` specifically. C++ and Java parse with [`tree-sitter`](https://tree-sitter.github.io/tree-sitter/) and weight known stdlib calls via a `KNOWN_COMPLEXITY` lookup (`cpp_analyzer.py` / `java_analyzer.py`), including correctly capping chained calls like `.stream().map().collect()` at one unit of work rather than counting every link, and distinguishing a copying collection constructor (`new HashSet<>(arr)`, O(n)) from an empty one (`new HashSet<>()`, O(1)). Python's `PythonASTAnalyzer` now has `visit_GeneratorExp` (it didn't before), so `sum(i*i for i in range(n))` correctly reports one loop at depth one instead of silently reporting zero — and now also has its own `KNOWN_COMPLEXITY` table for builtins, so `list(set(x))`-style deduplication correctly reports `stdlib_complexity_weight=2.0` instead of the `0.0` it silently reported before. See "On analysis depth, precisely" above for why Python's weighting is deliberately not chain-capped the same way Java's is.
- **Resolved definition, stated plainly: single-pass O(n) is EFFICIENT, on purpose.** The EFFICIENT bucket includes single-pass O(n) code — `single_pass_sum` in all three languages, and for the same reason, `py_genexpr_sum`. The rationale: EFFICIENT is meant to answer "does a meaningfully better algorithm exist for this problem shape," not "is this O(1)" — a single pass over data that fundamentally requires touching every element has no better algorithm available, so labeling it EFFICIENT is accurate, not generous. This was carried as an open "definitional question, needs a human decision" across several rounds; it's resolved now, with this stated reasoning as the actual definition rather than a placeholder for one. If you're citing "EFFICIENT" as a synonym for O(1) specifically, it isn't that here — it's "no better algorithm exists," which is the more useful thing to know anyway.
- **Relabeled, not just diagnosed:** the same bubble-sort implementation — real, textbook O(n²), empirically measured at growth exponent ≈2.0 via this exact audit — was filed under MODERATE in all three languages. That's now fixed directly in `PYTHON_TEMPLATES` / `CPP_TEMPLATES` / `JAVA_TEMPLATES`: bubble sort moved to the NEEDS_OPTIMIZATION bucket, with the measured exponent as the empirical justification recorded in a source comment at each of the three call sites. Unlike the stdlib-call cases above, this one **does** now show MATCH in all three languages — see the "Honest finding" callout below for why that contrast is the whole point.

  **A second, separate check worth being precise about:** the audit above compares the template's *design label* against *real measured execution* — it never calls the trained ML model. Whether the live model's own prediction for this exact snippet reflects the corrected label is a different question, and the answer differs by language: retraining on the corrected labels, the model now predicts `NEEDS_OPTIMIZATION` for C++ and Java bubble sort (confirmed directly), but Python's classifier still predicts `MODERATE` for the identical algorithm (confirmed directly, including with a much larger, unbounded-depth forest — so it's not a hyperparameter-tuning shortfall). The cause, verified by inspecting the training data directly: Python's bubble-sort feature vector (`loop_depth=2, max_nesting_depth=2`) sits in a region where 151 MODERATE-labeled Python rows outnumber 21 NEEDS_OPTIMIZATION-labeled rows in the training set, because many other MODERATE double-nested-loop templates produce an *identical* AST-derived feature signature — the formula-based features can't tell "nested loop with a cheap body" from "nested loop with a swap" apart; only real execution time would. This is a sharper version of the same underlying lesson: relabeling one template fixes that template's *design-vs-reality* audit result unconditionally, but only fixes the *trained model's* prediction for it when the surrounding feature space isn't already dominated by differently-labeled neighbors — which, for Python specifically, it currently is. Fixing this for real would mean training on real execution-measured features across the dataset, not just relabeling one template — see `HANDOFF.md` for why that's a larger, separately-scoped piece of work this round didn't take on.
- **A real bug, found via execution — now fixed, verified by execution again.** The C++ and Java prime-sieve templates had an unbounded outer loop (`i <= n` instead of `i <= sqrt(n)`), causing a 32-bit integer overflow that **crashed** the C++ version and **threw** `ArrayIndexOutOfBoundsException` in Java once `n` exceeded roughly 46,341. Python's version was bounded correctly from the start and never had this bug. Fixed in both `CPP_TEMPLATES`/`JAVA_TEMPLATES` and `selected_templates.py`'s copies, verified by direct compilation and execution — not just re-reading the diff — against known prime counts (π(100,000) = 9,592) and confirming no crash/exception at the exact `n` that used to fail. Full writeup, including why the obvious one-line fix would have been a *worse*, silent bug (naively bounding the whole loop stops prime *collection* early too, not just sieving): `benchmarks/real_templates/overflow_bug_report.md`. **Worth being precise about:** this fix does not move the 9/19 number above — both sieve entries were already `MATCH` before it, because the audit's own scaling `n` values (5,000 / 15,000 / 40,000) were deliberately chosen to stay safely below the ~46,341 overflow threshold, so the audit itself never exercised the crash either before or after the fix. The bug was real and is now fixed; the audit result is unaffected because the audit was never the thing that caught it.

**Honest finding — a fixed feature doesn't automatically fix a prediction, and neither does a fixed label, always.** The stdlib-call cases (`std::reverse`, `std::count`, `Arrays.stream().map().collect()`, `new HashSet<>(arr)`, and now the Python generator-expression case) all correctly report their real complexity in the *feature* the model sees — `stdlib_complexity_weight: 1.0` for the C++/Java cases, `loops: 1` instead of `loops: 0` for the Python case — verified end-to-end through the analyzers directly, through `CodeProfiler.profile_python/profile_cpp/profile_java`, through the live `/api/profile` endpoint, and through the MCP `analyze_code` tool. But the live model's *bucket prediction* for those specific snippets is still `EFFICIENT`, unchanged, because those templates are filed under EFFICIENT by design (the single-pass-O(n)-is-EFFICIENT choice) — the feature fix gave the model better eyesight, it didn't retrain the model or move the template to a different bucket, and for these particular templates the bucket was never wrong to begin with under the project's own definition.

Bubble sort was the deliberate contrast case, and the full arc is worth recording rather than just the ending. The AST/parse features were *already* correct (loop nesting depth 2, no stdlib call involved) — the problem was never feature extraction, it was that the template itself was filed under the wrong bucket. Moving it from `MODERATE` to `NEEDS_OPTIMIZATION` fixed the *design-vs-real-execution audit* immediately, unconditionally, in all three languages — but checking the *trained model's own prediction* directly (not assuming the label fix would propagate) turned up a real gap: true for C++ and Java, still false for Python, because Python's feature vector for this template sat in a region of the training data a second, differently-labeled template (`factorial-sum`) also occupied, diluting the signal. Two fix attempts at that gap — real-execution-derived training features, and checking whether the diluting template was itself mislabeled — were tried and both empirically failed; full account of both, including why, in `HANDOFF.md`. **A third approach did work**: rather than trying to give the model a better *time* signal, or relabel around the dilution, a detector for the actual *structural* difference between bubble sort and its neighbors — does the loop body contain a genuine in-place swap, not just a comparison — closed the gap directly, validated the same way as the recursion detector above (swept every template first; found exactly one match, bubble sort itself, before trusting it). The lesson from the two failures still holds — a fixed label doesn't automatically fix a trained model's behavior, and guessing at *why* before checking wastes effort on approaches that don't discriminate the way you'd hope. What changed is that a third, more targeted approach — built on a precise mechanistic diagnosis rather than a first intuition — closed it. See "Model self-consistency audit" below for the full, current numbers (2/230 remaining, not related to bubble sort at all anymore).

None of the C++/Java behavior changes anything about how the live API/MCP tool analyzes arbitrary submitted code for those two languages — that remains static, parse-based analysis without compiling or executing C++/Java input, by design, since compiling and running untrusted input is a real security problem this project deliberately doesn't take on. Python is the one exception, and it's opt-in and off by default — see "Python execution sandbox" above.

### Model self-consistency audit: does the trained model agree with its own training labels?

A different, narrower question than the real-template audit above: not "does the label match real execution," but "does the trained classifier actually predict the bucket its own training data says it should." Every template in `PYTHON_TEMPLATES` / `CPP_TEMPLATES` / `JAVA_TEMPLATES` (230 total) was run through the actual trained model and checked against the bucket it's filed under. Started at **7/230 mismatches (~3%)**.

**Two of the seven were genuine labeling bugs, not model weaknesses — both fixed.** A recursive template computing `2*f(n-1)+1` was filed under `NEEDS_OPTIMIZATION` in Python and C++ — it superficially resembles the Tower-of-Hanoi recurrence (genuinely exponential), but computing this specific value makes exactly one recursive call per invocation. Verified directly (compiled and run in both languages): exactly n calls at n=10/20/30, not 2ⁿ. The value returned grows exponentially; the computation to get there doesn't — those got conflated when the template was written. A second, near-identical bug was found while building the fix below: a *different* naive-Fibonacci template, differing only in a cosmetic base case (`n<=2: return 1` vs. `n<=1: return n`), was filed `MODERATE` in Python and Java while its otherwise-identical sibling was correctly `NEEDS_OPTIMIZATION` — verified exponential too (109 → 13,529 → 1,664,079 calls at n=10/20/30). Both fixed, in every language that had them.

**The rest needed a real feature, not a label fix, and both were built and validated before being trusted.** Naive Fibonacci (genuinely exponential — 177 → 21,891 → 2,692,537 calls) was predicted `MODERATE` in Python and Java. The obvious fix, counting self-calls and flagging 2+, would also flag merge sort, quicksort, and tree traversal — all correctly `MODERATE`, all making 2 self-calls too. The actual difference — divide-and-conquer shrinks its argument each call, naive Fibonacci barely does — isn't visible in a call count alone, and the three safe examples shrink their input three different syntactically ways (slicing, filtering, structural descent into a tree), so a naive version would trade a known small problem for a bigger, worse one. Built a detector for exactly those three patterns instead — `recursive_branching_risk` — and, critically, **tested it against every recursive template in the codebase before trusting it, not just the cases it was designed for**. The first version failed on quicksort (its shrinking happens in an earlier variable assignment, not at the call site) — caught by that same sweep, fixed, re-verified. Final result: flags exactly the genuinely-exponential templates (naive Fibonacci, its base-case variant, a triple-recursion version, two other genuinely-exponential backtracking templates), zero false positives across all 82 EFFICIENT/MODERATE templates tested. Ported to C++ and Java using the equivalent tree-sitter node types, same validation discipline, same result.

Separately, bubble sort was still predicted `MODERATE` — the dilution case discussed above, where a second template (`factorial-sum`) sharing its exact static feature bucket has zero conditionals and drags the average down. Checked whether *any* correctly-labeled template in the entire codebase contains a genuine in-place element swap (`arr[i], arr[j] = arr[j], arr[i]`-shaped tuple assignment) before building anything: exactly one match, bubble sort itself — meaning a swap detector has zero risk of misclassifying anything else, an unusually clean signal. Built `element_swap_weight`, verified the same way (swept every template first), wired in.

**Result: 2/230**, down from 7. Model test accuracy: 99.33% (up from 94.33% at the start of this investigation).

**The two that remain, and precisely why they're not force-fixed:**
- **Python's `factorial-sum` template** (still predicted `MODERATE`, intended `NEEDS_OPTIMIZATION`) — its real cost comes from repeated big-integer multiplication with a growing operand, not from anything a loop-shape or swap detector can see. Already checked whether this template is simply mislabeled instead (it isn't — verified real execution timing genuinely grows super-linearly: 0.052ms → 0.248ms → 0.964ms at n=50/100/200, more than a clean O(n²) doubling would predict). A real fix would need to detect "this loop multiplies a value that grows without bound," a meaningfully different and harder static-analysis problem than either feature built above — not attempted without the same validation discipline the first two required.
- **Java's DP-based Fibonacci** (predicted `EFFICIENT`, intended `MODERATE`) — an O(n)-time, O(n)-space iterative solution. This one is genuinely a boundary case, not a clear miss: a same-time, *better-space* O(1) iterative alternative exists, which is arguably exactly the kind of "meaningfully better algorithm available" that this project's own resolved EFFICIENT-definition (see above) says should keep something out of `EFFICIENT`. Under that reading, the template's `MODERATE` label may be more correct than the model's `EFFICIENT` prediction — a different, subtler definitional question (time-vs-space tradeoffs) than the one already resolved, and not one to resolve unilaterally alongside two already-substantial feature additions in the same pass.

**Worth being honest about, and reported rather than smoothed over**: fixing labels and retraining doesn't move one number in isolation — each retrain reshapes the whole decision surface. Across this investigation, several mismatches disappeared as side effects of fixes aimed at something else (C++'s factorial mismatch, an insertion-sort mismatch that appeared and then disappeared again), not because anyone targeted them directly. That's a real property of retraining a classifier on shared data, not a bug in any specific fix — every number in this section came from actually running the retrained model against all 230 templates, not from assuming a fix worked.

**Also worth recording plainly**: while making the second label fix, an editing mistake was caught before it shipped — a first attempt at removing one mislabeled template accidentally deleted four unrelated, correctly-labeled ones in the same edit (a Sudoku solver, N-Queens, a permutation-backtracking variant, and a subset-sum search). Caught by checking the resulting template count against what it should have been before moving on, not after — all four restored, final counts verified programmatically. Mentioned here because the same verify-before-trusting discipline that caught the quicksort false-positive above is what caught this too, and it's more honest to show that discipline catching a real mistake than to only show it working cleanly.

## Testing

```bash
# All tests
make test

# Unit tests only
python test_profiler.py

# API integration tests only
python test_app.py

# Benchmark validation (real execution, not unit tests)
python benchmarks/benchmark_validation.py
python benchmarks/real_templates/real_template_validation.py

# Lint
make lint
```

CI (`.github/workflows/ci.yml`) runs the unit/API test suite across Python 3.10–3.12, plus a dedicated job running the canonical benchmark suite as a real-execution smoke test on every push.

## Production Notes

- **Model training happens at Docker build time**, not request time — see Dockerfile comments. Without this, a cold container can take several minutes before serving its first request, including health checks.
- **Rate limiting** is enforced (`flask-limiter`), but its default in-memory storage is only correct with a single worker process. `gunicorn.conf.py` defaults to `workers=1, threads=4` for exactly this reason. Scale workers only after pointing `RATELIMIT_STORAGE_URI` at a shared backend (e.g. Redis).
- **SQLite history** runs in WAL mode with an extended busy-timeout for better concurrent-access tolerance, but the database file itself needs a persistent disk/volume to survive restarts — on an ephemeral-filesystem hosting platform, history will reset on every redeploy. Migrating to a hosted database is a future option, not something this project currently needs.
- **CORS** defaults to same-origin only; set `ALLOWED_ORIGINS` if serving the frontend from a different origin than the API.
- No authentication is implemented. This is a deliberate scope choice for a single-tenant demo deployment, not an oversight to silently work around.

## License

MIT
