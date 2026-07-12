"""
IntelliX MCP Server
====================
Exposes the existing CodeProfiler / MLPipeline as MCP tools, so any
MCP-compatible agent (Claude Desktop, Claude Code, etc.) can call into
IntelliX directly instead of going through the Flask web UI.

This is a thin wrapper -- it reuses profiler_main.py's CodeProfiler,
MLPipeline, and ConfigManager exactly as the Flask app does. It does not
duplicate or reimplement any analysis logic.

Run directly for a local stdio MCP server:
    python mcp_server.py

Register with Claude Desktop / Claude Code by pointing an MCP server
config at this file (see README section added below).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from profiler_main import (
    COMPLEXITY_LABELS,
    FEATURE_COLUMNS,
    CodeProfiler,
    ConfigManager,
    MLPipeline,
    generate_dataset,
)

mcp = FastMCP("intellix")

config = ConfigManager()

# Lazy-initialized so importing this module (e.g. for tests) doesn't
# immediately train or load a model.
_profiler: CodeProfiler | None = None
_model_info: dict = {}

SUPPORTED_LANGUAGES = ("python", "cpp", "java")

ANALYSIS_METHOD_NOTE = {
    "python": (
        "Real static analysis via Python's built-in `ast` module -- loop/"
        "conditional/call counts and nesting depth are exact. By default, "
        "execution_time_ms is a formula estimate derived from those AST "
        "counts, not measured execution. If profiling.enable_execution is "
        "set to true in config.yaml, execution_time_ms instead comes from "
        "actually running the submitted code in a subprocess with a "
        "wall-clock timeout (subprocess-level isolation only -- not "
        "hardened for untrusted multi-tenant use; see "
        "execute_python_sandboxed() in profiler_main.py). Every response "
        "includes a measured_execution boolean so a caller can tell which "
        "path produced a given result. memory_usage_kb always remains a "
        "formula estimate regardless of this setting."
    ),
    "cpp": (
        "Real static analysis via tree-sitter (`cpp_analyzer.py`) -- loop/"
        "conditional/call counts and nesting depth come from the actual "
        "parse tree, not text/brace counting. A KNOWN_COMPLEXITY lookup "
        "also recognizes common stdlib calls (std::sort, std::reverse, "
        "std::count, etc.) that hide loop-equivalent work. The code is "
        "never compiled or executed -- execution_time_ms and "
        "memory_usage_kb remain formula estimates derived from the parse, "
        "not measured execution."
    ),
    "java": (
        "Real static analysis via tree-sitter (`java_analyzer.py`) -- loop/"
        "conditional/call counts and nesting depth come from the actual "
        "parse tree, not text/brace counting. A KNOWN_COMPLEXITY lookup "
        "also recognizes common stdlib calls (.stream() chains, "
        ".contains(), copying collection constructors, etc.) that hide "
        "loop-equivalent work. No JVM is started and no bytecode is "
        "inspected -- execution_time_ms and memory_usage_kb remain formula "
        "estimates derived from the parse, not measured execution."
    ),
}

TRAINING_DATA_NOTE = (
    "The ML classifier is trained on synthetically generated, template-"
    "based code snippets labeled by which template bucket produced them -- "
    "not on measured real-world runtime data. Predictions reflect learned "
    "structural patterns (loop depth, nesting, call counts, etc.), not "
    "verified performance benchmarks. Treat the label/confidence as a "
    "heuristic signal, not a guarantee."
)


def _get_profiler() -> CodeProfiler:
    """Load the latest saved model if one exists; otherwise train a fast
    (untuned) one so the server is usable on a cold start without forcing
    callers to wait through a full GridSearchCV sweep."""
    global _profiler, _model_info
    if _profiler is not None:
        return _profiler

    try:
        ml = MLPipeline.load_latest()
        _model_info = {
            "accuracy": ml.accuracy,
            "f1_score": ml.f1,
            "classifier": ml.classifier_type,
            "source": "loaded_existing_model",
        }
    except FileNotFoundError:
        n = config.get("dataset.n_samples", 500)
        ctype = config.get("ml_model.classifier_type", "random_forest")
        df = generate_dataset(n, seed=config.get("dataset.seed", 42))
        ml = MLPipeline(classifier_type=ctype)
        # tune=False on cold start: GridSearchCV across the full grid is
        # too slow for a synchronous first tool call. Run
        # `python profiler_main.py` once beforehand (or call train_model
        # below with tune=True) to get a properly tuned model on disk;
        # this server will pick that up automatically next time.
        results = ml.train(df, tune=False)
        ml.save()
        _model_info = {
            "accuracy": results["test_accuracy"],
            "f1_score": results["test_f1_weighted"],
            "classifier": ctype,
            "source": "trained_untuned_on_cold_start",
        }

    _profiler = CodeProfiler(ml, config)
    return _profiler


def _validate(code: str, language: str) -> str | None:
    """Returns an error string, or None if input is valid."""
    if not code or not code.strip():
        return "Code cannot be empty"
    language = (language or "").strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        return f"Unsupported language '{language}'. Use one of: {', '.join(SUPPORTED_LANGUAGES)}"
    if len(code) > 100_000:
        return "Code exceeds 100KB limit"
    return None


@mcp.tool()
def analyze_code(code: str, language: str = "python") -> dict:
    """
    Analyze a code snippet's structural complexity and get an ML-based
    efficiency prediction (EFFICIENT / MODERATE / NEEDS_OPTIMIZATION)
    with confidence scores and improvement recommendations.

    IMPORTANT ACCURACY NOTES:
    - Python: real AST-based structural analysis; execution_time_ms is a
      formula estimate by default, or real measured subprocess execution
      time if profiling.enable_execution is set in config.yaml (check the
      measured_execution field in the response to see which applied).
      C++/Java: real tree-sitter parsing (not text-pattern matching) --
      code is never compiled or executed for these two languages. Call
      get_analysis_info() for the full breakdown.
    - The ML model is trained on synthetic template-generated data, not
      measured runtime benchmarks. Use the result as a structural
      complexity signal, not a verified performance measurement.

    Args:
        code: source code to analyze (max 100,000 characters).
        language: one of "python", "cpp", "java". Defaults to "python".

    Returns:
        On success: {"metrics": {...}, "ml_prediction": {"label", 
        "confidence", "probabilities"}, "recommendations": [...]}.
        On failure: {"error": "<reason>"}.
    """
    err = _validate(code, language)
    if err:
        return {"error": err}
    profiler = _get_profiler()
    return profiler.analyze(code.strip(), language.strip().lower())


@mcp.tool()
def batch_analyze_code(snippets: list[dict]) -> dict:
    """
    Analyze multiple code snippets in a single call. Useful for scanning
    several files/functions at once (e.g. all changed functions in a PR).

    Same accuracy caveats as analyze_code apply to every snippet here --
    call get_analysis_info() for details.

    Args:
        snippets: list of up to 50 dicts, each shaped like
            {"code": "...", "language": "python"}.

    Returns:
        {"total": int, "successful": int, "failed": int,
         "results": [...], "errors": [{"index", "error"}]}
    """
    if not snippets:
        return {"error": "No snippets provided"}
    if len(snippets) > 50:
        return {"error": "Maximum 50 snippets per batch"}

    profiler = _get_profiler()
    results, errors = [], []
    for i, snippet in enumerate(snippets):
        code = (snippet.get("code") or "").strip()
        language = (snippet.get("language") or "python").strip().lower()
        err = _validate(code, language)
        if err:
            errors.append({"index": i, "error": err})
            continue
        result = profiler.analyze(code, language)
        if "error" in result:
            errors.append({"index": i, "error": result["error"]})
        else:
            result["index"] = i
            results.append(result)

    return {
        "total": len(snippets),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@mcp.tool()
def get_analysis_info() -> dict:
    """
    Return metadata about the currently loaded ML model AND an explicit
    breakdown of what each language's analysis actually measures, so a
    caller can decide how much to trust a given prediction before acting
    on it (e.g. before recommending a refactor to a user).

    Returns:
        {"model_type", "accuracy", "f1_score", "features", "classes",
         "analysis_method": {per-language description},
         "training_data": "<disclosure string>"}
    """
    _get_profiler()
    return {
        "model_type": _model_info.get("classifier", "unknown"),
        "model_accuracy": _model_info.get("accuracy", 0),
        "model_f1_score": _model_info.get("f1_score", 0),
        "model_source": _model_info.get("source", "unknown"),
        "features": FEATURE_COLUMNS,
        "classes": COMPLEXITY_LABELS,
        "analysis_method": ANALYSIS_METHOD_NOTE,
        "training_data": TRAINING_DATA_NOTE,
        "python_execution_sandbox_enabled": config.get("profiling.enable_execution", False),
    }


if __name__ == "__main__":
    mcp.run()
