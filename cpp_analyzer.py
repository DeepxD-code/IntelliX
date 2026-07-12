"""
Real, parser-based structural analyzer for C++, replacing the previous
text-pattern heuristic (keyword/brace counting on the raw source string).

Two distinct upgrades bundled here, both with zero code-execution risk:

1. REAL PARSING instead of text matching. The old heuristic counted
   substring occurrences of 'for (', 'while (', etc. directly on the
   source text -- which breaks on anything inside a string literal or
   comment, on macro-expanded loops, and can't reliably tell a function
   call's name from surrounding noise. tree-sitter gives an actual
   concrete syntax tree, the same category of analysis Python's AST
   module already provides for Python.

2. A KNOWN-COMPLEXITY LOOKUP for common standard-library calls. This is
   the part that actually closes the specific blind spots found in
   benchmarks/real_templates/real_template_validation.py: std::reverse
   and std::count are real O(n) work with no literal 'for'/'while' in
   the source. A real parser correctly identifies them as call_expression
   nodes -- but knowing "this is a function call" isn't enough; the old
   heuristic already roughly counted calls and still missed them, because
   it weighted every call equally low. What's needed is recognizing WHICH
   calls are known to do O(n)/O(n log n) work internally, and weighting
   them accordingly. That's what KNOWN_COMPLEXITY does.

This does NOT execute or compile anything. It's static analysis only.
"""

from tree_sitter import Language, Parser
import tree_sitter_cpp as _tscpp

CPP_LANGUAGE = Language(_tscpp.language())

LOOP_NODE_TYPES = {'for_statement', 'while_statement', 'do_statement', 'for_range_loop'}
CONDITIONAL_NODE_TYPES = {'if_statement', 'switch_statement'}
FUNCTION_DEF_TYPES = {'function_definition'}
CLASS_DEF_TYPES = {'class_specifier', 'struct_specifier'}

# Weight is in "loop-equivalents" -- calibrated to combine sensibly with
# the existing complexity formula's loop_count term. Confirmed against
# the real-template audit: std::reverse and std::count were the two
# specific cases found to be invisible to the old heuristic.
KNOWN_COMPLEXITY = {
    # O(n log n)
    'sort': 1.5, 'stable_sort': 1.5, 'partial_sort': 1.5,
    # O(n)
    'reverse': 1.0, 'count': 1.0, 'count_if': 1.0,
    'find': 1.0, 'find_if': 1.0, 'accumulate': 1.0,
    'copy': 1.0, 'fill': 1.0, 'transform': 1.0,
    'unique': 1.0, 'remove': 1.0, 'remove_if': 1.0,
    'max_element': 1.0, 'min_element': 1.0,
    'all_of': 1.0, 'any_of': 1.0, 'none_of': 1.0, 'for_each': 1.0,
    # O(log n) -- small but non-zero
    'lower_bound': 0.3, 'upper_bound': 0.3, 'binary_search': 0.3,
}


def _walk(node):
    """Yield node and all descendants, depth-first -- tree-sitter's Python
    bindings don't provide a built-in equivalent of ast.walk()."""
    yield node
    for c in node.children:
        yield from _walk(c)


def _function_name(func_def_node) -> str | None:
    """Extract a function_definition node's own name via nested declarator
    field navigation (function_definition -> declarator[function_declarator]
    -> declarator[identifier])."""
    declarator = func_def_node.child_by_field_name('declarator')
    if declarator is None:
        return None
    # Unwrap pointer/reference declarators if present (e.g. `int* f(...)`) --
    # keep walking the 'declarator' field until we hit the identifier.
    while declarator is not None and declarator.type != 'function_declarator':
        declarator = declarator.child_by_field_name('declarator')
    if declarator is None:
        return None
    name_node = declarator.child_by_field_name('declarator')
    return name_node.text.decode() if name_node is not None else None


def _is_null_check(if_node) -> bool:
    """Does this if_statement's condition compare something to nullptr/NULL?
    Matches the standard base-case guard for tree/linked-structure recursion."""
    cond = if_node.child_by_field_name('condition')
    if cond is None:
        return False
    for n in _walk(cond):
        if n.type == 'null':  # nullptr
            return True
        if n.type == 'identifier' and n.text.decode() == 'NULL':
            return True
    return False


def _analyze_cpp_recursion_shape(func_def_node) -> int:
    """C++ port of the Python recursion-shape detector (see profiler_main.py's
    _analyze_recursion_shape for the full design rationale). Same three
    shrinking patterns, adapted to tree-sitter field names: division by a
    constant >=2 (directly or through a declared variable), and field-access
    into a null-checked pointer/reference (tree/linked-structure recursion).
    C++ has no comprehension-filter equivalent to check for -- none of this
    codebase's C++ templates use a filter-based divide step, so that pattern
    is intentionally not ported; only the two that are actually needed here.
    """
    fname = _function_name(func_def_node)
    if fname is None:
        return 0

    shrink_vars = set()
    for n in _walk(func_def_node):
        if n.type == 'init_declarator':
            name_node = n.child_by_field_name('declarator')
            value_node = n.child_by_field_name('value')
            if name_node is None or value_node is None or name_node.type != 'identifier':
                continue
            if value_node.type == 'binary_expression':
                op = value_node.child_by_field_name('operator')
                right = value_node.child_by_field_name('right')
                if op is not None and op.text.decode() == '/' and right is not None and right.type == 'number_literal':
                    try:
                        if float(right.text.decode()) >= 2:
                            shrink_vars.add(name_node.text.decode())
                    except ValueError:
                        pass

    has_null_basecase = any(n.type == 'if_statement' and _is_null_check(n) for n in _walk(func_def_node))

    def arg_is_shrinking(arg) -> bool:
        if arg.type == 'binary_expression':
            op = arg.child_by_field_name('operator')
            right = arg.child_by_field_name('right')
            if op is not None and op.text.decode() == '/' and right is not None and right.type == 'number_literal':
                try:
                    if float(right.text.decode()) >= 2:
                        return True
                except ValueError:
                    pass
        if arg.type == 'identifier' and arg.text.decode() in shrink_vars:
            return True
        if arg.type == 'field_expression' and has_null_basecase:
            return True
        return False

    self_calls = []
    for n in _walk(func_def_node):
        if n.type == 'call_expression':
            func_node = n.child_by_field_name('function')
            if func_node is not None and func_node.type == 'identifier' and func_node.text.decode() == fname:
                self_calls.append(n)

    if len(self_calls) < 2:
        return 0

    any_shrinking = False
    for call in self_calls:
        args_node = call.child_by_field_name('arguments')
        if args_node is None:
            continue
        for a in args_node.children:
            if a.type in ('(', ')', ','):
                continue
            if arg_is_shrinking(a):
                any_shrinking = True
                break
        if any_shrinking:
            break
    return 0 if any_shrinking else len(self_calls)


def _rightmost_name(node) -> str | None:
    """Extract the simple (rightmost) identifier from a call's function
    subtree, e.g. 'reverse' from 'std::reverse', 'begin' from 'r.begin'.
    Depth-first traversal naturally visits the rightmost leaf last."""
    name = None
    stack = [node]
    # iterative post-order-ish walk; just need the last identifier seen
    def walk(n):
        nonlocal name
        if n.type in ('identifier', 'field_identifier'):
            name = n.text.decode()
        for c in n.children:
            walk(c)
    walk(node)
    return name


class CppTreeSitterAnalyzer:
    """Walks a real C++ parse tree. Mirrors PythonASTAnalyzer's metrics
    shape so the rest of the pipeline doesn't need to change."""

    def __init__(self):
        self.metrics = {
            'loops': 0, 'function_calls': 0, 'conditionals': 0,
            'max_nesting_depth': 0, 'function_defs': 0, 'class_defs': 0,
            'stdlib_complexity_weight': 0.0, 'recursive_branching_risk': 0,
        }
        self._current_depth = 0
        self._parser = Parser(CPP_LANGUAGE)

    def analyze(self, code: str) -> dict:
        tree = self._parser.parse(code.encode('utf-8', errors='replace'))
        self._visit(tree.root_node)
        return self.metrics

    def _visit(self, node):
        node_type = node.type

        if node_type in LOOP_NODE_TYPES:
            self.metrics['loops'] += 1
            self._current_depth += 1
            self.metrics['max_nesting_depth'] = max(
                self.metrics['max_nesting_depth'], self._current_depth)
            for c in node.children:
                self._visit(c)
            self._current_depth -= 1
            return

        if node_type in CONDITIONAL_NODE_TYPES:
            self.metrics['conditionals'] += 1

        elif node_type in FUNCTION_DEF_TYPES:
            self.metrics['function_defs'] += 1
            risk = _analyze_cpp_recursion_shape(node)
            self.metrics['recursive_branching_risk'] = max(self.metrics['recursive_branching_risk'], risk)

        elif node_type in CLASS_DEF_TYPES:
            self.metrics['class_defs'] += 1

        elif node_type == 'call_expression':
            self.metrics['function_calls'] += 1
            func_node = node.child_by_field_name('function')
            if func_node is not None:
                name = _rightmost_name(func_node)
                if name in KNOWN_COMPLEXITY:
                    self.metrics['stdlib_complexity_weight'] += KNOWN_COMPLEXITY[name]

        for c in node.children:
            self._visit(c)
