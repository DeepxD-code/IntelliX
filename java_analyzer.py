"""
Real, parser-based structural analyzer for Java, replacing the previous
text-pattern heuristic (substring counting of 'for'/'while'/'if' anywhere
in the source, which is even cruder than the old C++ heuristic since it
doesn't even require the keyword to be followed by '(').

Same two upgrades as cpp_analyzer.py, same zero-execution-risk profile:
1. Real parsing via tree-sitter instead of text matching.
2. A known-complexity lookup for common Java stdlib calls that hide real
   O(n)/O(n log n) work behind a method call with no loop keyword in the
   source -- confirmed blind spots from the real-template audit:
   Arrays.stream(...).map(...), new HashSet<>(arr).
"""

from tree_sitter import Language, Parser
import tree_sitter_java as _tsjava

JAVA_LANGUAGE = Language(_tsjava.language())

LOOP_NODE_TYPES = {'for_statement', 'while_statement', 'do_statement', 'enhanced_for_statement'}
CONDITIONAL_NODE_TYPES = {'if_statement', 'switch_expression'}
METHOD_DEF_TYPES = {'method_declaration', 'constructor_declaration'}
CLASS_DEF_TYPES = {'class_declaration', 'interface_declaration'}

# Method names recognized regardless of receiver type, since tree-sitter
# alone (no real type resolution) can't always tell a java.util.List from
# an unrelated class with a method of the same name. This is a known,
# accepted limitation -- still strictly better than substring text
# matching, and scoped to names unlikely to collide in profiled snippets.
KNOWN_COMPLEXITY = {
    'sort': 1.5,                                   # Collections.sort / Arrays.sort
    'stream': 1.0,                                  # opens an O(n) pipeline
    'collect': 1.0,                                  # terminal stream op
    'forEach': 1.0,
    'contains': 1.0, 'indexOf': 1.0,                 # O(n) on List
    'addAll': 1.0, 'removeAll': 1.0, 'retainAll': 1.0,
}

# Constructors that copy an existing collection are O(n); detected
# separately since they're object_creation_expression nodes, not calls.
COPY_CONSTRUCTOR_TYPES = {'HashSet', 'ArrayList', 'LinkedList', 'TreeSet', 'HashMap', 'TreeMap'}


def _walk(node):
    """Yield node and all descendants, depth-first."""
    yield node
    for c in node.children:
        yield from _walk(c)


def _analyze_java_recursion_shape(method_node) -> int:
    """Java port of the recursion-shape detector (see profiler_main.py's
    _analyze_recursion_shape for the full design rationale, and
    cpp_analyzer.py's _analyze_cpp_recursion_shape for the C++ port).
    Same two shrinking patterns as the C++ version -- division by a constant
    >=2 (directly or through a declared local variable), and field access
    into a null-checked reference (tree/linked-structure recursion). No
    comprehension-filter pattern to port here either -- Java has no
    comprehension syntax, and none of this codebase's Java templates use a
    stream-filter divide step in recursive position.
    """
    name_node = method_node.child_by_field_name('name')
    if name_node is None:
        return 0
    fname = name_node.text.decode()

    shrink_vars = set()
    for n in _walk(method_node):
        if n.type == 'variable_declarator':
            vn = n.child_by_field_name('name')
            vv = n.child_by_field_name('value')
            if vn is None or vv is None:
                continue
            if vv.type == 'binary_expression':
                op = vv.child_by_field_name('operator')
                right = vv.child_by_field_name('right')
                if op is not None and op.text.decode() == '/' and right is not None and right.type == 'decimal_integer_literal':
                    try:
                        if float(right.text.decode()) >= 2:
                            shrink_vars.add(vn.text.decode())
                    except ValueError:
                        pass

    def is_null_check(if_node) -> bool:
        cond = if_node.child_by_field_name('condition')
        if cond is None:
            return False
        return any(n.type == 'null_literal' for n in _walk(cond))

    has_null_basecase = any(n.type == 'if_statement' and is_null_check(n) for n in _walk(method_node))

    def arg_is_shrinking(arg) -> bool:
        if arg.type == 'binary_expression':
            op = arg.child_by_field_name('operator')
            right = arg.child_by_field_name('right')
            if op is not None and op.text.decode() == '/' and right is not None and right.type == 'decimal_integer_literal':
                try:
                    if float(right.text.decode()) >= 2:
                        return True
                except ValueError:
                    pass
        if arg.type == 'identifier' and arg.text.decode() in shrink_vars:
            return True
        if arg.type == 'field_access' and has_null_basecase:
            return True
        return False

    self_calls = []
    for n in _walk(method_node):
        if n.type == 'method_invocation':
            name_field = n.child_by_field_name('name')
            obj_field = n.child_by_field_name('object')
            # Only unqualified self-calls (bare f(...), not obj.f(...)) --
            # matches the project's existing "bare name only" convention.
            if name_field is not None and obj_field is None and name_field.text.decode() == fname:
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


class JavaTreeSitterAnalyzer:
    """Walks a real Java parse tree. Mirrors PythonASTAnalyzer's metrics
    shape so the rest of the pipeline doesn't need to change."""

    def __init__(self):
        self.metrics = {
            'loops': 0, 'function_calls': 0, 'conditionals': 0,
            'max_nesting_depth': 0, 'function_defs': 0, 'class_defs': 0,
            'stdlib_complexity_weight': 0.0, 'recursive_branching_risk': 0,
        }
        self._current_depth = 0
        self._parser = Parser(JAVA_LANGUAGE)

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

        elif node_type in METHOD_DEF_TYPES:
            self.metrics['function_defs'] += 1
            risk = _analyze_java_recursion_shape(node)
            self.metrics['recursive_branching_risk'] = max(self.metrics['recursive_branching_risk'], risk)

        elif node_type in CLASS_DEF_TYPES:
            self.metrics['class_defs'] += 1

        elif node_type == 'method_invocation':
            self.metrics['function_calls'] += 1
            name_node = node.child_by_field_name('name')
            obj_node = node.child_by_field_name('object')
            # Only weight the chain-INITIATING call, not every recognized
            # name in a chain -- arr.stream().map(...).collect(...) is one
            # O(n) pipeline, not three. A call is chain-initiating if its
            # receiver isn't itself a method call.
            is_chain_initiating = obj_node is None or obj_node.type != 'method_invocation'
            if name_node is not None and is_chain_initiating:
                name = name_node.text.decode()
                if name in KNOWN_COMPLEXITY:
                    self.metrics['stdlib_complexity_weight'] += KNOWN_COMPLEXITY[name]

        elif node_type == 'object_creation_expression':
            type_node = node.child_by_field_name('type')
            args_node = node.child_by_field_name('arguments')
            if type_node is not None and args_node is not None:
                type_name = type_node.text.decode().split('<')[0].strip()
                # A copy-constructor call has exactly one argument and the
                # type is one of the known collection types -- e.g.
                # `new HashSet<>(arr)`. An empty `new HashSet<>()` has no
                # arguments and is O(1), correctly not flagged.
                has_arg = any(c.type not in ('(', ')', ',') for c in args_node.children)
                if type_name in COPY_CONSTRUCTOR_TYPES and has_arg:
                    self.metrics['stdlib_complexity_weight'] += 1.0

        for c in node.children:
            self._visit(c)
