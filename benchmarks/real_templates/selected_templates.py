"""
Curated sample of REAL templates from profiler_main.py's
PYTHON_TEMPLATES / CPP_TEMPLATES / JAVA_TEMPLATES, selected to test the
project's complexity-label pipeline against real measured execution --
not new canonical examples.

Each entry: (template_id, language, intended_bucket, source_template,
             placeholder_fills, input_kind, n_values)

input_kind tells the driver how to construct an input of size n:
  "list_int"   -> a list/vector/List<Integer> of n ints
  "string"     -> a string of length n
  "int_direct" -> n is passed directly as the function's argument
"""

PYTHON_SAMPLES = [
    # --- EFFICIENT bucket: builtin-hidden complexity ---
    ("py_set_dedup", "EFFICIENT",
     "def f(lst):\n    return list(set(lst))",
     {}, "list_int", [50_000, 200_000, 800_000]),

    ("py_genexpr_sum", "EFFICIENT",
     "def f(n):\n    return sum(i * i for i in range(n))",
     {}, "int_direct", [200_000, 800_000, 3_200_000]),

    # --- EFFICIENT bucket: the project's own definitional case
    #     (single O(n) pass, explicitly filed as "EFFICIENT" in the
    #     source comment "# --- O(n) single-pass patterns ---") ---
    ("py_single_pass_sum", "EFFICIENT",
     "def f(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total",
     {}, "list_int", [200_000, 800_000, 3_200_000]),

    # --- Relabeled MODERATE -> NEEDS_OPTIMIZATION: bubble sort (textbook O(n^2),
    #     empirically measured exponent ~2.0, exceeds the >1.5 threshold) ---
    ("py_bubble_sort", "NEEDS_OPTIMIZATION",
     "def f(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
     {}, "list_int", [500, 1_000, 2_000]),

    # --- MODERATE bucket: control (sieve, ~O(n log log n)) ---
    ("py_prime_sieve", "MODERATE",
     "def f(n):\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i * i, n + 1, i):\n                sieve[j] = False\n    return [i for i, p in enumerate(sieve) if p]",
     {}, "int_direct", [200_000, 800_000, 3_200_000]),

    # --- NEEDS_OPTIMIZATION bucket: control (visible 4x nested loop) ---
    ("py_quad_nested", "NEEDS_OPTIMIZATION",
     "def f(n):\n    c = 0\n    for i in range(n):\n        for j in range(n):\n            for k in range(n):\n                for l in range(n):\n                    c += 1\n    return c",
     {}, "int_direct", [8, 12, 16]),
]

CPP_SAMPLES = [
    ("cpp_string_reverse", "EFFICIENT",
     "std::string f(const std::string& s) { std::string r = s; std::reverse(r.begin(), r.end()); return r; }\n#include <algorithm>",
     {}, "string", [2_000_000, 8_000_000, 32_000_000]),

    ("cpp_std_count", "EFFICIENT",
     "int f(const std::string& s) { return std::count(s.begin(), s.end(), '{ch}'); }\n#include <algorithm>",
     {"ch": "a"}, "string", [2_000_000, 8_000_000, 32_000_000]),

    ("cpp_single_pass_sum", "EFFICIENT",
     "int f(const std::vector<int>& arr) { int t = 0; for (int x : arr) t += x; return t; }\n#include <vector>",
     {}, "list_int", [10_000_000, 40_000_000, 160_000_000]),

    # --- Relabeled MODERATE -> NEEDS_OPTIMIZATION: bubble sort (measured exponent ~2.0) ---
    ("cpp_bubble_sort", "NEEDS_OPTIMIZATION",
     "void f(std::vector<int>& arr) { for (size_t i = 0; i < arr.size(); i++) for (size_t j = 0; j < arr.size() - i - 1; j++) if (arr[j] > arr[j + 1]) std::swap(arr[j], arr[j + 1]); }\n#include <vector>",
     {}, "list_int", [2_000, 4_000, 8_000]),

    # Sieve overflow bug fixed (see overflow_bug_report.md) -- template now
    # matches the corrected CPP_TEMPLATES entry in profiler_main.py exactly.
    ("cpp_prime_sieve", "MODERATE",
     "std::vector<int> f(int n) { std::vector<bool> sieve(n + 1, true); for (int i = 2; (long long)i * i <= n; i++) { if (sieve[i]) { for (int j = i * i; j <= n; j += i) sieve[j] = false; } } std::vector<int> primes; for (int i = 2; i <= n; i++) if (sieve[i]) primes.push_back(i); return primes; }\n#include <vector>",
     {}, "int_direct", [5_000, 15_000, 40_000]),

    ("cpp_naive_fib", "NEEDS_OPTIMIZATION",
     "int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2); }",
     {}, "int_direct", [30, 35, 38]),

    ("cpp_quad_nested_foldtest", "NEEDS_OPTIMIZATION",
     "int f(int n) { int c = 0; for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++) for (int k = 1; k <= n; k++) for (int l = 1; l <= n; l++) c++; return c; }",
     {}, "int_direct", [10, 20, 40]),
]

JAVA_SAMPLES = [
    ("java_stream_upper", "EFFICIENT",
     "public List<String> f(String[] words) { return Arrays.stream(words).map(String::toUpperCase).collect(Collectors.toList()); }",
     {}, "string_array", [200_000, 800_000, 3_200_000]),

    ("java_hashset_ctor", "EFFICIENT",
     "public Set<Integer> f(List<Integer> arr) { return new HashSet<>(arr); }",
     {}, "list_int", [500_000, 2_000_000, 8_000_000]),

    ("java_single_pass_sum", "EFFICIENT",
     "public int f(List<Integer> arr) { int t = 0; for (int x : arr) t += x; return t; }",
     {}, "list_int", [2_000_000, 8_000_000, 32_000_000]),

    # --- Relabeled MODERATE -> NEEDS_OPTIMIZATION: bubble sort (measured exponent ~2.0) ---
    ("java_bubble_sort", "NEEDS_OPTIMIZATION",
     "public void f(List<Integer> arr) { for (int i = 0; i < arr.size(); i++) for (int j = 0; j < arr.size() - i - 1; j++) if (arr.get(j) > arr.get(j + 1)) Collections.swap(arr, j, j + 1); }",
     {}, "list_int", [2_000, 4_000, 8_000]),

    # Sieve overflow bug fixed (see overflow_bug_report.md) -- template now
    # matches the corrected JAVA_TEMPLATES entry in profiler_main.py exactly.
    ("java_prime_sieve", "MODERATE",
     "public List<Integer> f(int n) { boolean[] sieve = new boolean[n + 1]; for (int i = 2; (long) i * i <= n; i++) { if (!sieve[i]) { for (int j = i * i; j <= n; j += i) sieve[j] = true; } } List<Integer> primes = new ArrayList<>(); for (int i = 2; i <= n; i++) if (!sieve[i]) primes.add(i); return primes; }",
     {}, "int_direct", [5_000, 15_000, 40_000]),

    ("java_naive_fib", "NEEDS_OPTIMIZATION",
     "public int f(int n) { if (n <= 1) return n; return f(n - 1) + f(n - 2); }",
     {}, "int_direct", [30, 35, 38]),
]
