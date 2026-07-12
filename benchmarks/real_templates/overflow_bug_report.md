# Bug (FIXED): integer overflow crash in the C++/Java prime-sieve synthetic templates

**Status: FIXED.** The suggested fix at the bottom of this document was applied
to both `CPP_TEMPLATES['MODERATE']` and `JAVA_TEMPLATES['MODERATE']` in
`profiler_main.py` (and synced to `selected_templates.py`'s copies used by
the audit harness below). Verified by compiling and running both fixed
templates directly: correct prime output at n=50 (matches the known primes
up to 50 exactly), correct edge cases at n=0/1/2, and no crash/exception at
n=100,000 -- the exact input size that reproduced the original bug. See
"Fix applied" at the bottom for the exact diff that shipped, which differs
slightly from the originally-suggested one-line diff: bounding the outer
loop naively would have also stopped prime *collection* at sqrt(n),
silently returning wrong results (a worse bug than the crash). The shipped
fix separates sieving (correctly bounded by sqrt(n)) from prime collection
(a full pass over the whole range), mirroring the structure the Python
template already used correctly.

**Found by:** running `real_template_validation.py` against the verbatim
`CPP_TEMPLATES['MODERATE']` / `JAVA_TEMPLATES['MODERATE']` sieve template at
a realistic input size. Not findable by the existing heuristic pipeline,
since it never executes anything it analyzes.

## The bug

Both templates write the outer sieve loop as:

```cpp
for (int i = 2; i <= n; i++) {
    if (sieve[i]) {
        primes.push_back(i);
        for (int j = i * i; j <= n; j += i) sieve[j] = false;
    }
}
```

The outer loop should stop at `i <= sqrt(n)` — there's no need to check
multiples of primes larger than `sqrt(n)`, since any composite `<= n` has a
factor `<= sqrt(n)`. Instead it runs `i` all the way to `n`.

Once `i` exceeds `~46341` (`sqrt(INT_MAX)`), `i * i` overflows a 32-bit
`int`. In C++ this is undefined behavior; in practice it wraps to a
negative number, `j` starts negative, and `sieve[j]` is an out-of-bounds
access:

```
Program received signal SIGSEGV, Segmentation fault.
std::_Bit_reference::operator= (this=0x7fffffffe860, __x=false)
    at /usr/include/c++/13/bits/stl_bvector.h:109
#0  std::_Bit_reference::operator= (...)
#1  0x0000555555555398 in f (n=100000) at test_isolate.cpp:5
#2  0x0000555555555492 in main () at test_isolate.cpp:9
```

Java doesn't have undefined behavior on integer overflow — it wraps
silently and deterministically — but the result is the same class of bug,
surfaced as a catchable exception instead of a crash:

```
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException:
    Index -2146737495 out of bounds for length 100001
    at TestSieve.f(TestSieve.java:9)
```

Confirmed: crashes/throws for `n` roughly above 46,341² ≈ 2.1 billion's
square root threshold — concretely, both reproduced at `n=100,000`.

## Why Python's version doesn't have this bug

The parallel Python template bounds the outer loop correctly:

```python
for i in range(2, int(n**0.5) + 1):
```

This appears to have been written independently per language rather than
ported from a single source — the algorithmic bound is right in one
implementation and wrong in the other two.

## Practical impact

- `generate_dataset()` calls `_generate_program()` with whatever `n` the
  template's own placeholder filling produces; the sieve template's `{n}`
  placeholders (if any were used here — this one has none) aren't the
  issue. The risk is anyone who copies this exact template out of
  `CPP_TEMPLATES`/`JAVA_TEMPLATES` for use beyond synthetic-feature
  generation, or any future change that runs these templates against
  larger inputs.
- It does **not** affect current training, since `_compute_features()`
  never executes the C++/Java template strings — it only runs the
  heuristic text-pattern analysis on the source. This bug would only bite
  if/when real execution is ever wired in (e.g. a future, more complete
  version of the real-measurement approach used here).

## Fix applied

The original one-line suggestion below was insufficient on its own -- naively
changing only the outer loop's bound would make the function stop *collecting
primes* at sqrt(n) too, since the original code interleaved sieving and
collection in one loop. That's a silent correctness regression, not a fix:
the function would still "work" (no crash) but return the wrong answer
(only primes below sqrt(n), not all primes up to n).

The shipped fix splits the two concerns, matching the structure the Python
template already used correctly:

```diff
- std::vector<int> f(int n) { std::vector<bool> sieve(n + 1, true); std::vector<int> primes; for (int i = 2; i <= n; i++) { if (sieve[i]) { primes.push_back(i); for (int j = i * i; j <= n; j += i) sieve[j] = false; } } return primes; }
+ std::vector<int> f(int n) { std::vector<bool> sieve(n + 1, true); for (int i = 2; (long long)i * i <= n; i++) { if (sieve[i]) { for (int j = i * i; j <= n; j += i) sieve[j] = false; } } std::vector<int> primes; for (int i = 2; i <= n; i++) if (sieve[i]) primes.push_back(i); return primes; }
```

(Java: identical shape, `(long) i * i <= n` for the cast, inverted boolean
polarity preserved.) Verified via direct compilation + execution (not
assumed): both fixed templates were compiled and run standalone,
confirming (a) exact match against the known primes up to 50, (b) correct
edge-case behavior at n=0/1/2, and (c) no crash/exception at n=100,000,
the exact size that reproduced the original bug.

**Original suggested fix (kept for historical reference, superseded by the above):**

```diff
- for (int i = 2; i <= n; i++) {
+ for (int i = 2; (long long)i * i <= n; i++) {
```

(or compute `int limit = (int)std::sqrt((double)n);` once and loop
`i <= limit`, mirroring the Python version). Same fix applies to the Java
template.
