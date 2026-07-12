#include <iostream>
#include <chrono>
#include <cstdlib>
long long f(long long n) { return n % 97 == 0; }
int main(int argc, char** argv) {
    long long n = std::atoll(argv[1]);
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile long long r = f(n);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    std::cout << "ELAPSED_MS=" << ms.count() << " RESULT=" << r << std::endl;
    return 0;
}
