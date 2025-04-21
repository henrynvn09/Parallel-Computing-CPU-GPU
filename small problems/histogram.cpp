#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib> // For rand and srand
#include <ctime>   // For time
#include <chrono>
#include <functional>

using namespace std;
using namespace std::chrono;

void histogram(int *a, int n, int *h, int m) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        h[i] = 0;
    }

    #pragma omp parallel
    {
        int *local_hist = new int[m];

        #pragma omp for
        for (int i = 0; i < m; ++i) {
            local_hist[i] = 0;
        }
        
        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            local_hist[a[i]]++;
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < m; ++i) {
                h[i] += local_hist[i];
            }
        }
    }
}

// Sequential histogram implementation
void histogram_seq(int *a, int n, int *h, int m) {
    for (int i = 0; i < m; ++i) {
        h[i] = 0;
    }
    for (int i = 0; i < n; ++i) {
        h[a[i]] += 1;
    }
}

// Benchmarking function
template<typename Func>
long long benchmark(Func&& func, const string& name) {
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << name << " Runtime: " << duration.count() << " microseconds" << endl;
    return duration.count();
}

int main() {
    // Define the size of the input array and the number of bins in the histogram
    int n = 10000000;
    int m = 100000;

    // Create the input array 'a' with random values between 0 and m-1
    std::vector<int> a(n);
    std::srand(std::time(nullptr)); // Seed the random number generator
    for (int i = 0; i < n; ++i) {
        a[i] = std::rand() % m;
    }

    // Create the histogram arrays 'h_parallel' and 'h_sequential'
    std::vector<int> h_parallel(m);
    std::vector<int> h_sequential(m);

    // Benchmark parallel histogram
    benchmark([&]() {
        histogram(a.data(), n, h_parallel.data(), m);
    }, "Parallel Histogram");

    // Benchmark sequential histogram
    benchmark([&]() {
        histogram_seq(a.data(), n, h_sequential.data(), m);
    }, "Sequential Histogram");

    // Verify correctness
    bool is_correct = true;
    for (int i = 0; i < m; ++i) {
        if (h_parallel[i] != h_sequential[i]) {
            is_correct = false;
            break;
        }
    }

    cout << "Correctness check: " << (is_correct ? "Passed" : "Failed") << endl;

    return 0;
}
