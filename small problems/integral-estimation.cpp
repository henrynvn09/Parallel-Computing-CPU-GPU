#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib> // For rand and srand
#include <ctime>   // For time
#include <chrono>
#include <functional>
#include <cmath>

using namespace std;
using namespace std::chrono;


// use Riemann Sum to estimate the integral of f(x) = sqrt(x)/(1+x^3)
//  over the interval [0, 1]
double computeIntegral(int n) {
    const auto f = [](double x) {
        return sqrt(x) / (1 + x * x * x);
    };

    double sum = 0, dx = 1.0 / n;

    for (int i = 0; i <= n; ++i) {
        double x = i * dx; // Midpoint
        sum += f(x) * dx;
    }

    return sum;
}

double computeIntegralParallel(int n) {
    double sum = 0, dx = 1.0 / n;

    #pragma omp parallel for reduction(+:sum) num_threads(16)
    for (int i = 0; i <= n; ++i) {
        double x = i * dx;
        sum += dx * sqrt(x) / (1 + x * x * x);
    }

    return sum;
}

double computeIntegralParallel_2(int n) {
    double sum = 0, dx = 1.0 / n;
    double tmpSums[16] = {0};

    #pragma omp parallel num_threads(16)
    {
        int threadNum = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i <= n; ++i) {
            double x = i * dx;
            tmpSums[threadNum] += dx * sqrt(x) / (1 + x * x * x);
        }
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 16; ++i) {
        sum += tmpSums[i];
    }
    return sum;
}


int main(int argc, char* argv[]) {
    // default n for parallel version
    int n = 16000000;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    // Time serial version
    auto t1 = high_resolution_clock::now();
    double serialResult = computeIntegralParallel_2(n);
    auto t2 = high_resolution_clock::now();
    double serialTime = duration_cast<duration<double>>(t2 - t1).count();

    cout << "computeIntegralParallel_1() = " << serialResult
         << "  time = " << serialTime << " s\n";

    // Time parallel version
    auto t3 = high_resolution_clock::now();
    double parallelResult = computeIntegralParallel(n);
    auto t4 = high_resolution_clock::now();
    double parallelTime = duration_cast<duration<double>>(t4 - t3).count();

    cout << "computeIntegralParallel_2(" << n << ") = " << parallelResult
         << "  time = " << parallelTime << " s\n";

    return 0;
}