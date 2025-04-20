#include <bits/stdc++.h>
#include <chrono> // Include chrono for timing

using namespace std;
using namespace std::chrono; // Use chrono namespace

// Program name
const string NAME = "./tmp";
const string ext = "";
// Number of test cases
const int NTEST = 1;

mt19937_64 rd(chrono::steady_clock::now().time_since_epoch().count());
#define rand rd

void generateTestCase(ofstream& inp) {
    // Example: Generate a random array of integers
    int n = rand() % 1000000 + 1; // Random size between 1 and 100
    inp << n << endl; // Write the size of the array

    mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<> dist(1, 1000); // Random numbers between 1 and 1000

    for (int i = 0; i < n; ++i) {
        inp << dist(gen) << (i == n - 1 ? "" : " "); // Write random numbers
    }
    inp << endl;
}

// Re-written random function for convenience.
// This random function generates a number within the long long range.
// The generated number is within [L;R].
long long Rand(long long L, long long R) {
    assert(L <= R);
    return L + rd() % (R - L + 1);
}

int main() {
    srand(time(NULL));
    for (int iTest = 1; iTest <= NTEST; iTest++) {
        ofstream inp((NAME + ".inp").c_str());
        // Code for generating test cases here
        generateTestCase(inp); // Generate the test case
        inp.close();

        // Time the execution of the optimized code
        auto start_opt = high_resolution_clock::now();
        system((NAME + ext).c_str());
        auto stop_opt = high_resolution_clock::now();
        auto duration_opt = duration_cast<microseconds>(stop_opt - start_opt);

        // Time the execution of the brute-force code
        auto start_brute = high_resolution_clock::now();
        system((NAME + "_slow" + ext).c_str());
        auto stop_brute = high_resolution_clock::now();
        auto duration_brute = duration_cast<microseconds>(stop_brute - start_brute);

        // Compare the output files
        if (system(("fc " + NAME + ".out " + NAME + ".ans").c_str()) != 0) {
            cout << "Test " << iTest << ": WRONG!\n";
            return 0;
        }

        // Print the results and runtime
        cout << "Test " << iTest << ": CORRECT!\n";
        cout << "  " << NAME << " runtime: " << duration_opt.count() << " microseconds\n";
        cout << "  " << NAME << "_slow runtime: " << duration_brute.count() << " microseconds\n";
    }
    return 0;
}