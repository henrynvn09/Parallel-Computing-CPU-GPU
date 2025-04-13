#include <iostream>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <vector>

using namespace std;

int partition(vector<int>& a, int p, int r) {
    int pivot = a[r];
    int i = (p - 1);
    for (int j = p; j < r; j++) {
        if (a[j] <= pivot) {
            i++;
            swap(a[i], a[j]);
        }
    }
    swap(a[i + 1], a[r]);
    return (i + 1);
}

void quicksort_task(vector<int>& a, int p, int r) {
    if (p < r) {
        int q = partition(a, p, r);
        #pragma omp task shared(a) if(r - p > 1000)
        quicksort_task(a, p, q - 1);
        #pragma omp task shared(a) if(r - p > 1000)
        quicksort_task(a, q + 1, r);
        #pragma omp taskwait
    }
}

void function1(vector<int>& a) {
    // disable OpenMP parallelism for the main thread
    // omp_set_num_threads(1);

    #pragma omp parallel
    {
        quicksort_task(a, 0, a.size() - 1);
    }
}
int main() {
    vector<int> a; // Use vector for dynamic size

    // Read input from "tmp.inp"
    ifstream inp("tmp.inp");
    if (!inp.is_open()) {
        cerr << "Error: Could not open template.inp for reading." << endl;
        return 1;
    }
    int n;
    inp >> n; // Read the size of the array first
    a.resize(n); // Resize the vector to the correct size
    for (int i = 0; i < n; ++i) {
        inp >> a[i]; // Read the array elements
    }
    inp.close();

    function1(a);

    // Write output to "tmp.out"
    ofstream out("tmp.out");
    for (size_t i = 0; i < a.size(); i++) {
        out << a[i] << (i == a.size() - 1 ? "" : " "); // Add space between numbers
    }
    out << endl;
    out.close();

    return 0;
}
