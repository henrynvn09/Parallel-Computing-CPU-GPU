#include <iostream>
#include <algorithm>
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

void quicksort(vector<int>& a, int p, int r) {
    if (p < r) {
        int q = partition(a, p, r);
        quicksort(a, p, q - 1);
        quicksort(a, q + 1, r);
    }
}

void function1(vector<int>& a) {
    quicksort(a, 0, a.size() - 1); // No OpenMP
}

int main() {
    vector<int> a; // Use vector for dynamic size

    // Read input from "template.inp"
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

    // Write output to "template.out"
    ofstream out("tmp.ans");
    for (size_t i = 0; i < a.size(); i++) {
        out << a[i] << (i == a.size() - 1 ? "" : " "); // Add space between numbers
    }
    out << endl;
    out.close();

    return 0;
}