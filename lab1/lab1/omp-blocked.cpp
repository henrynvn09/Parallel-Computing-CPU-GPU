// Header inclusions, if any...

#include "lib/gemm.h"
#include <cstring>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  //TODO: Your code goes here...

  #pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

  int blockSizeI = 64;
  int blockSizeK = 16;
  int blockSizeJ = 128;
  int i, k, j, bi, bk, bj;
  

  #pragma omp parallel for shared(a, b, c) private(i, k, j, bi, bk, bj)
  for (bi = 0; bi < kI; bi += blockSizeI) {
    for (bk = 0; bk < kK; bk += blockSizeK) {
      for (bj = 0; bj < kJ; bj += blockSizeJ) {
        for (i = bi; i < bi + blockSizeI && i < kI; ++i) {
          for (k = bk; k < bk + blockSizeK && k < kK; ++k) {
            for (j = bj; j < bj + blockSizeJ && k < kJ; ++j) {
                c[i][j] += a[i][k] * b[k][j];
              }
            }
        }
      }
    }
  }

}
