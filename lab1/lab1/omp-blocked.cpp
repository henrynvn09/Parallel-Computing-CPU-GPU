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
  int blockSizeK = 64;
  int blockSizeJ = 64;
  int i, k, j, bi, bk, bj;  

  #pragma omp parallel for shared(a, b, c) private(i, k, j, bi, bk, bj)
  for (bi = 0; bi < kI; bi += blockSizeI) {
    for (bk = 0; bk < kK; bk += blockSizeK) {
      const int iMax = std::min(bi + blockSizeI, kI);
      const int kMax = std::min(bk + blockSizeK, kK);
      for (bj = 0; bj < kJ; bj += blockSizeJ) {
        for (i = bi; i < iMax; ++i) {
          for (k = bk; k < kMax; ++k) {
            for (j = bj; j < std::min(bj + blockSizeJ, kJ); ++j) {
              c[i][j] += a[i][k] * b[k][j];
            }
          }
        }
      }
    }

}
                         }