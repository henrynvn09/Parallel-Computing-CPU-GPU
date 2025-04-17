#define TILE_I 256
#define TILE_J 512
#define TILE_K 8

// Header inclusions, if any...

#include "lib/gemm.h"
#include <cstring>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
    float tileA[TILE_I][TILE_K], tileB[TILE_K][TILE_J];
    int i, j, k;

    #pragma omp parallel for private(i)
    for (i = 0; i < kI; ++i) {
        std::memset(c[i], 0, sizeof(float) * kJ);
    }

    #pragma omp parallel for private(tileA, tileB, i, j, k)
    for (int ib = 0; ib < kI; ib += TILE_I)
    {
        for (int jb = 0; jb < kJ; jb += TILE_J) {
            float tileC[TILE_I][TILE_J] = {0};

            for (int kb = 0; kb < kK; kb += TILE_K) {
                for (i = 0; i < TILE_I; ++i) {
                    for (k = 0; k < TILE_K; ++k) {
                        tileA[i][k] = a[ib + i][kb + k];
                    }
                }

                for (k = 0; k < TILE_K; ++k) {
                    for (j = 0; j < TILE_J; ++j) {
                        tileB[k][j] = b[kb + k][jb + j];
                    }
                }

                for (i = 0; i < TILE_I; ++i) {
                    for (j = 0; j < TILE_J; ++j) {
                        for (k = 0; k < TILE_K; ++k) {
                            tileC[i][j] += tileA[i][k] * tileB[k][j];
                        }
                    }
                }
            }

            for (i = 0; i < TILE_I; ++i) {
                for (j = 0; j < TILE_J; ++j) {
                    c[ib + i][jb + j] += tileC[i][j];
                }
            }
        }
    }
}
