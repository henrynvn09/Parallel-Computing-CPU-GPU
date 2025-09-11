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

    #pragma omp parallel for private(tileA, tileB, i, j, k) num_threads(4)
    for (int ib = 0; ib < kI; ib += TILE_I)
    {
        for (int jb = 0; jb < kJ; jb += TILE_J) {
            float tileC[TILE_I][TILE_J] = {0};

            for (int kb = 0; kb < kK; kb += TILE_K) {
                for (i = 0; i < TILE_I; ++i) {
                    tileA[i][0] = a[ib + i][kb];
                    tileA[i][1] = a[ib + i][kb + 1];
                    tileA[i][2] = a[ib + i][kb + 2];
                    tileA[i][3] = a[ib + i][kb + 3];
                    tileA[i][4] = a[ib + i][kb + 4];
                    tileA[i][5] = a[ib + i][kb + 5];
                    tileA[i][6] = a[ib + i][kb + 6];
                    tileA[i][7] = a[ib + i][kb + 7];
                }

                for (k = 0; k < TILE_K; ++k) {
                    for (j = 0; j < TILE_J; j+= 8) {
                        tileB[k][j] = b[kb + k][jb + j];
                        tileB[k][j + 1] = b[kb + k][jb + j + 1];
                        tileB[k][j + 2] = b[kb + k][jb + j + 2];
                        tileB[k][j + 3] = b[kb + k][jb + j + 3];
                        tileB[k][j + 4] = b[kb + k][jb + j + 4];
                        tileB[k][j + 5] = b[kb + k][jb + j + 5];
                        tileB[k][j + 6] = b[kb + k][jb + j + 6];
                        tileB[k][j + 7] = b[kb + k][jb + j + 7];
                    }
                }

                for (i = 0; i < TILE_I; ++i) {
                    for (j = 0; j < TILE_J; ++j) {
                        tileC[i][j] += tileA[i][0] * tileB[0][j];
                        tileC[i][j] += tileA[i][1] * tileB[1][j];
                        tileC[i][j] += tileA[i][2] * tileB[2][j];
                        tileC[i][j] += tileA[i][3] * tileB[3][j];
                        tileC[i][j] += tileA[i][4] * tileB[4][j];
                        tileC[i][j] += tileA[i][5] * tileB[5][j];
                        tileC[i][j] += tileA[i][6] * tileB[6][j];
                        tileC[i][j] += tileA[i][7] * tileB[7][j];
                    }
                }
            }

            for (i = 0; i < TILE_I; ++i) {
                for (j = 0; j < TILE_J; j+= 8) {
                    c[ib + i][jb + j] += tileC[i][j];
                    c[ib + i][jb + j + 1] += tileC[i][j + 1];
                    c[ib + i][jb + j + 2] += tileC[i][j + 2];
                    c[ib + i][jb + j + 3] += tileC[i][j + 3];
                    c[ib + i][jb + j + 4] += tileC[i][j + 4];
                    c[ib + i][jb + j + 5] += tileC[i][j + 5];
                    c[ib + i][jb + j + 6] += tileC[i][j + 6];
                    c[ib + i][jb + j + 7] += tileC[i][j + 7];
                }
            }
        }
    }
}
