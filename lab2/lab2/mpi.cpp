#define TILE_I 64
#define TILE_J 1024
#define TILE_K 8

// Header inclusions, if any...

#include <mpi.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)
#include <cstring>

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
float c[kI][kJ]) {
    
    // Checkerboard block: gather, scatter, broadcast, reduce

  // Caution: All three matrices are created and initialized ONLY at processor 0 (see lib/main.cpp for
// details). That means your code needs to explicitly send (parts of) matrices a and b from processor
// 0 to the other processors. After the computation, processor 0 should collect the data from all
// processors and store it in matrix c. The correctness and performance will be evaluated at
// processor 0.

    float tileA[TILE_I][TILE_K], tileB[TILE_K][TILE_J];
    int i, j, k, source;

    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int workers = size - 1; // exclude the master process

    if (rank == 0) {
        
        int offset = 0;
        int rows = kI / workers;
        
        // each time a row is read, send it to appropriate process in appropriate process in row of process grid
        // receiving processes will receive the 
        for (int r = 1; r < size; ++r) {
            MPI_Send(&offset, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * kK, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
            MPI_Send(&b, kK * kJ, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        // send matrix a and b to all other processesor 


        // compute 

        // collect the result from all processesor and store it in c
        for (int i = 1; i < size; ++i) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(&c[offset][0], rows * kJ, MPI_FLOAT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }
    else {
        // receive matrix a and b from process 0
        source = 0;

        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&a, rows * kK, MPI_FLOAT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&b, kK * kJ, MPI_FLOAT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // compute 
        for (int k = 0; k < kK; ++k) {
            for (int i = 0; i < rows; ++i) {
                c[i][j] = 0;
                for (int j = 0; j < kJ; ++j) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }
        // send the result to process 0
        MPI_Send(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows * kJ, MPI_FLOAT, source, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    // #pragma omp parallel for private(i, j, k) collapse(2)
    // for (int ii = 0; ii < kI; ii += TILE_I) {
    // for (int jj = 0; jj < kJ; jj += TILE_J) {
    // for (int kk = 0; kk < kK; kk += TILE_K) {
    //     for (int i = 0; i < TILE_I; ++i) {
    //         for (int j = 0; j < TILE_J; ++j) {
    //             float reg = c[ii + i][jj + j];
    //             for (int k = 0; k < TILE_K; ++k) {
    //                 reg += a[ii + i][kk + k] * b[kk + k][jj + j];
    //             }
    //             c[ii + i][jj + j] = reg;
    //         }
    //     }
    // } } }

}
