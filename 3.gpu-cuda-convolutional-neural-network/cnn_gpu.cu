// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...
#define TILE_WIDTH 16

#define TILE_WIDTH_CONV (TILE_WIDTH * 2)
#define SMEM_TILE (TILE_WIDTH_CONV + kKernel - 1)

__global__ void cnn_gpu(float *input, float *weight, float *bias, float *output) {

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int channel = blockIdx.z;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float subTileInput[SMEM_TILE][SMEM_TILE];

    float pool[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // by_input and bx_input are the starting indices of the input tile
    int by_input = by * TILE_WIDTH_CONV;
    int bx_input = bx * TILE_WIDTH_CONV;

    for (int c = 0; c < kNum; c += 1) {
        // Load input and weight into shared memory
        for (int i = ty * TILE_WIDTH + tx; i < SMEM_TILE * SMEM_TILE; i += TILE_WIDTH * TILE_WIDTH) {
            int x_shared = i / (SMEM_TILE * SMEM_TILE); 
            int y_shared = i % (SMEM_TILE * SMEM_TILE) / SMEM_TILE;
            int w_shared = i % (SMEM_TILE * SMEM_TILE) % SMEM_TILE;
                    
            int input_c = c + x_shared;
            int input_y = by_input + y_shared;
            int input_x = bx_input + w_shared;
            
            if (input_c < kNum && input_y < kInImSize && input_x < kInImSize) {
                subTileInput[y_shared][w_shared] = input(input_c, input_y, input_x);
            } else {
                subTileInput[y_shared][w_shared] = 0.0f;
            }
        }

        __syncthreads();

        // convolution layer
            for (int poolRow = 0; poolRow < 2; ++poolRow) {
                for (int poolCol = 0; poolCol < 2; ++poolCol) {
                    float conv_sum = 0.0f;

                    for (int krow = 0; krow < kKernel; ++krow) {
                        for (int kcol = 0; kcol < kKernel; ++kcol) {
                            int input_y = ty * 2 + poolRow + krow;
                            int input_x = tx * 2 + poolCol + kcol;
                            float inputVal = subTileInput[input_y][input_x];
                            conv_sum += inputVal * weight(channel, c, krow, kcol);
                        }
                    }
                    pool[poolRow][poolCol] += conv_sum;
                }
            }
        
        __syncthreads();
    }

    float max_val = INT_MIN;
    for (int poolRow = 0; poolRow < 2; ++poolRow) {
        for (int poolCol = 0; poolCol < 2; ++poolCol) {
            float relu_val = fmaxf(pool[poolRow][poolCol] + bias[channel], 0.0f);
            max_val = fmaxf(max_val, relu_val);
        }
    }
    output(channel, row, col) = max_val;
}