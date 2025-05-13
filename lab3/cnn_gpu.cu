// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...
__global__ void cnn_gpu(float *input, float *weight, float *bias, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < kNum && h < kImSize && w < kImSize) {
        // Initialize output with bias
        float reg = bias[i];

        // Convolution operation
        for (int j = 0; j < kNum; ++j) {
            for (int p = 0; p < kKernel; ++p) {
                for (int q = 0; q < kKernel; ++q) {
                    int w_idx = i * kNum * kKernel * kKernel + j * kKernel * kKernel + p * kKernel + q;
                    int h_idx = j * kImSize * kImSize + (h + p) * kImSize + (w + q);
                    reg += weight[w_idx] * input[h_idx];
                }
            }
        }

        // ReLU activation
        reg = fmaxf(0.f, reg);

        __threadfence();

        if (h % 2 == 0 && w % 2 == 0 && h/2 < kOutImSize && w/2 < kOutImSize) {
            // Compute output index
            int out_h = h/2;
            int out_w = w/2;
            int out_idx = i*(kOutImSize*kOutImSize) + out_h*kOutImSize + out_w;
            
            // Get indices for the 2x2 block
            int c_idx_00 = i*(kImSize*kImSize) + h*kImSize + w;
            int c_idx_01 = i*(kImSize*kImSize) + h*kImSize + (w+1);
            int c_idx_10 = i*(kImSize*kImSize) + (h+1)*kImSize + w;
            int c_idx_11 = i*(kImSize*kImSize) + (h+1)*kImSize + (w+1);
            
            // Compute max
            output[out_idx] = max(
                max(sum, input[c_idx_01]),
                max(input[c_idx_10], input[c_idx_11])
            );
        }
    }

}

