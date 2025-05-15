// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...

__global__ void cnn_gpu(float *input, float *weight, float *bias, float *output) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure the thread is within the bounds of the output
    if (channel >= kNum || h >= kOutImSize || w >= kOutImSize) {
        return;
    }

    float max_pooled_value = INT_MIN; // Initialize to -infinity

    // ph, pw are offsets within the pooling window (0 or 1)
    // [0, 1] x [0, 1] for 2x2 pooling
    for (int ph = 0; ph < 2; ++ph) {
        for (int pw = 0; pw < 2; ++pw) {
            int conv_out_y = h * 2 + ph;
            int conv_out_x = w * 2 + pw;

            float conv_sum = bias[channel];

            for (int ic = 0; ic < kNum; ++ic) {
                // Iterate over the each filter kernel 5x5
                for (int kh = 0; kh < kKernel; ++kh) {
                    for (int kw = 0; kw < kKernel; ++kw) {
                        int input_y = conv_out_y + kh;
                        int input_x = conv_out_x + kw;

                        float input_val = input(ic, input_y, input_x);
                        float weight_val = weight(channel, ic, kh, kw);
                        conv_sum += input_val * weight_val;
                    }
                }
            }

            // Apply ReLU activation
            float relu_val = fmaxf(conv_sum, 0.0f);

            max_pooled_value = fmaxf(max_pooled_value, relu_val);
        }
    }

    output(channel, h, w) = max_pooled_value;
}
