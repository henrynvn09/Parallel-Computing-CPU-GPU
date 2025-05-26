#include <ap_int.h>
#include <cstring>
#include <hls_stream.h>
#include <hls_vector.h>

/******************************************************************************************
These typedefs define *fixed-size vector types* for float values using HLS
library types. hls::vector<T, N> is a type that holds N elements of type T —
similar to std::array<T, N>, but optimized for High-Level Synthesis (HLS) to
allow efficient vector operations and memory transfers.

These types are used to optimize memory bandwidth by allowing wide accesses.
Instead of reading/writing 1 float at a time, we can transfer 4, 8, or 16 floats
in parallel.

Each typedef here defines a shorthand name for a specific vector width:
******************************************************************************************/
typedef hls::vector<float, 16>
    float16; // A vector of 16 float elements (512 bits total)
typedef hls::vector<float, 8>
    float8; // A vector of 8 float elements  (256 bits)
typedef hls::vector<float, 4>
    float4; // A vector of 4 float elements  (128 bits)
typedef hls::vector<float, 2>
    float2; // A vector of 2 float elements   (64 bits)
typedef hls::vector<float, 1>
    float1; // A "vector" of 1 float (just a wrapper — used for consistency)

/**
 * @brief Loads a 3D input tile from external memory (vinput) into a local array
 * (input).
 *
 * This function uses vectorized memory access (float4) to efficiently load data
 * from memory. Each float4 contains 4 float elements, allowing 128-bit wide
 * memory reads. The inner loop is pipelined to achieve high throughput (one
 * access per cycle).
 *
 * @param input   Local buffer to store the unpacked input tile [1][228][228].
 * @param vinput  Flattened input buffer in memory, accessed with float4
 * vectors.
 * @param d0      Depth offset (batch or input channel index).
 */
void load_input_S0(float input[1][228][228], float4 vinput[3326976], int d0) {

  /**
   * Prevents function inlining to give HLS better control over scheduling.
   * This is useful when the function is large or reused multiple times.
   */
#pragma HLS inline off

  /**
   * Iterate over the input dimensions.
   * - i0: input depth (fixed to 1 here)
   * - i1: height dimension
   * - i2: width dimension, step by 4 (because we read float4)
   */
  for (int i0 = 0; i0 < 1; i0 += 1) {
    for (int i1 = 0; i1 < 228; i1 += 1) {
      for (int i2 = 0; i2 < 228; i2 += 4) {

        /**
         * Pipeline the innermost loop with an initiation interval (II) of 1.
         * This enables the function to read one float4 per clock cycle.
         */
#pragma HLS pipeline II = 1

        /**
         * Load a float4 (4 floats) from external memory using a flattened
         * index. The index computation maps the 3D access to 1D array space.
         */
        float4 tmp_input = vinput[(i0 + d0 * 1) * 12996 + i1 * 57 + i2 / 4];

        /**
         * Unpack the float4 into individual scalar floats and store in the
         * local buffer. This makes the data ready for downstream computation.
         */
        input[i0][i1][i2 + 0] = tmp_input[0];
        input[i0][i1][i2 + 1] = tmp_input[1];
        input[i0][i1][i2 + 2] = tmp_input[2];
        input[i0][i1][i2 + 3] = tmp_input[3];
      }
    }
  }
}

/**
 * @brief Loads a 3D output tile from external memory (voutput) into a local
 * array (output).
 *
 * This function reads output feature maps from a memory buffer using vectorized
 * float16 types, which enables 512-bit wide memory access (16 floats per
 * transaction). The output is unpacked and stored into a local array for
 * further processing or accumulation.
 *
 * @param output   Local output buffer to populate [16][224][224].
 * @param voutput  Flattened vectorized output buffer in memory, accessed using
 * float16.
 * @param d0       Depth tile index (e.g., batch index or output channel
 * offset).
 */
void load_output_S0(float output[16][224][224], float16 voutput[802816],
                    int d0) {

#pragma HLS inline off

  /**
   * Iterate over the output tensor dimensions.
   * - i0: output channels (16 per tile)
   * - i1: output height (224)
   * - i2: output width (224), processed in blocks of 16 (float16)
   */
  for (int i0 = 0; i0 < 16; i0 += 1) {
    for (int i1 = 0; i1 < 224; i1 += 1) {
      for (int i2 = 0; i2 < 224; i2 += 16) {

        /**
         * Pipeline the innermost loop with II=1.
         * This allows one float16 to be read per cycle, maximizing throughput.
         */
#pragma HLS pipeline II = 1

        /**
         * Load a vector of 16 floats from the external memory.
         * The index flattens the 3D coordinates into a 1D memory address.
         */
        float16 tmp_output = voutput[(i0 + d0 * 16) * 3136 + i1 * 14 + i2 / 16];

        /**
         * Unpack each float from the vector and assign it to the local 3D
         * output buffer. This prepares the output tile for any further
         * processing or accumulation.
         */
        output[i0][i1][i2 + 0] = tmp_output[0];
        output[i0][i1][i2 + 1] = tmp_output[1];
        output[i0][i1][i2 + 2] = tmp_output[2];
        output[i0][i1][i2 + 3] = tmp_output[3];
        output[i0][i1][i2 + 4] = tmp_output[4];
        output[i0][i1][i2 + 5] = tmp_output[5];
        output[i0][i1][i2 + 6] = tmp_output[6];
        output[i0][i1][i2 + 7] = tmp_output[7];
        output[i0][i1][i2 + 8] = tmp_output[8];
        output[i0][i1][i2 + 9] = tmp_output[9];
        output[i0][i1][i2 + 10] = tmp_output[10];
        output[i0][i1][i2 + 11] = tmp_output[11];
        output[i0][i1][i2 + 12] = tmp_output[12];
        output[i0][i1][i2 + 13] = tmp_output[13];
        output[i0][i1][i2 + 14] = tmp_output[14];
        output[i0][i1][i2 + 15] = tmp_output[15];
      }
    }
  }
}

/**
 * @brief Loads a 4D weight tensor from external memory (vweight) into a local
 * array (weight).
 *
 * Each element is loaded using a float1 vector (i.e., scalar float wrapped for
 * interface consistency). The nested loops traverse the 4D weight structure
 * [16][256][5][5], reading one scalar at a time. This function can be a
 * bottleneck due to scalar reads.
 *
 * @param weight   Local buffer to store the unpacked weights [16][256][5][5].
 * @param vweight  Flattened external buffer containing weights, accessed as
 * float1 (1 float per entry).
 * @param d0       Tile index for the outer output channel dimension (e.g., for
 * batching or tiling). This corresponds to the i0 loop in the cnn function.
 */
void load_weight_S0(float weight[16][256][5][5], float1 vweight[1638400],
                    int d0_tile_offset) { // Renamed d0 for clarity

#pragma HLS inline off

  /**
   * Loop over the full 4D weight tensor for the current tile:
   * - i0_w: output channels within the tile (0-15)
   * - i1_w: input channels (0-255)
   * - i2_w: kernel height (0-4)
   * - i3_w: kernel width  (0-4)
   */
  for (int i0_w = 0; i0_w < 16; i0_w += 1) {     // Corresponds to weight[i0_w]...
    for (int i1_w = 0; i1_w < 256; i1_w += 1) {  // Corresponds to weight[][i1_w]...
      for (int i2_w = 0; i2_w < 5; i2_w += 1) {    // Corresponds to weight[][][i2_w]...
        for (int i3_w = 0; i3_w < 5; i3_w += 1) {  // Corresponds to weight[][][][i3_w]...

          /**
           * Pipeline the innermost loop with an initiation interval of 1.
           * Ensures one float1 is read per clock cycle, maximizing throughput
           * for this scalar interface.
           */
#pragma HLS pipeline II = 1

          /**
           * Read a single weight value from memory.
           * The flattened index computes the 1D offset from the 4D tensor
           * coordinates. d0_tile_offset is used to select the correct
           * 16-channel group of weights from the global vweight buffer.
           */
          float1 tmp_weight =
              vweight[(i0_w + d0_tile_offset * 16) * 6400 + i1_w * 25 + i2_w * 5 + i3_w];

          /**
           * Store the scalar value in the local 4D weight buffer.
           * tmp_weight[0] is used since float1 is a vector of 1 float.
           */
          weight[i0_w][i1_w][i2_w][i3_w] = tmp_weight[0];
        }
      }
    }
  }
}

/**
 * @brief Stores a 3D output tile from a local array into external memory using
 * float16 vectorized access.
 *
 * This function writes computed output feature maps back to memory.
 * It packs 16 scalar float values into a float16 vector and performs one wide
 * memory write (512 bits).
 *
 * @param output   Local buffer containing the computed output [16][224][224].
 * @param voutput  Flattened vectorized output buffer in memory (to be written).
 * @param d0       Depth tile index (e.g., batch index or output channel
 * offset).
 */
void store_output_S0(float output[16][224][224], float16 voutput[802816],
                     int d0) {

#pragma HLS inline off

  /**
   * Iterate over the output tensor dimensions:
   * - i0: output channels (tile of 16)
   * - i1: output height
   * - i2: output width, processed in chunks of 16 (since we write float16)
   */
  for (int i0 = 0; i0 < 16; i0 += 1) {
    for (int i1 = 0; i1 < 224; i1 += 1) {
      for (int i2 = 0; i2 < 224; i2 += 16) {

        /**
         * Pipeline the innermost loop with an initiation interval of 1.
         * This enables one float16 write per clock cycle for high throughput.
         */
#pragma HLS pipeline II = 1

        /**
         * Pack 16 scalar floats from the local buffer into a float16 vector.
         * This prepares the data for a wide memory write (512 bits).
         */
        float16 tmp_output;
        tmp_output[0] = output[i0][i1][i2 + 0];
        tmp_output[1] = output[i0][i1][i2 + 1];
        tmp_output[2] = output[i0][i1][i2 + 2];
        tmp_output[3] = output[i0][i1][i2 + 3];
        tmp_output[4] = output[i0][i1][i2 + 4];
        tmp_output[5] = output[i0][i1][i2 + 5];
        tmp_output[6] = output[i0][i1][i2 + 6];
        tmp_output[7] = output[i0][i1][i2 + 7];
        tmp_output[8] = output[i0][i1][i2 + 8];
        tmp_output[9] = output[i0][i1][i2 + 9];
        tmp_output[10] = output[i0][i1][i2 + 10];
        tmp_output[11] = output[i0][i1][i2 + 11];
        tmp_output[12] = output[i0][i1][i2 + 12];
        tmp_output[13] = output[i0][i1][i2 + 13];
        tmp_output[14] = output[i0][i1][i2 + 14];
        tmp_output[15] = output[i0][i1][i2 + 15];

        /**
         * Write the packed float16 vector to memory.
         * The memory index is flattened from the 3D output coordinates.
         */
        voutput[(i0 + d0 * 16) * 3136 + i1 * 14 + i2 / 16] = tmp_output;
      }
    }
  }
}

/**
 * @brief Top-level CNN function that performs a convolution operation.
 *
 * @param input   Local input image buffer [1][228][228].
 * @param output  Local output feature map buffer [16][224][224]. This buffer
 * is used for accumulation across input channels (j loop) and
 * is loaded/stored per output tile (i0 loop).
 * @param weight  Local weight (kernel) buffer [16][256][5][5].
 * @param vinput  Vectorized input memory buffer used for burst transfers.
 * @param vweight Vectorized weight memory buffer.
 * @param voutput Vectorized output memory buffer.
 */
void cnn(float input[1][228][228], float output[16][224][224],
         float weight[16][256][5][5], float4 vinput[3326976],
         float1 vweight[1638400], float16 voutput[802816]) {

  // i0 loop: Iterates over tiles of 16 output channels.
  // For example, if total output channels = 256, this loop runs 256/16 = 16 times.
  // The problem statement implies voutput is large enough for all output channels,
  // and d0 in load/store acts as an offset.
  // Assuming i0 iterates over these tiles. Let's assume 16 such tiles for illustration,
  // meaning 16*16 = 256 output channels in total.
  // The loop bound `16` for `i0` means we process 16 groups of 16 output channels.
  for (int i0 = 0; i0 < 16; i0++) { // Output channel tile index
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16 // Helps HLS estimate loop latency

    /**
     * Load the weights for the current output tile (16 output channels).
     * The 'i0' parameter to load_weight_S0 acts as a tile offset into vweight.
     * weight[local_out_ch][in_ch][k_h][k_w]
     */
    load_weight_S0(weight, vweight, i0);

    /**
     * Load the partial output tile from external memory into the local 'output'
     * array. This is crucial if results are accumulated across different i0 tiles
     * (e.g. if i0 was part of a batch dimension not shown) or if bias is added
     * by pre-loading. For standard convolution accumulation, this loads existing
     * values to be added upon, or zeros if it's the start.
     * output[local_out_ch][h][w]
     */
    load_output_S0(output, voutput, i0);


    // j loop: Iterates over input channels (0 to 255)
    for (int j = 0; j < 256; j++) {
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256

      /**
       * Load a section of the input corresponding to the current input channel 'j'.
       * Data is loaded from the vectorized memory (vinput) using float4 accesses.
       * The 'j' parameter to load_input_S0 acts as an input channel offset.
       * input[0][h][w]
       */
      load_input_S0(input, vinput, j);

      // i1 loop: Iterates over the 16 output channels within the current tile.
      // THIS LOOP IS FULLY UNROLLED to process 16 output channels in parallel.
      for (int i1 = 0; i1 < 16; i1++) {
#pragma HLS UNROLL factor=8 // Fully unroll for parallel computation of 16 output channels

        // h loop: Iterates over output height (0 to 223)
        for (int h = 0; h < 224; h++) { // Loop bound is 224 (0 to 223)
#pragma HLS LOOP_TRIPCOUNT min=224 max=224 avg=224

          // w loop: Iterates over output width (0 to 223)
          // THIS LOOP IS PIPELINED with II=1.
          // For each (h,w) pixel, we compute 16 output values (due to unrolled i1)
          // by convolving with the 5x5 kernel over the current input channel j.
          for (int w = 0; w < 224; w++) {
#pragma HLS LOOP_TRIPCOUNT min=224 max=224 avg=224
#pragma HLS PIPELINE II=1

            // p loop: Kernel height (0 to 4)
            // THIS LOOP IS FULLY UNROLLED.
            for (int p = 0; p < 5; p++) {
#pragma HLS UNROLL

              // q loop: Kernel width (0 to 4)
              // THIS LOOP IS FULLY UNROLLED.
              for (int q = 0; q < 5; q++) {
#pragma HLS UNROLL

                // Core convolution: output += weight * input
                // Accesses:
                //   output[i1][h][w] - `i1` is the unrolled loop, refers to local output channel
                //   weight[i1][j][p][q] - `i1` is unrolled local output ch, `j` is current input ch
                //   input[0][h+p][w+q] - `0` as input depth is 1 for this tile
                //
                // The `i0` (tile index) context is handled by which tile of weights
                // and initial outputs were loaded. `i1` is the index within that tile.
                output[i1][h][w] +=
                    weight[i1][j][p][q] * input[0][h + p][w + q];
              }
            }
          }
        }
      }
    } // End of j loop (input channels)

    /**
     * After processing all input channels (j loop) for the current output tile (i0),
     * store the computed output tile back to external memory.
     * Data is packed and written using float16 vectorized writes.
     */
    store_output_S0(output, voutput, i0);

  } // End of i0 loop (output channel tiles)
}

/**
 * @brief Top-level HLS kernel for CNN execution.
 *
 * This function is synthesized as the hardware entry point.
 * It connects memory-mapped AXI interfaces for input/output/weights
 * and invokes the main CNN function on local buffers.
 *
 * @param vinput   External vectorized input buffer (float4 format).
 * @param vweight  External vectorized weight buffer (float1 format).
 * @param voutput  External vectorized output buffer (float16 format).
 */
void kernel_cnn(float4 vinput[3326976], float1 vweight[1638400],
                float16 voutput[802816]) {

  /**
   * Define memory-mapped AXI4 interfaces for data transfer.
   */
#pragma HLS INTERFACE m_axi port = vinput offset = slave bundle = kernel_input
#pragma HLS INTERFACE m_axi port = voutput offset = slave bundle = kernel_output
#pragma HLS INTERFACE m_axi port = vweight offset = slave bundle = kernel_weight

  /**
   * Define control interface for the kernel.
   */
#pragma HLS INTERFACE s_axilite port=return bundle=control // For kernel completion signal

  /**
   * Local buffers for input, weights, and output feature maps.
   * These buffers are stored in on-chip BRAM/URAM for fast access.
   */
  static float input[1][228][228];   // Made static to ensure BRAM implementation
  static float output[16][224][224]; // Made static
  static float weight[16][256][5][5]; // Made static

  /**
   * Apply array partitioning to allow concurrent access to different array
   * dimensions. This improves parallelism and enables efficient pipelining of
   * inner loops. The factors are chosen to match unrolling and pipelining strategies.
   */

  // Input array partitioning:
  // - dim=1 (depth): Complete (size is 1)
  // - dim=2 (height): Cyclic factor 16 to support 5x5 window reads and load_input writes
  // - dim=3 (width): Cyclic factor 16 for 5x5 window and float4 writes in load_input
#pragma HLS ARRAY_PARTITION variable = input complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 16 dim = 2
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 16 dim = 3

  // Output array partitioning:
  // - dim=1 (local output channels): Complete (factor 16) for i1 loop unrolling
  // - dim=2 (height): Cyclic factor 16 to support pipelined writes
  // - dim=3 (width): Cyclic factor 16 to match float16 vectorization in store_output
  //   and support pipelined writes.
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 16 dim = 2
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 16 dim = 3

  // Weight array partitioning:
  // - dim=1 (local output channels): Complete (factor 16) for i1 loop unrolling
  // - dim=2 (input channels): Cyclic factor 16 to provide parallel access for pipelined j-loop stages
  //   (Complete partitioning for 256 input channels would be too large for BRAM usually)
  // - dim=3 (kernel height): Complete (factor 5) for p loop unrolling
  // - dim=4 (kernel width): Complete (factor 5) for q loop unrolling
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 16 dim = 2
#pragma HLS ARRAY_PARTITION variable = weight complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weight complete dim = 4

  /**
   * Call the main CNN function that performs the actual computation.
   */
  cnn(input, output, weight, vinput, vweight, voutput);
}
