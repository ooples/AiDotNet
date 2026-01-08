// Copyright (c) AiDotNet. All rights reserved.
// Fused operation kernels - GEMM + Bias + Activation in single pass.
// Eliminates memory round-trips for 20-50% performance gain.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Fused GPU kernels that combine multiple operations to eliminate memory round-trips.
    /// These are key optimizations missing from CLBlast.
    /// </summary>
    /// <remarks>
    /// <para><b>Performance Benefit:</b></para>
    /// <para>
    /// For DenseLayer forward pass, fusing GEMM + Bias + Activation:
    /// - Eliminates 2 global memory round-trips
    /// - Expected gain: 20-50% for memory-bound workloads
    /// </para>
    /// </remarks>
    internal static class FusedKernels
    {
        /// <summary>
        /// Gets all fused kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// FUSED KERNELS: GEMM + BIAS + ACTIVATION
// Single kernel for entire DenseLayer forward pass.
// Eliminates memory round-trips between operations.
// ===========================================================================

#define TILE_SIZE 16

// Fused GEMM + Bias + ReLU
// output = ReLU(A * B + bias)
__kernel void gemm_bias_relu(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    // Tile-based GEMM with fused bias and ReLU
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and ReLU activation
    if (row < M && col < N) {
        float result = sum + bias[col];
        C[row * N + col] = fmax(0.0f, result);  // ReLU
    }
}

// Fused GEMM + Bias + GELU (for Transformer FFN)
// output = GELU(A * B + bias)
__kernel void gemm_bias_gelu(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and GELU activation
    if (row < M && col < N) {
        float x = sum + bias[col];

        // GELU approximation
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEFF = 0.044715f;
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        C[row * N + col] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// Fused GEMM + Bias + Sigmoid
// output = Sigmoid(A * B + bias)
__kernel void gemm_bias_sigmoid(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and Sigmoid activation
    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = 1.0f / (1.0f + exp(-x));
    }
}

// Fused GEMM + Bias + Tanh
// output = Tanh(A * B + bias)
__kernel void gemm_bias_tanh(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and Tanh activation
    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = tanh(x);
    }
}

// Fused GEMM + Bias (no activation, but saves a kernel launch)
// output = A * B + bias
__kernel void gemm_bias(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Just bias add, no activation
    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

// ===========================================================================
// FUSED LAYERNORM + ACTIVATION KERNELS
// ===========================================================================

// Fused LayerNorm + ReLU
__kernel void layernorm_relu(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    int batchSize, int normalizedSize, float epsilon,
    __local float* scratch)
{
    int batch = get_group_id(0);
    if (batch >= batchSize) return;

    int tid = get_local_id(0);
    int baseIdx = batch * normalizedSize;

    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        localSum += input[baseIdx + i];
    }
    scratch[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0] / normalizedSize;
    barrier(CLK_LOCAL_MEM_FENCE);

    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float diff = input[baseIdx + i] - mean;
        localVar += diff * diff;
    }
    scratch[tid] = localVar;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float invStd = rsqrt(scratch[0] / normalizedSize + epsilon);

    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float normalized = (input[baseIdx + i] - mean) * invStd;
        float result = gamma[i] * normalized + beta[i];
        output[baseIdx + i] = fmax(0.0f, result);
    }
}

// Fused LayerNorm + GELU
__kernel void layernorm_gelu(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    int batchSize, int normalizedSize, float epsilon,
    __local float* scratch)
{
    int batch = get_group_id(0);
    if (batch >= batchSize) return;

    int tid = get_local_id(0);
    int baseIdx = batch * normalizedSize;

    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        localSum += input[baseIdx + i];
    }
    scratch[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0] / normalizedSize;
    barrier(CLK_LOCAL_MEM_FENCE);

    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float diff = input[baseIdx + i] - mean;
        localVar += diff * diff;
    }
    scratch[tid] = localVar;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float invStd = rsqrt(scratch[0] / normalizedSize + epsilon);

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float normalized = (input[baseIdx + i] - mean) * invStd;
        float x = gamma[i] * normalized + beta[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[baseIdx + i] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// Fused Residual + LayerNorm
__kernel void residual_layernorm(
    __global const float* input,
    __global const float* residual,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    int batchSize, int normalizedSize, float epsilon,
    __local float* scratch)
{
    int batch = get_group_id(0);
    if (batch >= batchSize) return;

    int tid = get_local_id(0);
    int baseIdx = batch * normalizedSize;

    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        localSum += input[baseIdx + i] + residual[baseIdx + i];
    }
    scratch[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = scratch[0] / normalizedSize;
    barrier(CLK_LOCAL_MEM_FENCE);

    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float val = input[baseIdx + i] + residual[baseIdx + i];
        float diff = val - mean;
        localVar += diff * diff;
    }
    scratch[tid] = localVar;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float invStd = rsqrt(scratch[0] / normalizedSize + epsilon);

    for (int i = tid; i < normalizedSize; i += get_local_size(0)) {
        float val = input[baseIdx + i] + residual[baseIdx + i];
        float normalized = (val - mean) * invStd;
        output[baseIdx + i] = gamma[i] * normalized + beta[i];
    }
}

// Fused Scale + Softmax
__kernel void scaled_softmax(
    __global const float* input,
    __global float* output,
    int batchSize, int seqLen, float scale,
    __local float* scratch)
{
    int batch = get_group_id(0);
    if (batch >= batchSize) return;

    int tid = get_local_id(0);
    int baseIdx = batch * seqLen;

    float localMax = -INFINITY;
    for (int i = tid; i < seqLen; i += get_local_size(0)) {
        float val = input[baseIdx + i] * scale;
        if (val > localMax) localMax = val;
    }
    scratch[tid] = localMax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (scratch[tid + s] > scratch[tid]) scratch[tid] = scratch[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float maxVal = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float localSum = 0.0f;
    for (int i = tid; i < seqLen; i += get_local_size(0)) {
        float val = input[baseIdx + i] * scale;
        localSum += exp(val - maxVal);
    }
    scratch[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float invSum = 1.0f / scratch[0];

    for (int i = tid; i < seqLen; i += get_local_size(0)) {
        float val = input[baseIdx + i] * scale;
        output[baseIdx + i] = exp(val - maxVal) * invSum;
    }
}

// Fused Bias + Dropout
__kernel void bias_dropout(
    __global const float* input,
    __global float* output,
    __global const float* bias,
    __global const uint* mask,
    int rows, int cols, float dropoutProb, float scale)
{
    int idx = get_global_id(0);
    int size = rows * cols;
    if (idx >= size) return;

    int col = idx % cols;
    float val = input[idx] + bias[col];
    uint m = mask[idx];
    output[idx] = (m != 0) ? val * scale : 0.0f;
}

// ===========================================================================
// FUSED BATCHNORM + ACTIVATION KERNELS (INFERENCE MODE)
// Uses running mean/var statistics, not batch statistics
// ===========================================================================

// Fused BatchNorm + ReLU (Inference)
// output = ReLU(gamma * (input - runningMean) / sqrt(runningVar + eps) + beta)
__kernel void batchnorm_relu(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global const float* runningMean,
    __global const float* runningVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;

    // Compute channel index
    const int c = (idx / spatialSize) % channels;

    // Normalize and apply affine transform
    float mean = runningMean[c];
    float var = runningVar[c];
    float invStd = rsqrt(var + epsilon);
    float g = gamma[c];
    float b = beta[c];

    float normalized = (input[idx] - mean) * invStd;
    float result = g * normalized + b;
    output[idx] = fmax(0.0f, result);  // ReLU
}

// Fused BatchNorm + GELU (Inference)
__kernel void batchnorm_gelu(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global const float* runningMean,
    __global const float* runningVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;

    const int c = (idx / spatialSize) % channels;

    float mean = runningMean[c];
    float var = runningVar[c];
    float invStd = rsqrt(var + epsilon);
    float g = gamma[c];
    float b = beta[c];

    float normalized = (input[idx] - mean) * invStd;
    float x = g * normalized + b;

    // GELU approximation
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}

// Fused BatchNorm + Sigmoid (Inference)
__kernel void batchnorm_sigmoid(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global const float* runningMean,
    __global const float* runningVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;

    const int c = (idx / spatialSize) % channels;

    float mean = runningMean[c];
    float var = runningVar[c];
    float invStd = rsqrt(var + epsilon);
    float g = gamma[c];
    float b = beta[c];

    float normalized = (input[idx] - mean) * invStd;
    float x = g * normalized + b;
    output[idx] = 1.0f / (1.0f + exp(-x));  // Sigmoid
}

// Fused BatchNorm + Tanh (Inference)
__kernel void batchnorm_tanh(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global const float* runningMean,
    __global const float* runningVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;

    const int c = (idx / spatialSize) % channels;

    float mean = runningMean[c];
    float var = runningVar[c];
    float invStd = rsqrt(var + epsilon);
    float g = gamma[c];
    float b = beta[c];

    float normalized = (input[idx] - mean) * invStd;
    float x = g * normalized + b;
    output[idx] = tanh(x);
}

// Fused Residual + BatchNorm + ReLU (for ResNet skip connections)
__kernel void residual_batchnorm_relu(
    __global const float* input,
    __global const float* residual,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global const float* runningMean,
    __global const float* runningVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;

    const int c = (idx / spatialSize) % channels;

    float mean = runningMean[c];
    float var = runningVar[c];
    float invStd = rsqrt(var + epsilon);
    float g = gamma[c];
    float b = beta[c];

    // BatchNorm the input, then add residual
    float normalized = (input[idx] - mean) * invStd;
    float result = g * normalized + b + residual[idx];
    output[idx] = fmax(0.0f, result);  // ReLU
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "gemm_bias_relu", "gemm_bias_gelu",
                "gemm_bias_sigmoid", "gemm_bias_tanh", "gemm_bias",
                "layernorm_relu", "layernorm_gelu",
                "residual_layernorm", "scaled_softmax", "bias_dropout",
                "batchnorm_relu", "batchnorm_gelu",
                "batchnorm_sigmoid", "batchnorm_tanh", "residual_batchnorm_relu"
            };
        }
    }
}
