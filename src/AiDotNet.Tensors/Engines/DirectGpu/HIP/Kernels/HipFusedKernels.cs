// Copyright (c) AiDotNet. All rights reserved.
// Fused operation kernels for HIP - GEMM + Bias + Activation in single pass.
// Eliminates memory round-trips for 20-50% performance gain.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// Fused HIP kernels that combine multiple operations to eliminate memory round-trips.
/// Uses shared memory tiling for optimal GPU performance.
/// </summary>
internal static class HipFusedKernels
{
    /// <summary>
    /// Gets all fused kernel sources.
    /// </summary>
    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define TILE_SIZE 16

// ===========================================================================
// FUSED KERNELS: GEMM + BIAS + ACTIVATION
// Single kernel for entire DenseLayer forward pass.
// Eliminates memory round-trips between operations.
// ===========================================================================

// Fused GEMM + Bias + ReLU
// output = ReLU(A * B + bias)
extern ""C"" __global__ void gemm_bias_relu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = sum + bias[col];
        C[row * N + col] = fmaxf(0.0f, result);
    }
}

// Fused GEMM + Bias + GELU (for Transformer FFN)
// output = GELU(A * B + bias)
extern ""C"" __global__ void gemm_bias_gelu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float x = sum + bias[col];

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEFF = 0.044715f;
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        C[row * N + col] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Fused GEMM + Bias + Sigmoid
// output = Sigmoid(A * B + bias)
extern ""C"" __global__ void gemm_bias_sigmoid(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = 1.0f / (1.0f + expf(-x));
    }
}

// Fused GEMM + Bias + Tanh
// output = Tanh(A * B + bias)
extern ""C"" __global__ void gemm_bias_tanh(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = tanhf(x);
    }
}

// Fused GEMM + Bias (no activation)
// output = A * B + bias
extern ""C"" __global__ void gemm_bias(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

// Fused GEMM + Bias + Swish (SiLU)
// output = Swish(A * B + bias) = x * sigmoid(x)
extern ""C"" __global__ void gemm_bias_swish(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float x = sum + bias[col];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        C[row * N + col] = x * sigmoid;
    }
}

// Fused GEMM + Bias + LeakyReLU
// output = LeakyReLU(A * B + bias, alpha)
extern ""C"" __global__ void gemm_bias_leaky_relu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K, float alpha)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = x >= 0.0f ? x : alpha * x;
    }
}

// ===========================================================================
// FUSED LAYERNORM + ACTIVATION KERNELS
// Single kernel for LayerNorm + scale/bias + activation.
// ===========================================================================

// Fused LayerNorm + ReLU
// output = ReLU(gamma * (x - mean) / sqrt(var + eps) + beta)
extern ""C"" __global__ void layernorm_relu(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batchSize, int normalizedSize, float epsilon)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];
    float* sdata = smem;

    int tid = threadIdx.x;
    int baseIdx = batch * normalizedSize;

    // Compute mean using parallel reduction
    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        localSum += input[baseIdx + i];
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / normalizedSize;
    __syncthreads();

    // Compute variance
    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float diff = input[baseIdx + i] - mean;
        localVar += diff * diff;
    }
    sdata[tid] = localVar;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / normalizedSize + epsilon);

    // Normalize, scale, shift, and apply ReLU
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float normalized = (input[baseIdx + i] - mean) * invStd;
        float result = gamma[i] * normalized + beta[i];
        output[baseIdx + i] = fmaxf(0.0f, result);
    }
}

// Fused LayerNorm + GELU (critical for Transformers)
extern ""C"" __global__ void layernorm_gelu(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batchSize, int normalizedSize, float epsilon)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];
    float* sdata = smem;

    int tid = threadIdx.x;
    int baseIdx = batch * normalizedSize;

    // Compute mean
    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        localSum += input[baseIdx + i];
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / normalizedSize;
    __syncthreads();

    // Compute variance
    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float diff = input[baseIdx + i] - mean;
        localVar += diff * diff;
    }
    sdata[tid] = localVar;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / normalizedSize + epsilon);

    // Normalize, scale, shift, and apply GELU
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float normalized = (input[baseIdx + i] - mean) * invStd;
        float x = gamma[i] * normalized + beta[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[baseIdx + i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// ===========================================================================
// FUSED RESIDUAL + LAYERNORM KERNELS (for Transformer blocks)
// ===========================================================================

// Fused Residual Add + LayerNorm
// output = LayerNorm(x + residual)
extern ""C"" __global__ void residual_layernorm(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batchSize, int normalizedSize, float epsilon)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];
    float* sdata = smem;

    int tid = threadIdx.x;
    int baseIdx = batch * normalizedSize;

    // Compute mean of (input + residual)
    float localSum = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        localSum += input[baseIdx + i] + residual[baseIdx + i];
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / normalizedSize;
    __syncthreads();

    // Compute variance
    float localVar = 0.0f;
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float val = input[baseIdx + i] + residual[baseIdx + i];
        float diff = val - mean;
        localVar += diff * diff;
    }
    sdata[tid] = localVar;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / normalizedSize + epsilon);

    // Normalize, scale, shift
    for (int i = tid; i < normalizedSize; i += blockDim.x) {
        float val = input[baseIdx + i] + residual[baseIdx + i];
        float normalized = (val - mean) * invStd;
        output[baseIdx + i] = gamma[i] * normalized + beta[i];
    }
}

// ===========================================================================
// FUSED SOFTMAX KERNELS (for Attention)
// ===========================================================================

// Fused Scale + Softmax (for attention: softmax(Q*K^T / sqrt(d)))
extern ""C"" __global__ void scaled_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batchSize, int seqLen, float scale)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int baseIdx = batch * seqLen;

    // Find max for numerical stability
    float localMax = -INFINITY;
    for (int i = tid; i < seqLen; i += blockDim.x) {
        float val = input[baseIdx + i] * scale;
        if (val > localMax) localMax = val;
    }
    smem[tid] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }
    float maxVal = smem[0];
    __syncthreads();

    // Compute sum of exp
    float localSum = 0.0f;
    for (int i = tid; i < seqLen; i += blockDim.x) {
        float val = input[baseIdx + i] * scale;
        localSum += expf(val - maxVal);
    }
    smem[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sumExp = smem[0];
    float invSum = 1.0f / sumExp;

    // Compute softmax
    for (int i = tid; i < seqLen; i += blockDim.x) {
        float val = input[baseIdx + i] * scale;
        output[baseIdx + i] = expf(val - maxVal) * invSum;
    }
}

// ===========================================================================
// FUSED BIAS + DROPOUT KERNELS
// ===========================================================================

// Fused Bias + Dropout (for training)
extern ""C"" __global__ void bias_dropout(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const unsigned int* __restrict__ mask,
    int rows, int cols, float dropoutProb, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx >= size) return;

    int col = idx % cols;
    float val = input[idx] + bias[col];

    // Apply dropout using pre-generated mask
    unsigned int m = mask[idx];
    output[idx] = (m != 0) ? val * scale : 0.0f;
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
            "gemm_bias_relu",
            "gemm_bias_gelu",
            "gemm_bias_sigmoid",
            "gemm_bias_tanh",
            "gemm_bias",
            "gemm_bias_swish",
            "gemm_bias_leaky_relu",
            "layernorm_relu",
            "layernorm_gelu",
            "residual_layernorm",
            "scaled_softmax",
            "bias_dropout"
        };
    }
}
