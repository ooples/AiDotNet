// Copyright (c) AiDotNet. All rights reserved.
// CUDA activation kernels and simple elementwise ops.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    internal static class CudaActivationKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

extern ""C"" __global__ void relu(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x > 0.0f ? x : 0.0f;
}

extern ""C"" __global__ void sigmoid(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = 1.0f / (1.0f + expf(-x));
}

extern ""C"" __global__ void tanh_activation(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = tanhf(input[idx]);
}

extern ""C"" __global__ void gelu(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const float sqrt2OverPi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x = input[idx];
    float x3 = x * x * x;
    float inner = sqrt2OverPi * (x + coeff * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

extern ""C"" __global__ void swish(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

extern ""C"" __global__ void softmax(const float* input, float* output, int batchSize, int features)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    float maxVal = -INFINITY;
    int baseIdx = batch * features;
    for (int f = 0; f < features; f++)
    {
        float v = input[baseIdx + f];
        if (v > maxVal) maxVal = v;
    }

    float sumExp = 0.0f;
    for (int f = 0; f < features; f++)
    {
        float expVal = expf(input[baseIdx + f] - maxVal);
        output[baseIdx + f] = expVal;
        sumExp += expVal;
    }

    float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 1.0f;
    for (int f = 0; f < features; f++)
    {
        output[baseIdx + f] *= invSum;
    }
}

extern ""C"" __global__ void add_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] + B[idx];
}

extern ""C"" __global__ void subtract_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] - B[idx];
}

extern ""C"" __global__ void multiply_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    C[idx] = A[idx] * B[idx];
}

extern ""C"" __global__ void divide_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float b = B[idx];
    C[idx] = (b != 0.0f) ? (A[idx] / b) : 0.0f;
}

extern ""C"" __global__ void min_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float a = A[idx];
    float b = B[idx];
    C[idx] = a < b ? a : b;
}

extern ""C"" __global__ void max_vectors(const float* A, const float* B, float* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float a = A[idx];
    float b = B[idx];
    C[idx] = a > b ? a : b;
}

extern ""C"" __global__ void scale_vector(const float* A, float* B, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = A[idx] * scalar;
}

extern ""C"" __global__ void abs_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = A[idx];
    B[idx] = x < 0.0f ? -x : x;
}

extern ""C"" __global__ void exp_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = expf(A[idx]);
}

extern ""C"" __global__ void log_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = logf(A[idx]);
}

extern ""C"" __global__ void log2_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = log2f(A[idx]);
}

extern ""C"" __global__ void exp2_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = exp2f(A[idx]);
}

extern ""C"" __global__ void exp10_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = powf(10.0f, A[idx]);
}

extern ""C"" __global__ void expm1_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = expf(A[idx]) - 1.0f;
}

extern ""C"" __global__ void log1p_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = logf(1.0f + A[idx]);
}

extern ""C"" __global__ void sqrt_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sqrtf(A[idx]);
}

extern ""C"" __global__ void sign_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = A[idx];
    B[idx] = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}

extern ""C"" __global__ void power_scalar(const float* A, float* B, float exponent, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = powf(A[idx], exponent);
}
extern ""C"" __global__ void reduce_sum(const float* input, float* output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < (unsigned int)size) ? input[idx] : 0.0f;
    scratch[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            scratch[tid] += scratch[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = scratch[0];
}

extern ""C"" __global__ void reduce_max(const float* input, float* output, int size)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < (unsigned int)size) ? input[idx] : -INFINITY;
    scratch[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float other = scratch[tid + s];
            if (other > scratch[tid])
                scratch[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = scratch[0];
}

extern ""C"" __global__ void sum_axis(const float* input, float* output, int outerSize, int reduceSize)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned int)outerSize) return;

    float sum = 0.0f;
    unsigned int baseOffset = idx * (unsigned int)reduceSize;
    for (int i = 0; i < reduceSize; ++i)
        sum += input[baseOffset + (unsigned int)i];
    output[idx] = sum;
}

extern ""C"" __global__ void bias_add(float* data, const float* bias, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx >= size) return;
    int col = idx % cols;
    data[idx] += bias[col];
}

// Bias add with separate output: C[i,j] = A[i,j] + bias[j]
extern ""C"" __global__ void bias_add_out(const float* A, const float* bias, float* C, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx >= size) return;
    int col = idx % cols;
    C[idx] = A[idx] + bias[col];
}

// Conv2D bias add in NCHW format: output[b,c,h,w] += bias[c]
// Memory layout: output is [batch, channels, height, width] in row-major order
// For element at linear index idx:
//   - spatial_idx = idx % spatialSize (position within H*W)
//   - channel = (idx / spatialSize) % channels
extern ""C"" __global__ void conv2d_bias_add(float* output, const float* bias, int batch, int channels, int spatialSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * spatialSize;
    if (idx >= totalSize) return;
    int channel = (idx / spatialSize) % channels;
    output[idx] += bias[channel];
}

// ===========================================================================
// TRIGONOMETRIC KERNELS
// ===========================================================================

extern ""C"" __global__ void sin_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sinf(A[idx]);
}

extern ""C"" __global__ void cos_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = cosf(A[idx]);
}

extern ""C"" __global__ void tan_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = tanf(A[idx]);
}

extern ""C"" __global__ void asin_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = asinf(A[idx]);
}

extern ""C"" __global__ void acos_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = acosf(A[idx]);
}

extern ""C"" __global__ void atan_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = atanf(A[idx]);
}

// ===========================================================================
// HYPERBOLIC KERNELS
// ===========================================================================

extern ""C"" __global__ void sinh_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = sinhf(A[idx]);
}

extern ""C"" __global__ void cosh_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = coshf(A[idx]);
}

extern ""C"" __global__ void asinh_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = asinhf(A[idx]);
}

extern ""C"" __global__ void acosh_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = acoshf(A[idx]);
}

extern ""C"" __global__ void atanh_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = atanhf(A[idx]);
}

// ===========================================================================
// ADDITIONAL UNARY KERNELS
// ===========================================================================

extern ""C"" __global__ void reciprocal_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = 1.0f / A[idx];
}

extern ""C"" __global__ void cbrt_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = cbrtf(A[idx]);
}

extern ""C"" __global__ void log10_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = log10f(A[idx]);
}

extern ""C"" __global__ void negate_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = -A[idx];
}

extern ""C"" __global__ void floor_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = floorf(A[idx]);
}

extern ""C"" __global__ void ceil_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = ceilf(A[idx]);
}

extern ""C"" __global__ void round_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = roundf(A[idx]);
}

extern ""C"" __global__ void trunc_vector(const float* A, float* B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    B[idx] = truncf(A[idx]);
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                // Activations
                "relu",
                "sigmoid",
                "tanh_activation",
                "gelu",
                "swish",
                "softmax",
                // Element-wise binary
                "add_vectors",
                "subtract_vectors",
                "multiply_vectors",
                "divide_vectors",
                "min_vectors",
                "max_vectors",
                // Scalar ops
                "scale_vector",
                "power_scalar",
                // Unary math
                "abs_vector",
                "exp_vector",
                "log_vector",
                "log2_vector",
                "exp2_vector",
                "exp10_vector",
                "expm1_vector",
                "log1p_vector",
                "sqrt_vector",
                "sign_vector",
                // Trigonometric
                "sin_vector",
                "cos_vector",
                "tan_vector",
                "asin_vector",
                "acos_vector",
                "atan_vector",
                // Hyperbolic
                "sinh_vector",
                "cosh_vector",
                "asinh_vector",
                "acosh_vector",
                "atanh_vector",
                // Additional unary
                "reciprocal_vector",
                "cbrt_vector",
                "log10_vector",
                "negate_vector",
                "floor_vector",
                "ceil_vector",
                "round_vector",
                "trunc_vector",
                // Reductions
                "reduce_sum",
                "reduce_max",
                "sum_axis",
                "bias_add",
                "bias_add_out",
                "conv2d_bias_add"
            };
        }
    }
}

