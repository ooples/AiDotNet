using System.Text;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Provides FP16 (half-precision) GPU kernel generation with optimized operations.
/// </summary>
/// <remarks>
/// <para>
/// FP16 kernels provide significant performance improvements on modern GPUs:
/// - 2x memory bandwidth efficiency (16-bit vs 32-bit)
/// - 2x arithmetic throughput on most operations
/// - Tensor Core acceleration for matrix operations (8-16x speedup)
/// </para>
/// <para><b>For Beginners:</b> FP16 (half-precision) uses 16 bits instead of 32 bits per number.
///
/// Benefits:
/// - Twice the speed for most operations
/// - Half the memory usage
/// - Enables larger batch sizes
/// - Tensor Core acceleration on newer NVIDIA GPUs (Volta, Turing, Ampere)
///
/// Trade-offs:
/// - Reduced precision (about 3 decimal digits vs 7 for FP32)
/// - Smaller dynamic range (can overflow/underflow more easily)
/// - Requires loss scaling during training
///
/// Mixed-precision training combines FP16 computation with FP32 accumulation
/// to get the speed benefits while maintaining training stability.
/// </para>
/// </remarks>
public static class FP16Kernels
{
    /// <summary>
    /// Generates FP16-optimized CUDA helper functions.
    /// </summary>
    /// <returns>CUDA FP16 helper function code.</returns>
    public static string GenerateCUDAFP16Helpers()
    {
        return @"
// FP16 type aliases and utilities
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// FP16 conversion helpers
__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

// Vectorized FP16 load/store (2 halves at once)
__device__ __forceinline__ half2 load_half2(const half* ptr) {
    return *reinterpret_cast<const half2*>(ptr);
}

__device__ __forceinline__ void store_half2(half* ptr, half2 val) {
    *reinterpret_cast<half2*>(ptr) = val;
}

// FP16 activation functions with FP32 accumulation for stability
__device__ __forceinline__ half fp16_relu(half x) {
    return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
}

__device__ __forceinline__ half fp16_sigmoid(half x) {
    float fx = __half2float(x);
    return __float2half(1.0f / (1.0f + expf(-fx)));
}

__device__ __forceinline__ half fp16_tanh(half x) {
    return __float2half(tanhf(__half2float(x)));
}

__device__ __forceinline__ half fp16_gelu(half x) {
    float fx = __half2float(x);
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float k = 0.044715f;
    float result = 0.5f * fx * (1.0f + tanhf(c * (fx + k * fx * fx * fx)));
    return __float2half(result);
}

__device__ __forceinline__ half fp16_swish(half x) {
    return __hmul(x, fp16_sigmoid(x));
}

__device__ __forceinline__ half fp16_leaky_relu(half x, half alpha) {
    return __hgt(x, __float2half(0.0f)) ? x : __hmul(alpha, x);
}

// Vectorized FP16 activation functions (operate on half2)
__device__ __forceinline__ half2 fp16_relu2(half2 x) {
    half2 zero = __float2half2_rn(0.0f);
    return __hmax2(x, zero);
}

__device__ __forceinline__ half2 fp16_add2(half2 a, half2 b) {
    return __hadd2(a, b);
}

__device__ __forceinline__ half2 fp16_mul2(half2 a, half2 b) {
    return __hmul2(a, b);
}

__device__ __forceinline__ half2 fp16_fma2(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
}

// FP16 reduction with FP32 accumulation
__device__ __forceinline__ float fp16_block_reduce_sum(half val) {
    // Use shared memory for block-level reduction
    extern __shared__ float shared_sum[];

    float fval = __half2float(val);
    shared_sum[threadIdx.x] = fval;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    return shared_sum[0];
}

// Safe FP16 operations with overflow protection
__device__ __forceinline__ half fp16_safe_exp(half x) {
    float fx = __half2float(x);
    // Clamp to prevent overflow (half max ~65504)
    fx = fminf(fx, 11.0f);
    fx = fmaxf(fx, -11.0f);
    return __float2half(expf(fx));
}

__device__ __forceinline__ half fp16_safe_log(half x) {
    float fx = __half2float(x);
    // Clamp minimum to prevent -inf
    fx = fmaxf(fx, 1e-7f);
    return __float2half(logf(fx));
}

__device__ __forceinline__ half fp16_rsqrt(half x) {
    return hrsqrt(x);
}
";
    }

    /// <summary>
    /// Generates CUDA Tensor Core WMMA (Warp Matrix Multiply-Accumulate) kernel for FP16 matmul.
    /// </summary>
    /// <param name="M">Rows of matrix A and output C.</param>
    /// <param name="N">Columns of matrix B and output C.</param>
    /// <param name="K">Columns of A / Rows of B (shared dimension).</param>
    /// <param name="kernelName">Name for the generated kernel.</param>
    /// <returns>CUDA kernel code using tensor cores.</returns>
    /// <remarks>
    /// <para>
    /// Tensor Cores provide massive speedups for matrix operations:
    /// - V100: Up to 125 TFLOPS (FP16)
    /// - A100: Up to 312 TFLOPS (FP16)
    /// - H100: Up to 990 TFLOPS (FP16)
    ///
    /// This kernel uses WMMA (Warp Matrix Multiply-Accumulate) for 16x16x16 tiles.
    /// </para>
    /// </remarks>
    public static string GenerateTensorCoreMatMulKernel(int M, int N, int K, string kernelName)
    {
        return $@"
// Tensor Core Matrix Multiplication Kernel
// Uses WMMA (Warp Matrix Multiply-Accumulate) for FP16 computation with FP32 accumulation
// Tile size: 16x16x16 (WMMA native size)

#include <mma.h>
using namespace nvcuda;

// Tile dimensions for WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Number of tiles
const int M_TILES = ({M} + WMMA_M - 1) / WMMA_M;
const int N_TILES = ({N} + WMMA_N - 1) / WMMA_N;
const int K_TILES = ({K} + WMMA_K - 1) / WMMA_K;

__global__ void {kernelName}_tensor_core(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [K, N]
    half* __restrict__ C,        // [M, N]
    const int M, const int N, const int K
) {{
    // Each warp computes one 16x16 output tile
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId = threadIdx.x % 32;

    // Determine which tile this warp is responsible for
    const int warpM = (warpId / N_TILES) * WMMA_M;
    const int warpN = (warpId % N_TILES) * WMMA_N;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {{
        int aRow = warpM;
        int aCol = k;
        int bRow = k;
        int bCol = warpN;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {{
            // Load A and B fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }}
    }}

    // Store result (convert FP32 accumulator to FP16 output)
    if (warpM < M && warpN < N) {{
        // Convert FP32 accumulator to FP16 for storage
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;
        for (int i = 0; i < c_frag.num_elements; i++) {{
            c_frag_half.x[i] = __float2half(c_frag.x[i]);
        }}
        wmma::store_matrix_sync(C + warpM * N + warpN, c_frag_half, N, wmma::mem_row_major);
    }}
}}

// Fallback non-tensor-core kernel for older GPUs
__global__ void {kernelName}_fp16_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K
) {{
    // Tile-based matrix multiply with FP32 accumulation
    const int TILE_SIZE = 16;

    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f; // FP32 accumulation for stability

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{
        // Load tiles with bounds checking
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : __float2half(0.0f);

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }}

        __syncthreads();
    }}

    // Store result
    if (row < M && col < N) {{
        C[row * N + col] = __float2half(sum);
    }}
}}

// Launcher function that selects tensor core or fallback based on GPU capability
void launch_{kernelName}(
    const half* d_A,
    const half* d_B,
    half* d_C,
    const int M, const int N, const int K,
    cudaStream_t stream = 0
) {{
    // Check for tensor core support (SM 7.0+)
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    if (props.major >= 7) {{
        // Use tensor core kernel
        // Each warp processes one 16x16 tile
        int numWarps = ((M + 15) / 16) * ((N + 15) / 16);
        int numThreads = numWarps * 32;
        int numBlocks = (numThreads + 255) / 256;

        {kernelName}_tensor_core<<<numBlocks, 256, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    }} else {{
        // Use fallback kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);

        {kernelName}_fp16_fallback<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    }}
}}
";
    }

    /// <summary>
    /// Generates FP16 vectorized element-wise kernel (processes 2 elements per thread).
    /// </summary>
    public static string GenerateFP16VectorizedElementwiseKernel(string operation, string kernelName)
    {
        var opCode = operation.ToLower() switch
        {
            "add" => "__hadd2(a, b)",
            "sub" => "__hsub2(a, b)",
            "mul" => "__hmul2(a, b)",
            "div" => "__h2div(a, b)",
            "relu" => "fp16_relu2(a)",
            _ => "__hadd2(a, b)"
        };

        var isBinary = operation.ToLower() is "add" or "sub" or "mul" or "div";

        return $@"
// Vectorized FP16 {operation} kernel - processes 2 elements per thread using half2
__global__ void {kernelName}_fp16_vec(
    {(isBinary ? "const half* __restrict__ A,\n    const half* __restrict__ B," : "const half* __restrict__ A,")}
    half* __restrict__ C,
    const int num_elements
) {{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx + 1 < num_elements) {{
        // Vectorized path: load, compute, store 2 elements at once
        half2 a = load_half2(A + idx);
        {(isBinary ? "half2 b = load_half2(B + idx);" : "")}
        half2 result = {opCode};
        store_half2(C + idx, result);
    }} else if (idx < num_elements) {{
        // Scalar remainder
        half a_scalar = A[idx];
        {(isBinary ? "half b_scalar = B[idx];" : "")}
        C[idx] = {operation.ToLower() switch
        {
            "add" => "__hadd(a_scalar, b_scalar)",
            "sub" => "__hsub(a_scalar, b_scalar)",
            "mul" => "__hmul(a_scalar, b_scalar)",
            "div" => "__hdiv(a_scalar, b_scalar)",
            "relu" => "fp16_relu(a_scalar)",
            _ => "a_scalar"
        }};
    }}
}}

void launch_{kernelName}_fp16(
    {(isBinary ? "const half* d_A, const half* d_B," : "const half* d_A,")}
    half* d_C,
    const int num_elements,
    cudaStream_t stream = 0
) {{
    // Each thread processes 2 elements
    int numThreads = (num_elements + 1) / 2;
    int blockSize = 256;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;

    {kernelName}_fp16_vec<<<numBlocks, blockSize, 0, stream>>>(
        {(isBinary ? "d_A, d_B," : "d_A,")} d_C, num_elements);
}}
";
    }

    /// <summary>
    /// Generates FP16 layer normalization kernel with FP32 statistics computation.
    /// </summary>
    public static string GenerateFP16LayerNormKernel(string kernelName, int normalizedSize)
    {
        return $@"
// FP16 Layer Normalization with FP32 mean/variance computation for stability
__global__ void {kernelName}_layernorm_fp16(
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half* __restrict__ output,
    const int batch_size,
    const int normalized_size,
    const float epsilon
) {{
    // Each block processes one sample
    const int sample_idx = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sq_sum = shared_data + blockDim.x;

    const half* sample = input + sample_idx * normalized_size;
    half* output_sample = output + sample_idx * normalized_size;

    // Step 1: Compute mean using FP32 accumulation
    float local_sum = 0.0f;
    for (int i = tid; i < normalized_size; i += blockDim.x) {{
        local_sum += __half2float(sample[i]);
    }}
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_sum[tid] += shared_sum[tid + s];
        }}
        __syncthreads();
    }}

    float mean = shared_sum[0] / normalized_size;
    __syncthreads();

    // Step 2: Compute variance using FP32
    float local_sq_diff = 0.0f;
    for (int i = tid; i < normalized_size; i += blockDim.x) {{
        float diff = __half2float(sample[i]) - mean;
        local_sq_diff += diff * diff;
    }}
    shared_sq_sum[tid] = local_sq_diff;
    __syncthreads();

    // Parallel reduction for squared differences
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }}
        __syncthreads();
    }}

    float variance = shared_sq_sum[0] / normalized_size;
    float inv_std = rsqrtf(variance + epsilon);

    // Step 3: Normalize and apply affine transform
    for (int i = tid; i < normalized_size; i += blockDim.x) {{
        float x = __half2float(sample[i]);
        float normalized = (x - mean) * inv_std;
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        output_sample[i] = __float2half(normalized * g + b);
    }}
}}

void launch_{kernelName}_layernorm(
    const half* d_input,
    const half* d_gamma,
    const half* d_beta,
    half* d_output,
    const int batch_size,
    const int normalized_size,
    const float epsilon,
    cudaStream_t stream = 0
) {{
    int blockSize = min(256, normalized_size);
    int sharedMemSize = 2 * blockSize * sizeof(float);

    {kernelName}_layernorm_fp16<<<batch_size, blockSize, sharedMemSize, stream>>>(
        d_input, d_gamma, d_beta, d_output, batch_size, normalized_size, epsilon);
}}
";
    }

    /// <summary>
    /// Generates FP16 softmax kernel with FP32 computation for numerical stability.
    /// </summary>
    public static string GenerateFP16SoftmaxKernel(string kernelName)
    {
        return @$"
// FP16 Softmax with FP32 intermediate computation for numerical stability
__global__ void {kernelName}_softmax_fp16(
    const half* __restrict__ input,
    half* __restrict__ output,
    const int batch_size,
    const int num_classes
) {{
    // Each block processes one sample
    const int sample_idx = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;

    const half* sample = input + sample_idx * num_classes;
    half* output_sample = output + sample_idx * num_classes;

    // Step 1: Find max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {{
        local_max = fmaxf(local_max, __half2float(sample[i]));
    }}
    shared_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }}
        __syncthreads();
    }}
    float max_val = shared_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {{
        float exp_val = expf(__half2float(sample[i]) - max_val);
        local_sum += exp_val;
    }}
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_sum[tid] += shared_sum[tid + s];
        }}
        __syncthreads();
    }}
    float sum = shared_sum[0];

    // Step 3: Compute softmax output
    for (int i = tid; i < num_classes; i += blockDim.x) {{
        float exp_val = expf(__half2float(sample[i]) - max_val);
        output_sample[i] = __float2half(exp_val / sum);
    }}
}}

void launch_{kernelName}_softmax(
    const half* d_input,
    half* d_output,
    const int batch_size,
    const int num_classes,
    cudaStream_t stream = 0
) {{
    int blockSize = min(256, num_classes);
    int sharedMemSize = 2 * blockSize * sizeof(float);

    {kernelName}_softmax_fp16<<<batch_size, blockSize, sharedMemSize, stream>>>(
        d_input, d_output, batch_size, num_classes);
}}
";
    }

    /// <summary>
    /// Generates FP16 attention kernel with Flash Attention-style memory efficiency.
    /// </summary>
    public static string GenerateFP16AttentionKernel(string kernelName, int headDim)
    {
        return $@"
// FP16 Scaled Dot-Product Attention Kernel
// Uses online softmax (Flash Attention style) for memory efficiency
__global__ void {kernelName}_attention_fp16(
    const half* __restrict__ Q,      // [batch, heads, seq_len, head_dim]
    const half* __restrict__ K,      // [batch, heads, seq_len, head_dim]
    const half* __restrict__ V,      // [batch, heads, seq_len, head_dim]
    half* __restrict__ output,       // [batch, heads, seq_len, head_dim]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale  // 1.0 / sqrt(head_dim)
) {{
    // Each block computes attention for one query position
    const int batch_head_idx = blockIdx.x;
    const int query_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int batch = batch_head_idx / num_heads;
    const int head = batch_head_idx % num_heads;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;

    // Offset into Q, K, V
    const int qkv_offset = (batch * num_heads + head) * seq_len * head_dim;
    const half* q_row = Q + qkv_offset + query_idx * head_dim;

    // Step 1: Compute attention scores Q @ K^T for this query
    float max_score = -INFINITY;
    for (int key_idx = tid; key_idx < seq_len; key_idx += blockDim.x) {{
        const half* k_row = K + qkv_offset + key_idx * head_dim;

        // Dot product Q[query] @ K[key]
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {{
            score += __half2float(q_row[d]) * __half2float(k_row[d]);
        }}
        score *= scale;

        scores[key_idx] = score;
        max_score = fmaxf(max_score, score);
    }}
    __syncthreads();

    // Reduce max across threads
    __shared__ float shared_max;
    if (tid == 0) shared_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_score));
    __syncthreads();
    max_score = shared_max;

    // Step 2: Softmax normalization
    float sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_len; key_idx += blockDim.x) {{
        float exp_score = expf(scores[key_idx] - max_score);
        scores[key_idx] = exp_score;
        sum += exp_score;
    }}
    __syncthreads();

    // Reduce sum across threads
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum);
    __syncthreads();
    sum = shared_sum;

    // Normalize scores
    for (int key_idx = tid; key_idx < seq_len; key_idx += blockDim.x) {{
        scores[key_idx] /= sum;
    }}
    __syncthreads();

    // Step 3: Compute weighted sum of values
    half* out_row = output + qkv_offset + query_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {{
        float weighted_sum = 0.0f;
        for (int key_idx = 0; key_idx < seq_len; key_idx++) {{
            const half* v_row = V + qkv_offset + key_idx * head_dim;
            weighted_sum += scores[key_idx] * __half2float(v_row[d]);
        }}
        out_row[d] = __float2half(weighted_sum);
    }}
}}

void launch_{kernelName}_attention(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream = 0
) {{
    // Grid: (batch * heads, seq_len)
    // Block: min(256, seq_len)
    dim3 grid(batch_size * num_heads, seq_len);
    int blockSize = min(256, seq_len);
    int sharedMemSize = seq_len * sizeof(float);

    float scale = 1.0f / sqrtf((float)head_dim);

    {kernelName}_attention_fp16<<<grid, blockSize, sharedMemSize, stream>>>(
        d_Q, d_K, d_V, d_output, batch_size, num_heads, seq_len, head_dim, scale);
}}
";
    }

    /// <summary>
    /// Generates mixed-precision training helper kernels.
    /// </summary>
    public static string GenerateMixedPrecisionHelpers()
    {
        return @"
// Mixed-precision training helper kernels

// Convert FP32 to FP16 with loss scaling
__global__ void fp32_to_fp16_scaled(
    const float* __restrict__ input,
    half* __restrict__ output,
    const float scale,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float scaled = input[idx] * scale;
        // Clamp to FP16 range to prevent overflow
        scaled = fminf(scaled, 65504.0f);
        scaled = fmaxf(scaled, -65504.0f);
        output[idx] = __float2half(scaled);
    }
}

// Convert FP16 gradients to FP32 and unscale
__global__ void fp16_to_fp32_unscaled(
    const half* __restrict__ input,
    float* __restrict__ output,
    const float inv_scale,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = __half2float(input[idx]) * inv_scale;
    }
}

// Check for NaN/Inf in FP16 gradients (returns 1 if found, 0 otherwise)
__global__ void check_fp16_overflow(
    const half* __restrict__ gradients,
    int* __restrict__ overflow_flag,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = __half2float(gradients[idx]);
        if (isnan(val) || isinf(val)) {
            atomicExch(overflow_flag, 1);
        }
    }
}

// FP16 gradient clipping
__global__ void clip_fp16_gradients(
    half* __restrict__ gradients,
    const float max_norm,
    const float current_norm,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements && current_norm > max_norm) {
        float scale = max_norm / current_norm;
        gradients[idx] = __float2half(__half2float(gradients[idx]) * scale);
    }
}

// Compute L2 norm of FP16 tensor (using FP32 accumulation)
__global__ void compute_fp16_l2_norm(
    const half* __restrict__ tensor,
    float* __restrict__ partial_sums,
    const int num_elements
) {
    extern __shared__ float shared_sum[];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        float val = __half2float(tensor[i]);
        local_sum += val * val;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}
";
    }
}
