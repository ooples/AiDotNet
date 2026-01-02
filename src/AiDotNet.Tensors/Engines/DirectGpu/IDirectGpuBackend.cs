namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Interface for direct GPU backend implementations (OpenCL, CUDA, etc.).
/// All operations use float32 for optimal GPU performance.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// This interface abstracts vendor-specific GPU implementations while enforcing
/// float32-only operations for maximum performance. Generic type conversion
/// happens at a higher level (DirectGpuEngine), keeping backends simple and fast.
/// </para>
/// <para><b>Performance Targets:</b></para>
/// <list type="bullet">
/// <item>GEMM: 10,000+ GFLOPS (10-100x faster than CLBlast)</item>
/// <item>Fused ops: 20-50% faster than separate operations</item>
/// <item>Memory: GPU-resident buffers to eliminate transfer overhead</item>
/// </list>
/// </remarks>
public interface IDirectGpuBackend : IDisposable
{
    /// <summary>
    /// Gets whether this backend is available and initialized.
    /// </summary>
    bool IsAvailable { get; }

    /// <summary>
    /// Gets the backend name (e.g., "OpenCL", "CUDA").
    /// </summary>
    string BackendName { get; }

    /// <summary>
    /// Gets the GPU device name.
    /// </summary>
    string DeviceName { get; }

    /// <summary>
    /// Gets the GPU vendor (AMD, NVIDIA, Intel).
    /// </summary>
    string DeviceVendor { get; }

    /// <summary>
    /// Gets the number of compute units on the GPU.
    /// </summary>
    int ComputeUnits { get; }

    /// <summary>
    /// Gets the total global memory in bytes.
    /// </summary>
    long GlobalMemoryBytes { get; }

    /// <summary>
    /// Gets the local (shared) memory per workgroup in bytes.
    /// </summary>
    long LocalMemoryBytes { get; }

    #region Memory Management

    /// <summary>
    /// Allocates a GPU buffer and uploads data.
    /// </summary>
    /// <param name="data">CPU data to upload.</param>
    /// <returns>Handle to the GPU buffer.</returns>
    IGpuBuffer AllocateBuffer(float[] data);

    /// <summary>
    /// Allocates an empty GPU buffer.
    /// </summary>
    /// <param name="size">Number of float elements.</param>
    /// <returns>Handle to the GPU buffer.</returns>
    IGpuBuffer AllocateBuffer(int size);

    /// <summary>
    /// Downloads GPU buffer contents to CPU.
    /// </summary>
    /// <param name="buffer">GPU buffer to download.</param>
    /// <returns>CPU array with buffer contents.</returns>
    float[] DownloadBuffer(IGpuBuffer buffer);

    /// <summary>
    /// Downloads GPU buffer contents to existing CPU array.
    /// </summary>
    /// <param name="buffer">GPU buffer to download.</param>
    /// <param name="destination">Destination CPU array.</param>
    void DownloadBuffer(IGpuBuffer buffer, float[] destination);

    #endregion

    #region GEMM Operations

    /// <summary>
    /// General matrix multiplication: C = alpha * A * B + beta * C
    /// </summary>
    /// <param name="A">GPU buffer for matrix A (M x K).</param>
    /// <param name="B">GPU buffer for matrix B (K x N).</param>
    /// <param name="C">GPU buffer for matrix C (M x N), also output.</param>
    /// <param name="M">Rows of A and C.</param>
    /// <param name="N">Columns of B and C.</param>
    /// <param name="K">Columns of A and rows of B.</param>
    /// <param name="alpha">Scalar multiplier for A*B.</param>
    /// <param name="beta">Scalar multiplier for C.</param>
    void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);

    /// <summary>
    /// Matrix multiplication with new output buffer: C = A * B
    /// </summary>
    /// <param name="A">GPU buffer for matrix A (M x K).</param>
    /// <param name="B">GPU buffer for matrix B (K x N).</param>
    /// <param name="M">Rows of A.</param>
    /// <param name="N">Columns of B.</param>
    /// <param name="K">Columns of A and rows of B.</param>
    /// <returns>GPU buffer containing result C (M x N).</returns>
    IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K);

    /// <summary>
    /// Batched matrix multiplication for many small matrices: C[i] = alpha * A[i] * B[i] + beta * C[i]
    /// </summary>
    /// <remarks>
    /// <para>Uses work-stealing persistent kernel for efficient processing of many small matrix multiplications.</para>
    /// <para>All matrices within a batch must have the same dimensions (M x K for A, K x N for B).</para>
    /// <para>Matrices are stored concatenated in memory: A[0], A[1], ..., A[batchCount-1].</para>
    /// </remarks>
    /// <param name="A">GPU buffer containing all A matrices concatenated (batchCount * M * K elements).</param>
    /// <param name="B">GPU buffer containing all B matrices concatenated (batchCount * K * N elements).</param>
    /// <param name="C">GPU buffer for output C matrices concatenated (batchCount * M * N elements).</param>
    /// <param name="M">Rows of each A matrix and each C matrix.</param>
    /// <param name="N">Columns of each B matrix and each C matrix.</param>
    /// <param name="K">Columns of each A matrix and rows of each B matrix.</param>
    /// <param name="batchCount">Number of matrix multiplication operations to perform.</param>
    /// <param name="alpha">Scalar multiplier for A*B (default 1.0).</param>
    /// <param name="beta">Scalar multiplier for C (default 0.0).</param>
    void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f);

    #endregion

    #region Fused Operations (Eliminates Memory Round-Trips)

    /// <summary>
    /// Fused GEMM + Bias + ReLU: output = ReLU(A * B + bias)
    /// </summary>
    /// <param name="A">Input matrix (M x K).</param>
    /// <param name="B">Weight matrix (K x N).</param>
    /// <param name="bias">Bias vector (N elements).</param>
    /// <param name="M">Batch size / rows.</param>
    /// <param name="N">Output features.</param>
    /// <param name="K">Input features.</param>
    /// <returns>Output buffer (M x N).</returns>
    IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K);

    /// <summary>
    /// Fused GEMM + Bias + GELU: output = GELU(A * B + bias)
    /// </summary>
    IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K);

    /// <summary>
    /// Fused GEMM + Bias + Sigmoid: output = Sigmoid(A * B + bias)
    /// </summary>
    IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K);

    /// <summary>
    /// Fused GEMM + Bias + Tanh: output = Tanh(A * B + bias)
    /// </summary>
    IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K);

    #endregion

    #region Element-wise Operations

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise subtraction: C = A - B
    /// </summary>
    void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise multiplication: C = A * B
    /// </summary>
    void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise division: C = A / B
    /// </summary>
    void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise minimum: C = min(A, B)
    /// </summary>
    void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise maximum: C = max(A, B)
    /// </summary>
    void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Scalar multiplication: B = A * scalar
    /// </summary>
    void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size);

    /// <summary>
    /// Power with scalar exponent: B = A ^ exponent
    /// </summary>
    void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size);

    /// <summary>
    /// Absolute value: B = abs(A)
    /// </summary>
    void Abs(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Exponential: B = exp(A)
    /// </summary>
    void Exp(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Base-2 exponential: B = 2^A
    /// </summary>
    void Exp2(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Base-10 exponential: B = 10^A
    /// </summary>
    void Exp10(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Expm1: B = exp(A) - 1
    /// </summary>
    void ExpM1(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Natural log: B = log(A)
    /// </summary>
    void Log(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Base-2 log: B = log2(A)
    /// </summary>
    void Log2(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Log1p: B = log(1 + A)
    /// </summary>
    void Log1P(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Square root: B = sqrt(A)
    /// </summary>
    void Sqrt(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Sign: B = sign(A)
    /// </summary>
    void Sign(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// ReLU activation: B = max(0, A)
    /// </summary>
    void Relu(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Sigmoid activation: B = 1 / (1 + exp(-A))
    /// </summary>
    void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Tanh activation: B = tanh(A)
    /// </summary>
    void Tanh(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// GELU activation: B = 0.5 * A * (1 + tanh(sqrt(2/pi) * (A + 0.044715 * A^3)))
    /// </summary>
    void Gelu(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Softmax activation along last dimension.
    /// </summary>
    void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features);

    #endregion

    #region Trigonometric Operations

    /// <summary>
    /// Sine: B = sin(A)
    /// </summary>
    void Sin(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Cosine: B = cos(A)
    /// </summary>
    void Cos(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Tangent: B = tan(A)
    /// </summary>
    void Tan(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Arc sine: B = asin(A)
    /// </summary>
    void Asin(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Arc cosine: B = acos(A)
    /// </summary>
    void Acos(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Arc tangent: B = atan(A)
    /// </summary>
    void Atan(IGpuBuffer A, IGpuBuffer B, int size);

    #endregion

    #region Hyperbolic Operations

    /// <summary>
    /// Hyperbolic sine: B = sinh(A)
    /// </summary>
    void Sinh(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Hyperbolic cosine: B = cosh(A)
    /// </summary>
    void Cosh(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Inverse hyperbolic sine: B = asinh(A)
    /// </summary>
    void Asinh(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Inverse hyperbolic cosine: B = acosh(A)
    /// </summary>
    void Acosh(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Inverse hyperbolic tangent: B = atanh(A)
    /// </summary>
    void Atanh(IGpuBuffer A, IGpuBuffer B, int size);

    #endregion

    #region Additional Unary Operations

    /// <summary>
    /// Reciprocal: B = 1/A
    /// </summary>
    void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Cube root: B = cbrt(A)
    /// </summary>
    void Cbrt(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Base-10 logarithm: B = log10(A)
    /// </summary>
    void Log10(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Negate: B = -A
    /// </summary>
    void Negate(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Floor: B = floor(A)
    /// </summary>
    void Floor(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Ceiling: B = ceil(A)
    /// </summary>
    void Ceiling(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Round: B = round(A)
    /// </summary>
    void Round(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>
    /// Truncate: B = trunc(A)
    /// </summary>
    void Truncate(IGpuBuffer A, IGpuBuffer B, int size);

    #endregion

    #region Sparse Operations (2:4 Structured Sparsity)

    /// <summary>
    /// Enforces 2:4 structured sparsity on a dense matrix.
    /// Every group of 4 consecutive elements will have exactly 2 zeros (the smallest magnitude values).
    /// </summary>
    /// <param name="denseInput">Dense input matrix (M x K).</param>
    /// <param name="sparseValues">Output buffer for compressed values (M x K/2).</param>
    /// <param name="sparseIndices">Output buffer for packed indices (M x K/4 bytes).</param>
    /// <param name="M">Number of rows.</param>
    /// <param name="K">Number of columns (must be divisible by 4).</param>
    void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K);

    /// <summary>
    /// Decompresses a 2:4 sparse matrix back to dense format.
    /// </summary>
    /// <param name="sparseValues">Compressed values (M x K/2).</param>
    /// <param name="sparseIndices">Packed indices (M x K/4 bytes).</param>
    /// <param name="denseOutput">Output dense matrix (M x K).</param>
    /// <param name="M">Number of rows.</param>
    /// <param name="K">Number of columns.</param>
    void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K);

    /// <summary>
    /// Sparse GEMM with 2:4 structured sparsity: C = alpha * A_sparse * B + beta * C
    /// Provides up to 2x speedup over dense GEMM for 50% sparse matrices.
    /// </summary>
    /// <param name="sparseAValues">Compressed A values (M x K/2).</param>
    /// <param name="sparseAIndices">Packed A indices (M x K/4 bytes).</param>
    /// <param name="B">Dense B matrix (K x N).</param>
    /// <param name="C">Output C matrix (M x N).</param>
    /// <param name="M">Rows of A and C.</param>
    /// <param name="N">Columns of B and C.</param>
    /// <param name="K">Columns of A and rows of B (original dense dimension).</param>
    /// <param name="alpha">Scalar for A*B.</param>
    /// <param name="beta">Scalar for C.</param>
    void SparseGemm(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f);

    /// <summary>
    /// Fused sparse GEMM with bias and ReLU: output = ReLU(A_sparse * B + bias)
    /// </summary>
    IGpuBuffer SparseGemmBiasRelu(
        IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices,
        IGpuBuffer B, IGpuBuffer bias,
        int M, int N, int K);

    /// <summary>
    /// Allocates a byte buffer for sparse indices (1 byte per group of 4 elements).
    /// </summary>
    /// <param name="size">Number of bytes.</param>
    /// <returns>GPU buffer for byte data.</returns>
    IGpuBuffer AllocateByteBuffer(int size);

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Sum all elements in buffer.
    /// </summary>
    float Sum(IGpuBuffer A, int size);

    /// <summary>
    /// Find maximum element in buffer.
    /// </summary>
    float Max(IGpuBuffer A, int size);

    /// <summary>
    /// Sum along axis for batched data.
    /// </summary>
    void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize);

    #endregion

    #region Synchronization

    /// <summary>
    /// Waits for all GPU operations to complete.
    /// </summary>
    void Synchronize();

    #endregion

    #region Convolution Operations

    void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW);

    void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW);

    void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW);

    void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW);

    void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW);

    void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW);

    #endregion

    #region Pooling Operations

    void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW);

    void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW);

    void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad);

    void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad);

    void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width);
    void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width);
    void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth);

    #endregion

    #region Normalization Operations

    void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training);

    void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon);

    void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon);

    void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon);

    void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon);

    void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon);

    void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon);

    #endregion

    #region Dropout and Regularization

    void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training);
    void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate);

    #endregion

    #region Embedding Operations

    void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim);
    void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize);
    IGpuBuffer AllocateIntBuffer(int size);
    IGpuBuffer AllocateIntBuffer(int[] data);

    #endregion

    #region Attention Operations

    void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal);

    void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal);

    void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal);

    #endregion

    #region Transpose and Reshape

    void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols);
    void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols);
    void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation);
    void Copy(IGpuBuffer source, IGpuBuffer destination, int size);
    void Fill(IGpuBuffer buffer, float value, int size);

    #endregion

    #region Activation Gradients

    void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size);
    void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size);
    void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size);
    void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size);
    void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features);
    void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size);
    void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size);
    void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size);
    void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size);
    void Swish(IGpuBuffer A, IGpuBuffer B, int size);
    void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size);
    void Silu(IGpuBuffer A, IGpuBuffer B, int size);
    void Mish(IGpuBuffer A, IGpuBuffer B, int size);
    void Softplus(IGpuBuffer A, IGpuBuffer B, int size);
    void Hardswish(IGpuBuffer A, IGpuBuffer B, int size);

    #endregion

    #region Loss Functions

    float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses);
    void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses);
    float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size);
    void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size);
    float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size);
    void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size);
    float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta);
    void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta);

    #endregion

    #region Gradient Clipping and Utility

    void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size);
    float L2Norm(IGpuBuffer A, int size);
    void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size);
    void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size);
    void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size);
    void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize);
    void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize);

    #endregion

    #region Comparison Operations

    void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    #endregion

    #region Statistics

    void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize);
    void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize);
    void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize);
    void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize);

    #endregion

    #region Optimizer Operations

    void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size);

    void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size);

    void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size);

    #endregion
}

/// <summary>
/// Handle to a GPU memory buffer.
/// </summary>
public interface IGpuBuffer : IDisposable
{
    /// <summary>
    /// Gets the number of float elements in the buffer.
    /// </summary>
    int Size { get; }

    /// <summary>
    /// Gets the size in bytes.
    /// </summary>
    long SizeInBytes { get; }

    /// <summary>
    /// Gets the native handle (IntPtr to OpenCL cl_mem or CUDA CUdeviceptr).
    /// </summary>
    IntPtr Handle { get; }
}
