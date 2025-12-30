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
    /// Element-wise multiplication: C = A * B
    /// </summary>
    void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Scalar multiplication: B = A * scalar
    /// </summary>
    void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size);

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
