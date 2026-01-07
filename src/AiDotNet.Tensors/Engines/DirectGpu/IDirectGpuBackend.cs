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

    /// <summary>
    /// Copies data between GPU buffers.
    /// </summary>
    /// <param name="source">Source buffer.</param>
    /// <param name="srcOffset">Source offset in elements.</param>
    /// <param name="destination">Destination buffer.</param>
    /// <param name="destOffset">Destination offset in elements.</param>
    /// <param name="size">Number of elements to copy.</param>
    void Copy(IGpuBuffer source, int srcOffset, IGpuBuffer destination, int destOffset, int size);

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

    /// <summary>
    /// Fused GEMM + Bias (no activation): output = A * B + bias
    /// </summary>
    /// <param name="A">Input matrix (M x K).</param>
    /// <param name="B">Weight matrix (K x N).</param>
    /// <param name="bias">Bias vector (N elements), broadcast across all M rows.</param>
    /// <param name="M">Batch size / rows.</param>
    /// <param name="N">Output features.</param>
    /// <param name="K">Input features.</param>
    /// <returns>Output buffer (M x N).</returns>
    IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K);

    #endregion

    #region Broadcast Operations

    /// <summary>
    /// Adds a bias vector to each row of a matrix: C[i,j] = A[i,j] + bias[j]
    /// </summary>
    /// <param name="A">Input matrix (M x N).</param>
    /// <param name="bias">Bias vector (N elements).</param>
    /// <param name="C">Output matrix (M x N).</param>
    /// <param name="M">Number of rows (batch size).</param>
    /// <param name="N">Number of columns (features).</param>
    void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N);

    /// <summary>
    /// Adds bias to Conv2D output in NCHW format.
    /// Operation: output[b, c, h, w] += bias[c]
    /// </summary>
    /// <param name="output">Output buffer to modify in-place [batch, channels, height, width].</param>
    /// <param name="bias">Bias vector [channels].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="spatialSize">Height * Width (spatial dimensions).</param>
    void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize);

    /// <summary>
    /// Broadcast multiply along last axis: C[i,j] = A[i,j] * B[j]
    /// where A has shape (outerSize, innerSize) and B has shape (innerSize).
    /// </summary>
    /// <param name="A">Input buffer (outerSize * innerSize elements).</param>
    /// <param name="B">Broadcast buffer (innerSize elements).</param>
    /// <param name="C">Output buffer (outerSize * innerSize elements).</param>
    /// <param name="outerSize">Number of outer elements (rows).</param>
    /// <param name="innerSize">Number of inner elements (columns / broadcast dimension).</param>
    void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize);

    /// <summary>
    /// Broadcast multiply along first axis: C[i,j] = A[i,j] * B[i]
    /// where A has shape (outerSize, innerSize) and B has shape (outerSize).
    /// </summary>
    /// <param name="A">Input buffer (outerSize * innerSize elements).</param>
    /// <param name="B">Broadcast buffer (outerSize elements).</param>
    /// <param name="C">Output buffer (outerSize * innerSize elements).</param>
    /// <param name="outerSize">Number of outer elements (rows / broadcast dimension).</param>
    /// <param name="innerSize">Number of inner elements (columns).</param>
    void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize);

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

    /// <summary>
    /// Squash activation for capsule networks: output = ||v||² / (1 + ||v||²) × v / ||v||
    /// </summary>
    void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon);

    /// <summary>
    /// Backward pass for squash activation.
    /// </summary>
    void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon);

    /// <summary>
    /// Tile tensor along batch dimension (axis 0).
    /// Input: [1, innerSize], Output: [repeats, innerSize]
    /// </summary>
    void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize);

    /// <summary>
    /// Tile tensor along any axis.
    /// outerSize = product of dimensions before axis
    /// axisSize = dimension at axis (original)
    /// innerSize = product of dimensions after axis
    /// </summary>
    void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats);

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

    #region CSR Sparse Operations (General Sparsity)

    /// <summary>
    /// Sparse Matrix-Dense Matrix multiplication (SpMM) in CSR format: C = A_csr * B
    /// Efficient for general sparse matrices like graph adjacency matrices.
    /// </summary>
    /// <param name="csrValues">Non-zero values of sparse A [nnz].</param>
    /// <param name="csrColIndices">Column indices for each value [nnz] (int32).</param>
    /// <param name="csrRowPointers">Row pointers [M+1] (int32).</param>
    /// <param name="denseB">Dense matrix B [K x N].</param>
    /// <param name="output">Output matrix C [M x N].</param>
    /// <param name="M">Rows of A and C.</param>
    /// <param name="K">Columns of A (rows of B).</param>
    /// <param name="N">Columns of B and C.</param>
    /// <param name="nnz">Number of non-zero elements in A.</param>
    void CsrSpMM(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer output,
        int M, int K, int N, int nnz);

    /// <summary>
    /// Fused CSR SpMM with bias addition: C = A_csr * B + bias
    /// </summary>
    /// <param name="csrValues">Non-zero values of sparse A [nnz].</param>
    /// <param name="csrColIndices">Column indices for each value [nnz] (int32).</param>
    /// <param name="csrRowPointers">Row pointers [M+1] (int32).</param>
    /// <param name="denseB">Dense matrix B [K x N].</param>
    /// <param name="bias">Bias vector [N] to add to each row.</param>
    /// <param name="output">Output matrix C [M x N].</param>
    /// <param name="M">Rows of A and C.</param>
    /// <param name="K">Columns of A (rows of B).</param>
    /// <param name="N">Columns of B and C.</param>
    /// <param name="nnz">Number of non-zero elements in A.</param>
    void CsrSpMMBias(
        IGpuBuffer csrValues,
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer denseB,
        IGpuBuffer bias,
        IGpuBuffer output,
        int M, int K, int N, int nnz);

    /// <summary>
    /// Scatter-add operation for graph aggregation: output[target[i]] += values[i] * input[source[i]]
    /// Efficient for edge-based message passing in GNNs.
    /// </summary>
    /// <param name="input">Input node features [numNodes x features].</param>
    /// <param name="sourceIndices">Source node indices [numEdges] (int32).</param>
    /// <param name="targetIndices">Target node indices [numEdges] (int32).</param>
    /// <param name="edgeValues">Edge weights [numEdges]. Can be null for unweighted graphs.</param>
    /// <param name="output">Output aggregated features [numNodes x features]. Must be zero-initialized.</param>
    /// <param name="numNodes">Number of nodes.</param>
    /// <param name="numEdges">Number of edges.</param>
    /// <param name="features">Number of features per node.</param>
    void ScatterAddEdges(
        IGpuBuffer input,
        IGpuBuffer sourceIndices,
        IGpuBuffer targetIndices,
        IGpuBuffer? edgeValues,
        IGpuBuffer output,
        int numNodes, int numEdges, int features);

    /// <summary>
    /// CSR-based max aggregation: output[i, f] = max over neighbors j of input[j, f]
    /// Used for graph neural network max pooling over neighbors.
    /// </summary>
    /// <param name="csrColIndices">Column indices (neighbor indices) [nnz] (int32).</param>
    /// <param name="csrRowPointers">Row pointers [M+1] (int32).</param>
    /// <param name="input">Input features [K x N] where K is number of source nodes.</param>
    /// <param name="output">Output features [M x N].</param>
    /// <param name="M">Number of target nodes (rows).</param>
    /// <param name="K">Number of source nodes.</param>
    /// <param name="N">Number of features.</param>
    void CsrSegmentedMax(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N);

    /// <summary>
    /// CSR-based min aggregation: output[i, f] = min over neighbors j of input[j, f]
    /// Used for graph neural network min pooling over neighbors.
    /// </summary>
    /// <param name="csrColIndices">Column indices (neighbor indices) [nnz] (int32).</param>
    /// <param name="csrRowPointers">Row pointers [M+1] (int32).</param>
    /// <param name="input">Input features [K x N] where K is number of source nodes.</param>
    /// <param name="output">Output features [M x N].</param>
    /// <param name="M">Number of target nodes (rows).</param>
    /// <param name="K">Number of source nodes.</param>
    /// <param name="N">Number of features.</param>
    void CsrSegmentedMin(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N);

    /// <summary>
    /// CSR-based standard deviation aggregation: output[i, f] = stddev over neighbors j of input[j, f]
    /// Computes sqrt(variance) per segment for graph neural networks.
    /// </summary>
    /// <param name="csrColIndices">Column indices (neighbor indices) [nnz] (int32).</param>
    /// <param name="csrRowPointers">Row pointers [M+1] (int32).</param>
    /// <param name="input">Input features [K x N] where K is number of source nodes.</param>
    /// <param name="output">Output features [M x N].</param>
    /// <param name="M">Number of target nodes (rows).</param>
    /// <param name="K">Number of source nodes.</param>
    /// <param name="N">Number of features.</param>
    /// <param name="epsilon">Small value for numerical stability (default 1e-8).</param>
    void CsrSegmentedStdDev(
        IGpuBuffer csrColIndices,
        IGpuBuffer csrRowPointers,
        IGpuBuffer input,
        IGpuBuffer output,
        int M, int K, int N,
        float epsilon = 1e-8f);

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

    /// <summary>
    /// Performs locally connected 2D convolution where each spatial position has unique weights.
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth].</param>
    /// <param name="weights">Weights tensor [outH, outW, outChannels, inChannels, kernelH, kernelW].</param>
    /// <param name="bias">Optional bias tensor [outChannels].</param>
    /// <param name="output">Output tensor [batch, outChannels, outHeight, outWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="kernelH">Kernel height.</param>
    /// <param name="kernelW">Kernel width.</param>
    /// <param name="strideH">Stride in height dimension.</param>
    /// <param name="strideW">Stride in width dimension.</param>
    void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW);

    /// <summary>
    /// Backward pass for locally connected conv2d - computes input gradients.
    /// </summary>
    void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW);

    /// <summary>
    /// Backward pass for locally connected conv2d - computes weight gradients.
    /// </summary>
    void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW);

    /// <summary>
    /// Backward pass for locally connected conv2d - computes bias gradients.
    /// </summary>
    void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth);

    /// <summary>
    /// Performs deformable 2D convolution with learnable offsets (DCNv1/v2).
    /// </summary>
    /// <param name="input">Input tensor [batch, inChannels, inHeight, inWidth].</param>
    /// <param name="weights">Weights tensor [outChannels, inChannels/groups, kernelH, kernelW].</param>
    /// <param name="offsets">Offsets tensor [batch, 2*kernelH*kernelW*deformGroups, outH, outW].</param>
    /// <param name="mask">Optional modulation mask [batch, kernelH*kernelW*deformGroups, outH, outW] (for DCNv2).</param>
    /// <param name="output">Output tensor [batch, outChannels, outHeight, outWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="kernelH">Kernel height.</param>
    /// <param name="kernelW">Kernel width.</param>
    /// <param name="strideH">Stride in height dimension.</param>
    /// <param name="strideW">Stride in width dimension.</param>
    /// <param name="padH">Padding in height dimension.</param>
    /// <param name="padW">Padding in width dimension.</param>
    /// <param name="dilationH">Dilation in height dimension.</param>
    /// <param name="dilationW">Dilation in width dimension.</param>
    /// <param name="groups">Number of convolution groups.</param>
    /// <param name="deformGroups">Number of deformable groups.</param>
    void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups);

    /// <summary>
    /// Backward pass for deformable conv2d - computes input gradients.
    /// </summary>
    void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups);

    /// <summary>
    /// Backward pass for deformable conv2d - computes weight gradients.
    /// </summary>
    void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups);

    /// <summary>
    /// Backward pass for deformable conv2d - computes offset gradients.
    /// </summary>
    void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups);

    /// <summary>
    /// Backward pass for deformable conv2d - computes mask gradients (for DCNv2).
    /// </summary>
    void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups);

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

    /// <summary>
    /// Performs 3D max pooling on volumetric input data.
    /// Input/output are in NCDHW format (batch, channels, depth, height, width).
    /// </summary>
    /// <param name="input">Input buffer [batch * channels * inDepth * inHeight * inWidth].</param>
    /// <param name="output">Output buffer [batch * channels * outDepth * outHeight * outWidth].</param>
    /// <param name="indices">Optional output buffer for max indices [batch * channels * outDepth * outHeight * outWidth * 3] for backprop.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inDepth">Input depth.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outDepth">Output depth.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="kernelD">Pooling kernel depth.</param>
    /// <param name="kernelH">Pooling kernel height.</param>
    /// <param name="kernelW">Pooling kernel width.</param>
    /// <param name="strideD">Stride in depth dimension.</param>
    /// <param name="strideH">Stride in height dimension.</param>
    /// <param name="strideW">Stride in width dimension.</param>
    void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW);

    /// <summary>
    /// Backward pass for 3D max pooling (NCDHW format).
    /// Routes gradients back to the positions that had max values in the forward pass.
    /// </summary>
    /// <param name="gradOutput">Gradient of loss w.r.t. output [batch * channels * outDepth * outHeight * outWidth].</param>
    /// <param name="indices">Flat indices from forward pass [batch * channels * outDepth * outHeight * outWidth].</param>
    /// <param name="gradInput">Gradient w.r.t. input (output) [batch * channels * inDepth * inHeight * inWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inDepth">Input depth.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outDepth">Output depth.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth);

    #endregion

    #region Spatial Transformer Operations

    /// <summary>
    /// Generates an affine sampling grid for spatial transformation.
    /// Given a batch of 2x3 affine transformation matrices (theta), generates a grid of
    /// normalized coordinates [-1, 1] that can be used with GridSample.
    /// </summary>
    /// <param name="theta">Affine transformation matrices [batch, 2, 3] = [batch * 6] in row-major.</param>
    /// <param name="grid">Output sampling grid [batch, outputHeight, outputWidth, 2] = [batch * outputHeight * outputWidth * 2].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="outputHeight">Height of the output grid.</param>
    /// <param name="outputWidth">Width of the output grid.</param>
    void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth);

    /// <summary>
    /// Samples from input using a sampling grid with bilinear interpolation.
    /// Given an input tensor and a grid of sampling locations, produces an output by
    /// sampling the input at the specified grid locations using bilinear interpolation.
    /// </summary>
    /// <param name="input">Input tensor [batch, channels, inHeight, inWidth] in NCHW format.</param>
    /// <param name="grid">Sampling grid [batch, outHeight, outWidth, 2] with (x, y) coordinates in [-1, 1].</param>
    /// <param name="output">Output tensor [batch, channels, outHeight, outWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels; otherwise to edge pixels.</param>
    void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false);

    /// <summary>
    /// Backward pass for GridSample - computes gradients for input and grid.
    /// </summary>
    /// <param name="gradOutput">Gradient from upstream [batch, channels, outHeight, outWidth].</param>
    /// <param name="input">Original input from forward pass [batch, channels, inHeight, inWidth].</param>
    /// <param name="grid">Sampling grid from forward pass [batch, outHeight, outWidth, 2].</param>
    /// <param name="gradInput">Gradient with respect to input [batch, channels, inHeight, inWidth].</param>
    /// <param name="gradGrid">Gradient with respect to grid [batch, outHeight, outWidth, 2].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="paddingMode">Padding mode: 0=zeros, 1=border, 2=reflection.</param>
    /// <param name="alignCorners">If true, [-1, 1] maps to corner pixels.</param>
    void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false);

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

    /// <summary>
    /// Computes the backward pass for RMS normalization.
    /// </summary>
    /// <param name="gradOutput">Gradient from the next layer (batchSize x normalizedSize).</param>
    /// <param name="input">Original input from forward pass (batchSize x normalizedSize).</param>
    /// <param name="gamma">Scale parameters (normalizedSize).</param>
    /// <param name="saveRms">Saved RMS values from forward pass (batchSize).</param>
    /// <param name="gradInput">Output gradient with respect to input (batchSize x normalizedSize).</param>
    /// <param name="gradGamma">Output gradient with respect to gamma (normalizedSize).</param>
    /// <param name="batchSize">Number of samples in the batch.</param>
    /// <param name="normalizedSize">Size of the normalized dimension.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon);

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

    /// <summary>
    /// Memory-efficient FlashAttention with log-sum-exp statistics for backward pass.
    /// </summary>
    /// <param name="query">Query tensor buffer [batch * heads * seqQ * headDim].</param>
    /// <param name="key">Key tensor buffer [batch * heads * seqK * headDim].</param>
    /// <param name="value">Value tensor buffer [batch * heads * seqK * headDim].</param>
    /// <param name="output">Output buffer [batch * heads * seqQ * headDim].</param>
    /// <param name="softmaxStats">Log-sum-exp statistics [batch * heads * seqQ].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="seqQ">Query sequence length.</param>
    /// <param name="seqK">Key sequence length.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(headDim)).</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal);

    /// <summary>
    /// Backward pass for FlashAttention using recomputation for memory efficiency.
    /// </summary>
    void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal);

    /// <summary>
    /// Grouped Query Attention - multiple query heads share same KV heads.
    /// </summary>
    /// <param name="query">Query buffer [batch * numQHeads * seqQ * headDim].</param>
    /// <param name="key">Key buffer [batch * numKVHeads * seqK * headDim].</param>
    /// <param name="value">Value buffer [batch * numKVHeads * seqK * headDim].</param>
    /// <param name="output">Output buffer [batch * numQHeads * seqQ * headDim].</param>
    /// <param name="attentionWeights">Optional attention weights [batch * numQHeads * seqQ * seqK].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="numQHeads">Number of query heads.</param>
    /// <param name="numKVHeads">Number of key-value heads (numQHeads must be divisible by numKVHeads).</param>
    /// <param name="seqQ">Query sequence length.</param>
    /// <param name="seqK">Key sequence length.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="scale">Scaling factor.</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal);

    /// <summary>
    /// Backward pass for Grouped Query Attention.
    /// </summary>
    void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale);

    #endregion

    #region Transpose and Reshape

    void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols);
    void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols);
    void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation);
    void Copy(IGpuBuffer source, IGpuBuffer destination, int size);
    void Fill(IGpuBuffer buffer, float value, int size);

    /// <summary>
    /// Copies a 2D region from source to destination with different strides.
    /// Useful for concatenating features: dest[row, destColOffset:destColOffset+srcCols] = src[row, :]
    /// </summary>
    /// <param name="source">Source buffer [numRows x srcCols].</param>
    /// <param name="destination">Destination buffer [numRows x destTotalCols].</param>
    /// <param name="numRows">Number of rows to copy.</param>
    /// <param name="srcCols">Number of columns in source.</param>
    /// <param name="destTotalCols">Total columns in destination.</param>
    /// <param name="destColOffset">Column offset in destination where source data is copied.</param>
    void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows, int srcCols, int destTotalCols, int destColOffset);

    /// <summary>
    /// Performs nearest-neighbor upsampling on 2D spatial data.
    /// Each input pixel is replicated scaleFactor x scaleFactor times in the output.
    /// </summary>
    /// <param name="input">Input buffer [batchChannels x height x width].</param>
    /// <param name="output">Output buffer [batchChannels x (height*scaleFactor) x (width*scaleFactor)].</param>
    /// <param name="batchChannels">Combined batch and channel dimensions.</param>
    /// <param name="height">Input height.</param>
    /// <param name="width">Input width.</param>
    /// <param name="scaleFactor">Upsampling scale factor (applied to both height and width).</param>
    void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor);

    /// <summary>
    /// Performs 3D nearest-neighbor upsampling on volumetric data.
    /// Each voxel is replicated to fill a [scaleD x scaleH x scaleW] block.
    /// </summary>
    /// <param name="input">Input buffer in NCDHW format [batch * channels * inDepth * inHeight * inWidth].</param>
    /// <param name="output">Output buffer in NCDHW format [batch * channels * outDepth * outHeight * outWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inDepth">Input depth.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="scaleD">Upsampling factor for depth dimension.</param>
    /// <param name="scaleH">Upsampling factor for height dimension.</param>
    /// <param name="scaleW">Upsampling factor for width dimension.</param>
    void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW);

    /// <summary>
    /// Backward pass for 3D nearest neighbor upsampling.
    /// Accumulates gradients from each output voxel in a scale block back to the corresponding input voxel.
    /// </summary>
    /// <param name="gradOutput">Gradient w.r.t. output [batch * channels * outDepth * outHeight * outWidth].</param>
    /// <param name="gradInput">Gradient w.r.t. input (output) [batch * channels * inDepth * inHeight * inWidth].</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="inDepth">Input depth.</param>
    /// <param name="inHeight">Input height.</param>
    /// <param name="inWidth">Input width.</param>
    /// <param name="scaleD">Upsampling factor for depth dimension.</param>
    /// <param name="scaleH">Upsampling factor for height dimension.</param>
    /// <param name="scaleW">Upsampling factor for width dimension.</param>
    void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW);

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
    void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size);
    void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size);
    void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size);

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

    /// <summary>
    /// Computes the backward pass for scatter-add operation.
    /// The gradient of scatter-add is a gather operation that collects gradients from destination
    /// positions back to source positions.
    /// </summary>
    /// <param name="gradDestination">Gradient from the next layer at destination positions.</param>
    /// <param name="indices">Indices that were used in the forward scatter-add (same as forward).</param>
    /// <param name="gradSource">Output gradient with respect to source.</param>
    /// <param name="numIndices">Number of indices/source elements.</param>
    /// <param name="featureSize">Size of each feature vector being scattered.</param>
    void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource,
        int numIndices, int featureSize);

    void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize);

    #endregion

    #region Comparison Operations

    void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);
    void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>
    /// Element-wise not-equal comparison against a scalar: C[i] = (A[i] != scalar) ? 1.0f : 0.0f
    /// </summary>
    /// <param name="A">Input buffer.</param>
    /// <param name="C">Output buffer with 1.0f where not equal, 0.0f where equal.</param>
    /// <param name="scalar">Scalar value to compare against.</param>
    /// <param name="size">Number of elements.</param>
    void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size);

    #endregion

    #region Statistics

    void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize);
    void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize);
    void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize);
    void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize);

    /// <summary>
    /// Selects the top K largest values and their indices along the last axis.
    /// </summary>
    /// <param name="A">Input buffer with shape [outerSize, reduceSize] in row-major order.</param>
    /// <param name="values">Output buffer for top K values [outerSize, K].</param>
    /// <param name="indices">Output buffer for top K indices [outerSize, K] (int buffer).</param>
    /// <param name="outerSize">Number of rows (batch dimension).</param>
    /// <param name="reduceSize">Size of the axis to select from (columns).</param>
    /// <param name="k">Number of top elements to select.</param>
    /// <param name="sorted">If true, output is sorted in descending order.</param>
    void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true);

    /// <summary>
    /// Maximum reduction along axis: B[i] = max(A[i, :]).
    /// </summary>
    /// <param name="A">Input buffer with shape [outerSize, reduceSize] in row-major order.</param>
    /// <param name="B">Output buffer with shape [outerSize].</param>
    /// <param name="outerSize">Number of output elements (rows).</param>
    /// <param name="reduceSize">Size of the axis to reduce (columns).</param>
    void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize);

    /// <summary>
    /// ArgMax reduction along axis: B[i] = argmax(A[i, :]).
    /// Returns indices as floats.
    /// </summary>
    /// <param name="A">Input buffer with shape [outerSize, reduceSize].</param>
    /// <param name="indices">Output buffer with shape [outerSize].</param>
    /// <param name="outerSize">Number of rows.</param>
    /// <param name="reduceSize">Number of columns.</param>
    void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize);

    #endregion

    #region Optimizer Operations

    void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size);

    void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size);

    void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size);

    #endregion

    #region FFT and Signal Processing

    /// <summary>
    /// Computes 1D complex-to-complex FFT on GPU.
    /// </summary>
    /// <param name="inputReal">Real part of input [n elements].</param>
    /// <param name="inputImag">Imaginary part of input [n elements].</param>
    /// <param name="outputReal">Real part of output [n elements].</param>
    /// <param name="outputImag">Imaginary part of output [n elements].</param>
    /// <param name="n">FFT size (must be power of 2).</param>
    /// <param name="inverse">If true, compute inverse FFT.</param>
    void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse);

    /// <summary>
    /// Computes 1D real-to-complex FFT (RFFT) on GPU.
    /// </summary>
    /// <param name="input">Real input [n elements].</param>
    /// <param name="outputReal">Real part of output [n/2+1 elements].</param>
    /// <param name="outputImag">Imaginary part of output [n/2+1 elements].</param>
    /// <param name="n">Input size (must be power of 2).</param>
    void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n);

    /// <summary>
    /// Computes 1D complex-to-real inverse FFT (IRFFT) on GPU.
    /// </summary>
    /// <param name="inputReal">Real part of input [n/2+1 elements].</param>
    /// <param name="inputImag">Imaginary part of input [n/2+1 elements].</param>
    /// <param name="output">Real output [n elements].</param>
    /// <param name="n">Output size (must be power of 2).</param>
    void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n);

    /// <summary>
    /// Computes batched 1D FFT for multiple signals.
    /// </summary>
    /// <param name="inputReal">Real parts [batch * n elements].</param>
    /// <param name="inputImag">Imaginary parts [batch * n elements].</param>
    /// <param name="outputReal">Real parts of output [batch * n elements].</param>
    /// <param name="outputImag">Imaginary parts of output [batch * n elements].</param>
    /// <param name="batch">Number of signals.</param>
    /// <param name="n">FFT size per signal.</param>
    /// <param name="inverse">If true, compute inverse FFT.</param>
    void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse);

    /// <summary>
    /// Computes 2D FFT on GPU.
    /// </summary>
    /// <param name="inputReal">Real part of input [height * width elements].</param>
    /// <param name="inputImag">Imaginary part of input [height * width elements].</param>
    /// <param name="outputReal">Real part of output [height * width elements].</param>
    /// <param name="outputImag">Imaginary part of output [height * width elements].</param>
    /// <param name="height">Height dimension.</param>
    /// <param name="width">Width dimension.</param>
    /// <param name="inverse">If true, compute inverse 2D FFT.</param>
    void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse);

    /// <summary>
    /// Applies a window function element-wise.
    /// </summary>
    /// <param name="input">Input signal.</param>
    /// <param name="window">Window function coefficients.</param>
    /// <param name="output">Windowed output.</param>
    /// <param name="n">Signal length.</param>
    void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n);

    /// <summary>
    /// Computes magnitude from complex values: sqrt(real² + imag²).
    /// </summary>
    /// <param name="real">Real part.</param>
    /// <param name="imag">Imaginary part.</param>
    /// <param name="magnitude">Output magnitude.</param>
    /// <param name="n">Number of elements.</param>
    void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n);

    /// <summary>
    /// Computes phase from complex values: atan2(imag, real).
    /// </summary>
    /// <param name="real">Real part.</param>
    /// <param name="imag">Imaginary part.</param>
    /// <param name="phase">Output phase in radians.</param>
    /// <param name="n">Number of elements.</param>
    void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n);

    /// <summary>
    /// Converts polar coordinates (magnitude, phase) to complex (real, imag).
    /// </summary>
    /// <param name="magnitude">Magnitude values.</param>
    /// <param name="phase">Phase values in radians.</param>
    /// <param name="real">Output real part.</param>
    /// <param name="imag">Output imaginary part.</param>
    /// <param name="n">Number of elements.</param>
    void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n);

    /// <summary>
    /// Applies Mel filterbank to power spectrogram.
    /// </summary>
    /// <param name="powerSpec">Power spectrogram [numFrames * numFreqs].</param>
    /// <param name="filterbank">Mel filterbank matrix [nMels * numFreqs].</param>
    /// <param name="melSpec">Output Mel spectrogram [numFrames * nMels].</param>
    /// <param name="numFrames">Number of time frames.</param>
    /// <param name="numFreqs">Number of frequency bins (nFft/2+1).</param>
    /// <param name="nMels">Number of Mel bands.</param>
    void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels);

    /// <summary>
    /// Converts power spectrogram to decibel scale.
    /// </summary>
    /// <param name="power">Power spectrogram.</param>
    /// <param name="db">Output in decibels.</param>
    /// <param name="n">Number of elements.</param>
    /// <param name="refValue">Reference value for 0 dB.</param>
    /// <param name="minDb">Minimum dB value (floor).</param>
    void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb);

    /// <summary>
    /// Converts decibel values back to power.
    /// </summary>
    /// <param name="db">Input in decibels.</param>
    /// <param name="power">Output power spectrogram.</param>
    /// <param name="n">Number of elements.</param>
    /// <param name="refValue">Reference value that was used for 0 dB.</param>
    void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue);

    #endregion

    #region Random Number Generation

    /// <summary>
    /// Generates uniformly distributed random numbers on GPU.
    /// </summary>
    /// <param name="output">Output buffer [size].</param>
    /// <param name="size">Number of elements.</param>
    /// <param name="min">Minimum value (inclusive).</param>
    /// <param name="max">Maximum value (exclusive).</param>
    /// <param name="seed">Random seed.</param>
    void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed);

    /// <summary>
    /// Generates normally distributed (Gaussian) random numbers on GPU using Box-Muller transform.
    /// </summary>
    /// <param name="output">Output buffer [size].</param>
    /// <param name="size">Number of elements.</param>
    /// <param name="mean">Mean of the distribution.</param>
    /// <param name="stdDev">Standard deviation of the distribution.</param>
    /// <param name="seed">Random seed.</param>
    void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed);

    #endregion

    #region Specialized Layer Operations

    /// <summary>
    /// Computes RBF kernel: exp(-epsilon * ||x - c||^2)
    /// </summary>
    /// <param name="input">Input buffer [batch * inputDim].</param>
    /// <param name="centers">Centers buffer [numCenters * inputDim].</param>
    /// <param name="epsilons">Epsilons buffer [numCenters].</param>
    /// <param name="output">Output buffer [batch * numCenters].</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="numCenters">Number of centers.</param>
    /// <param name="inputDim">Input dimension.</param>
    void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output,
        int batchSize, int numCenters, int inputDim);

    /// <summary>
    /// Updates weights using STDP learning rule.
    /// </summary>
    /// <param name="weights">Weights buffer [numPre * numPost].</param>
    /// <param name="preTrace">Presynaptic trace buffer [numPre].</param>
    /// <param name="postTrace">Postsynaptic trace buffer [numPost].</param>
    /// <param name="preSpike">Presynaptic spike buffer [numPre].</param>
    /// <param name="postSpike">Postsynaptic spike buffer [numPost].</param>
    /// <param name="ltpRate">LTP learning rate.</param>
    /// <param name="ltdRate">LTD learning rate.</param>
    /// <param name="homeostasisRate">Homeostasis rate.</param>
    /// <param name="minWeight">Minimum weight.</param>
    /// <param name="maxWeight">Maximum weight.</param>
    /// <param name="numPre">Number of presynaptic neurons.</param>
    /// <param name="numPost">Number of postsynaptic neurons.</param>
    void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace,
        IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate,
        float minWeight, float maxWeight,
        int numPre, int numPost);

    /// <summary>
    /// Updates traces and detects spikes on GPU.
    /// </summary>
    /// <param name="traces">Traces buffer (in/out) [size].</param>
    /// <param name="spikes">Spikes buffer (out) [size].</param>
    /// <param name="input">Input buffer [size].</param>
    /// <param name="decay">Trace decay factor.</param>
    /// <param name="threshold">Spike threshold.</param>
    /// <param name="size">Number of neurons.</param>
    void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input,
        float decay, float threshold, int size);

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
