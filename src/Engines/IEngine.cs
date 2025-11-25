using AiDotNet.LinearAlgebra;

namespace AiDotNet.Engines;

/// <summary>
/// Execution engine for mathematical operations.
/// Implementations can target CPU, GPU, or other accelerators.
/// </summary>
/// <remarks>
/// <para>
/// The IEngine interface provides a pluggable execution model for AiDotNet.
/// By swapping implementations, users can transparently accelerate computations
/// on different hardware without changing their code.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "compute backend".
///
/// - CpuEngine: Runs operations on your CPU using standard C# code
/// - GpuEngine: Runs operations on your GPU for massive speedups
/// - Future: TPU, distributed computing, etc.
///
/// Your code stays the same - just swap the engine to change where it runs!
/// </para>
/// <para><b>DEPRECATED:</b> This interface is being migrated to AiDotNet.Tensors.Engines.IEngine.
/// New projects should use the AiDotNet.Tensors package for GPU/CPU acceleration.
/// This interface will be removed in a future version.</para>
/// </remarks>
[Obsolete("Use AiDotNet.Tensors.Engines.IEngine from the AiDotNet.Tensors package instead. This interface will be removed in a future version.")]
public interface IEngine
{
    /// <summary>
    /// Gets the name of this engine.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this engine supports GPU acceleration.
    /// </summary>
    bool SupportsGpu { get; }

    #region Vector Operations

    /// <summary>
    /// Adds two vectors element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Add<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Subtracts vector b from vector a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Subtract<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Multiplies two vectors element-wise (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Multiply<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Multiplies a vector by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new vector with all elements multiplied by the scalar.</returns>
    Vector<T> Multiply<T>(Vector<T> vector, T scalar);

    /// <summary>
    /// Divides vector a by vector b element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The numerator vector.</param>
    /// <param name="b">The denominator vector.</param>
    /// <returns>A new vector containing the element-wise quotient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <exception cref="DivideByZeroException">Thrown when any element of b is zero.</exception>
    Vector<T> Divide<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Divides a vector by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to divide.</param>
    /// <param name="scalar">The scalar divisor.</param>
    /// <returns>A new vector with all elements divided by the scalar.</returns>
    /// <exception cref="DivideByZeroException">Thrown when scalar is zero.</exception>
    Vector<T> Divide<T>(Vector<T> vector, T scalar);

    /// <summary>
    /// Computes the square root of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the square roots.</returns>
    Vector<T> Sqrt<T>(Vector<T> vector);

    /// <summary>
    /// Raises each element of the vector to the specified power.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="exponent">The exponent to raise elements to.</param>
    /// <returns>A new vector with elements raised to the power.</returns>
    Vector<T> Power<T>(Vector<T> vector, T exponent);

    /// <summary>
    /// Computes the element-wise maximum of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is max(a[i], b[i]).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for AdaMax optimizer.</para>
    /// </remarks>
    Vector<T> Max<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise minimum of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is min(a[i], b[i]).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for various optimizers.</para>
    /// </remarks>
    Vector<T> Min<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the absolute value of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the absolute values.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for AdaMax and other optimizers.</para>
    /// </remarks>
    Vector<T> Abs<T>(Vector<T> vector);

    /// <summary>
    /// Computes the exponential (e^x) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the exponentials.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for natural gradient optimizers.</para>
    /// </remarks>
    Vector<T> Exp<T>(Vector<T> vector);

    /// <summary>
    /// Computes the natural logarithm of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the logarithms.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for natural gradient optimizers.</para>
    /// <para>
    /// <b>Note:</b> For elements <= 0, the behavior is:
    /// - Zero input produces NegativeInfinity
    /// - Negative input produces NaN
    /// - No exception is thrown (silent NaN propagation)
    /// </para>
    /// </remarks>
    Vector<T> Log<T>(Vector<T> vector);

    /// <summary>
    /// Computes the sign (-1, 0, or +1) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the signs.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for Lion optimizer.</para>
    /// </remarks>
    Vector<T> Sign<T>(Vector<T> vector);

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The sum of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Reduction operation that sums all elements: result = v[0] + v[1] + ... + v[n-1].
    /// Critical for computing totals, norms, and other aggregate statistics.
    /// CPU implementation uses parallel reduction for large vectors.
    /// GPU implementation uses warp-level reduction primitives for maximum efficiency.
    /// </para>
    /// </remarks>
    T Sum<T>(Vector<T> vector);

    /// <summary>
    /// Computes the dot product (inner product) of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The dot product of the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Computes result = sum(a[i] * b[i]) for all i.
    /// Fundamental operation in linear algebra used for:
    /// - Computing similarities and distances
    /// - Matrix-vector products (each row dot product with vector)
    /// - Neural network forward/backward passes
    /// - ARIMA/time series predictions
    /// </para>
    /// <para>
    /// CPU implementation uses SIMD and parallel reduction.
    /// GPU implementation uses warp-level primitives for maximum throughput.
    /// This is one of the most performance-critical operations in deep learning.
    /// </para>
    /// </remarks>
    T DotProduct<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the mean (average) of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The mean of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Computes result = sum(v[i]) / length.
    /// Equivalent to Sum(vector) divided by vector length, but may use optimized implementations.
    /// Used extensively in statistics, normalization, and time series analysis.
    /// </para>
    /// </remarks>
    T Mean<T>(Vector<T> vector);

    /// <summary>
    /// Creates a vector filled with a constant value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the vector to create.</param>
    /// <param name="value">The value to fill all elements with.</param>
    /// <returns>A new vector with all elements set to the specified value.</returns>
    Vector<T> Fill<T>(int length, T value);

    /// <summary>
    /// Creates a vector filled with zeros.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the vector to create.</param>
    /// <returns>A new vector with all elements set to zero.</returns>
    Vector<T> FillZero<T>(int length);

    /// <summary>
    /// Generates a dropout mask where each element is either zero or a scale value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the mask vector to create.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0 to 1).</param>
    /// <param name="scale">Scale value for kept elements.</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <returns>A new vector containing the dropout mask.</returns>
    Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null);

    /// <summary>
    /// Copies elements from a vector to a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="source">The source vector.</param>
    /// <param name="destination">The destination tensor.</param>
    void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination);
    /// <summary>
    /// Generates Gaussian random noise using the Box-Muller transform.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the noise vector to create.</param>
    /// <param name="mean">The mean of the Gaussian distribution.</param>
    /// <param name="standardDeviation">The standard deviation of the Gaussian distribution.</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <returns>A new vector containing Gaussian random noise.</returns>
    Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null);

    #endregion

    #region Activation Functions

    /// <summary>
    /// Computes the hyperbolic tangent of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing tanh values between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tanh activation function: tanh(x) = (e^x - e^-x) / (e^x + e^-x).
    /// Commonly used in hidden layers of neural networks.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6× speedup for float).
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Vector<T> Tanh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the sigmoid function of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing sigmoid values between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Sigmoid activation function: σ(x) = 1 / (1 + e^-x).
    /// Commonly used for binary classification and gate functions in LSTMs/GRUs.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6× speedup for float).
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Vector<T> Sigmoid<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Rectified Linear Unit (ReLU) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector where each element is max(0, x).</returns>
    /// <remarks>
    /// <para>
    /// ReLU activation function: ReLU(x) = max(0, x).
    /// Most commonly used activation in modern deep learning.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Vector<T> ReLU<T>(Vector<T> vector);

    /// <summary>
    /// Computes the hyperbolic tangent of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor containing tanh values between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of Tanh for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Tensor<T> Tanh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the sigmoid function of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor containing sigmoid values between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of Sigmoid for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Tensor<T> Sigmoid<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the ReLU of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor where each element is max(0, x).</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of ReLU for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses ILGPU kernels.
    /// </para>
    /// </remarks>
    Tensor<T> ReLU<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the GELU (Gaussian Error Linear Unit) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with GELU activation applied.</returns>
    /// <remarks>
    /// <para>
    /// GELU activation: x * Φ(x) where Φ is the standard Gaussian cumulative distribution.
    /// Commonly used in transformers (BERT, GPT) and modern architectures.
    /// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// </para>
    /// </remarks>
    Vector<T> GELU<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Mish activation of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with Mish activation applied.</returns>
    /// <remarks>
    /// <para>
    /// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// Smooth, self-regularizing activation function with better performance than ReLU in some tasks.
    /// </para>
    /// </remarks>
    Vector<T> Mish<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Swish/SiLU activation of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with Swish activation applied.</returns>
    /// <remarks>
    /// <para>
    /// Swish/SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
    /// Used in EfficientNet and other modern architectures. Self-gated activation.
    /// </para>
    /// </remarks>
    Vector<T> Swish<T>(Vector<T> vector);

    /// <summary>
    /// Computes the ELU (Exponential Linear Unit) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="alpha">Scale factor for negative values (default 1.0).</param>
    /// <returns>A new vector with ELU activation applied.</returns>
    /// <remarks>
    /// <para>
    /// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// Helps with vanishing gradient problem and can produce negative outputs.
    /// </para>
    /// </remarks>
    Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0);

    /// <summary>
    /// Computes the GELU of each element in the tensor.
    /// </summary>
    Tensor<T> GELU<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the Mish activation of each element in the tensor.
    /// </summary>
    Tensor<T> Mish<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the Swish/SiLU activation of each element in the tensor.
    /// </summary>
    Tensor<T> Swish<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the ELU of each element in the tensor.
    /// </summary>
    Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0);

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <summary>
    /// Performs matrix-matrix multiplication (GEMM: General Matrix Multiply).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix (M x K).</param>
    /// <param name="b">The second matrix (K x N).</param>
    /// <returns>The product matrix (M x N).</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-007: GEMM</b></para>
    /// <para>
    /// Matrix multiplication is O(n³) - highly computationally intensive.
    /// GPU acceleration provides 100-1000x speedup for large matrices.
    /// Essential for dense neural network layers.
    /// </para>
    /// </remarks>
    Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Performs matrix-vector multiplication (GEMV).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="matrix">The matrix (M x N).</param>
    /// <param name="vector">The vector (N elements).</param>
    /// <returns>The result vector (M elements).</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-008: GEMV</b></para>
    /// <para>
    /// Computes result[i] = sum(matrix[i, j] * vector[j]) for all i.
    /// Critical for neural network inference.
    /// </para>
    /// </remarks>
    Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector);

    /// <summary>
    /// Transposes a matrix (rows become columns).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The input matrix (M x N).</param>
    /// <returns>The transposed matrix (N x M).</returns>
    /// <remarks>
    /// <para><b>US-GPU-009: Matrix Transpose</b></para>
    /// <para>
    /// Required for backpropagation in neural networks.
    /// GPU implementation uses shared memory for coalesced access.
    /// </para>
    /// </remarks>
    Matrix<T> MatrixTranspose<T>(Matrix<T> matrix);

    /// <summary>
    /// Adds two matrices element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix.</param>
    /// <param name="b">The second matrix.</param>
    /// <returns>A new matrix containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="matrix">The matrix to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new matrix with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar);

    /// <summary>
    /// Subtracts matrix b from matrix a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix.</param>
    /// <param name="b">The second matrix.</param>
    /// <returns>A new matrix containing the element-wise difference (a - b).</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Computes the sum of squared elements of a matrix (used for Frobenius norm computation).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The input matrix.</param>
    /// <returns>The sum of all squared elements: sum_{i,j} matrix[i,j]^2</returns>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// <para>
    /// This is used to compute the squared Frobenius norm: ||A||_F^2 = sum_{i,j} A_{ij}^2
    /// To get the actual Frobenius norm, take sqrt of the result.
    /// </para>
    /// </remarks>
    T MatrixSumOfSquares<T>(Matrix<T> matrix);

    /// <summary>
    /// Swaps two columns in a matrix in-place using vectorized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The matrix to modify.</param>
    /// <param name="col1">The first column index.</param>
    /// <param name="col2">The second column index.</param>
    /// <remarks>
    /// GPU-accelerated column swapping for matrix decompositions.
    /// </remarks>
    void SwapColumns<T>(Matrix<T> matrix, int col1, int col2);

    /// <summary>
    /// Swaps two rows in a matrix in-place using vectorized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The matrix to modify.</param>
    /// <param name="row1">The first row index.</param>
    /// <param name="row2">The second row index.</param>
    /// <remarks>
    /// GPU-accelerated row swapping for matrix decompositions.
    /// </remarks>
    void SwapRows<T>(Matrix<T> matrix, int row1, int row2);

    /// <summary>
    /// Computes the outer product of two vectors: result[i,j] = a[i] * b[j].
    /// </summary>
    /// <typeparam name="T">The numeric type of vector elements.</typeparam>
    /// <param name="a">The first vector (length M).</param>
    /// <param name="b">The second vector (length N).</param>
    /// <returns>An M×N matrix containing the outer product.</returns>
    /// <remarks>
    /// GPU-accelerated outer product for SVD and other decompositions.
    /// </remarks>
    Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Extracts a column from a matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="columnIndex">The column index to extract.</param>
    /// <returns>A vector containing the column values.</returns>
    /// <remarks>
    /// GPU-accelerated column extraction.
    /// </remarks>
    Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex);

    /// <summary>
    /// Extracts a row from a matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="rowIndex">The row index to extract.</param>
    /// <returns>A vector containing the row values.</returns>
    /// <remarks>
    /// GPU-accelerated row extraction.
    /// </remarks>
    Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex);

    /// <summary>
    /// Sets a column in a matrix from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The target matrix.</param>
    /// <param name="columnIndex">The column index to set.</param>
    /// <param name="values">The vector of values to set.</param>
    /// <remarks>
    /// GPU-accelerated column setting.
    /// </remarks>
    void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values);

    /// <summary>
    /// Sets a row in a matrix from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The target matrix.</param>
    /// <param name="rowIndex">The row index to set.</param>
    /// <param name="values">The vector of values to set.</param>
    /// <remarks>
    /// GPU-accelerated row setting.
    /// </remarks>
    void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values);

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <summary>
    /// Performs batched matrix multiplication on 3D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor [B, M, K] - B batches of M×K matrices.</param>
    /// <param name="b">The second tensor [B, K, N] - B batches of K×N matrices.</param>
    /// <returns>The result tensor [B, M, N] - B batches of M×N matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-013: BatchMatMul</b></para>
    /// <para>
    /// Batched matrix multiplication performs C[i] = A[i] @ B[i] for all i in the batch.
    /// Critical for transformer models and attention mechanisms where multiple matrices
    /// must be multiplied in parallel.
    /// </para>
    /// <para>
    /// Input shapes:
    /// - a: [B, M, K] where B = batch size, M = rows, K = inner dimension
    /// - b: [B, K, N] where N = columns
    /// Output: [B, M, N]
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup by processing all batches in parallel.
    /// </para>
    /// </remarks>
    Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Performs result[i] = a[i] + b[i] for all elements.
    /// Both tensors must have identical shapes.
    /// GPU acceleration provides significant speedup for large tensors.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Subtracts tensor b from tensor a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies two tensors element-wise (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies a tensor by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new tensor with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Divides tensor a by tensor b element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The numerator tensor.</param>
    /// <param name="b">The denominator tensor.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <exception cref="DivideByZeroException">Thrown when any element of b is zero.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Performs 2D max pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (e.g., 2 for 2x2 pooling).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 4D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-012: MaxPool2D</b></para>
    /// <para>
    /// Max pooling downsamples the spatial dimensions by taking the maximum value
    /// in each pooling window. Commonly used in CNNs for:
    /// - Reducing spatial dimensions
    /// - Providing translation invariance
    /// - Reducing computation in deeper layers
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_height = floor((height + 2*padding - poolSize) / stride) + 1
    /// output_width = floor((width + 2*padding - poolSize) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 20-100x speedup for large feature maps.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 2D average pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (e.g., 2 for 2x2 pooling).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 4D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-012: AvgPool2D</b></para>
    /// <para>
    /// Average pooling downsamples the spatial dimensions by taking the average value
    /// in each pooling window. Often used as an alternative to max pooling for:
    /// - Smoother downsampling
    /// - Preserving more spatial information
    /// - Global average pooling before final classification layer
    /// </para>
    /// <para>
    /// GPU acceleration provides 20-100x speedup for large feature maps.
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 2D max pooling with asymmetric pool size and stride, returning max indices for backpropagation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="maxIndices">Output: indices of max elements for backpropagation [batch, channels, outH, outW, 2].</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <remarks>
    /// <para><b>US-GPU-012b: MaxPool2D with indices</b></para>
    /// <para>
    /// This overload supports asymmetric pooling parameters and returns max indices
    /// needed for gradient routing during backpropagation.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices);

    /// <summary>
    /// Computes the gradient of MaxPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output [batch, channels, outH, outW].</param>
    /// <param name="maxIndices">The max indices from forward pass [batch, channels, outH, outW, 2].</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW] used in forward pass.</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para><b>US-GPU-012c: MaxPool2D Backward</b></para>
    /// <para>
    /// Routes gradients only to the max positions identified during forward pass.
    /// GPU acceleration provides 20-100x speedup.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs 2D average pooling with asymmetric pool size and stride.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <remarks>
    /// <para><b>US-GPU-012d: AvgPool2D with asymmetric parameters</b></para>
    /// </remarks>
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride);

    /// <summary>
    /// Computes the gradient of AvgPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output [batch, channels, outH, outW].</param>
    /// <param name="inputShape">The shape of the original input [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW] used in forward pass.</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para><b>US-GPU-012e: AvgPool2D Backward</b></para>
    /// <para>
    /// Distributes gradients equally to all positions in each pooling window.
    /// GPU acceleration provides 20-100x speedup.
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs depthwise 2D convolution where each input channel is convolved independently.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, multiplier, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="padding">The padding [padH, padW].</param>
    /// <returns>The convolved tensor [batch, in_channels * multiplier, output_height, output_width].</returns>
    /// <remarks>
    /// <para><b>US-GPU-013: DepthwiseConv2D</b></para>
    /// <para>
    /// Depthwise convolution applies separate filters to each input channel.
    /// This is a key component of MobileNet and other efficient architectures.
    /// GPU acceleration provides 20-100x speedup.
    /// </para>
    /// </remarks>
    Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of DepthwiseConv2D with respect to the input.
    /// </summary>
    Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of DepthwiseConv2D with respect to the kernel.
    /// </summary>
    Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

    /// <summary>
    /// Performs 2D transposed convolution (deconvolution) for upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, out_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="padding">The padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding for size adjustment [outPadH, outPadW].</param>
    /// <returns>The upsampled tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: ConvTranspose2D</b></para>
    /// <para>
    /// Transposed convolution upsamples spatial dimensions, commonly used in:
    /// - GANs for image generation
    /// - Autoencoders for decoding
    /// - Semantic segmentation for dense predictions
    /// </para>
    /// </remarks>
    Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding);

    /// <summary>
    /// Computes the gradient of ConvTranspose2D with respect to the input.
    /// </summary>
    Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of ConvTranspose2D with respect to the kernel.
    /// </summary>
    Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

    /// <summary>
    /// Performs 2D convolution on a 4D input tensor using a 4D kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add to the input. Defaults to 0.</param>
    /// <param name="dilation">The spacing between kernel elements. Defaults to 1.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-011: Conv2D</b></para>
    /// <para>
    /// 2D convolution is the core operation in convolutional neural networks (CNNs).
    /// It applies learned filters to detect features like edges, textures, and patterns.
    /// Critical for:
    /// - Image classification (ResNet, VGG, etc.)
    /// - Object detection (YOLO, Faster R-CNN)
    /// - Semantic segmentation (U-Net, DeepLab)
    /// - Style transfer and image generation
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_height = floor((height + 2*padding - dilation*(kernel_height-1) - 1) / stride) + 1
    /// output_width = floor((width + 2*padding - dilation*(kernel_width-1) - 1) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup for typical CNN layers.
    /// This is the most computationally expensive operation in deep learning.
    /// </para>
    /// </remarks>
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Performs 2D convolution with asymmetric stride, padding, and dilation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-011b: Conv2D with asymmetric parameters</b></para>
    /// <para>
    /// This overload supports different stride/padding/dilation values for height and width dimensions,
    /// which is required by some network architectures (e.g., certain ResNet variants, asymmetric kernels).
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_height = floor((height + 2*padH - dilationH*(kernel_height-1) - 1) / strideH) + 1
    /// output_width = floor((width + 2*padW - dilationW*(kernel_width-1) - 1) / strideW) + 1
    /// </para>
    /// </remarks>
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, outH, outW].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="inputShape">The shape of the original input tensor [batch, in_channels, height, width].</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-011c: Conv2D Backward Input</b></para>
    /// <para>
    /// Computes ∂L/∂input given ∂L/∂output using transposed convolution (deconvolution).
    /// This is essential for backpropagation through convolutional layers.
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup, critical for training CNNs.
    /// </para>
    /// </remarks>
    Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the kernel (weights).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, outH, outW].</param>
    /// <param name="input">The original input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernelShape">The shape of the kernel [out_channels, in_channels, kernelH, kernelW].</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel.</returns>
    /// <remarks>
    /// <para><b>US-GPU-011d: Conv2D Backward Kernel</b></para>
    /// <para>
    /// Computes ∂L/∂kernel given ∂L/∂output using cross-correlation between input and gradient.
    /// This is essential for learning the convolutional filters during training.
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup, critical for training CNNs.
    /// </para>
    /// </remarks>
    Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Transposes a 2D tensor (matrix represented as tensor).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input 2D tensor to transpose.</param>
    /// <returns>The transposed tensor where rows become columns.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor is not 2D.</exception>
    /// <remarks>
    /// <para><b>Phase C: JIT Compilation Support</b></para>
    /// <para>
    /// Transposes a 2D tensor by swapping its dimensions. For a tensor with shape [M, N],
    /// the result has shape [N, M].
    /// </para>
    /// <para>
    /// GPU acceleration provides significant speedup for large tensors.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTranspose<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs matrix multiplication on two 2D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first 2D tensor with shape [M, N].</param>
    /// <param name="b">The second 2D tensor with shape [N, P].</param>
    /// <returns>The result tensor with shape [M, P].</returns>
    /// <exception cref="ArgumentException">Thrown when tensors are not 2D or inner dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>Phase C: JIT Compilation Support</b></para>
    /// <para>
    /// Performs standard matrix multiplication C = A x B where:
    /// - A has shape [M, N]
    /// - B has shape [N, P]
    /// - C has shape [M, P]
    /// </para>
    /// <para>
    /// This is distinct from BatchMatMul which handles batched operations on higher-dimensional tensors.
    /// Use this method for 2D tensor matrix operations in computation graphs.
    /// </para>
    /// <para>
    /// GPU acceleration provides 10-100x speedup depending on matrix sizes.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b);

    #endregion

    #region Normalization and Activation Operations

    /// <summary>
    /// Applies softmax activation along the specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor where values along the axis sum to 1.</returns>
    Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward softmax pass.</param>
    /// <param name="axis">The axis along which softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Applies batch normalization to a 2D tensor [batch, features].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, features].</param>
    /// <param name="gamma">Scale parameter with shape [features].</param>
    /// <param name="beta">Shift parameter with shape [features].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean with shape [features].</param>
    /// <param name="variance">Output: computed variance with shape [features].</param>
    /// <returns>The normalized tensor.</returns>
    Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for batch normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    /// <summary>
    /// Applies layer normalization to a 2D tensor [batch, features].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, features].</param>
    /// <param name="gamma">Scale parameter with shape [features].</param>
    /// <param name="beta">Shift parameter with shape [features].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean per sample with shape [batch].</param>
    /// <param name="variance">Output: computed variance per sample with shape [batch].</param>
    /// <returns>The normalized tensor.</returns>
    Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for layer normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the maximum value along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the maximum.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <param name="maxIndices">Output: indices of maximum values for backward pass.</param>
    /// <returns>The tensor containing maximum values.</returns>
    Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices);

    /// <summary>
    /// Computes the backward pass for reduce max.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="maxIndices">The indices of maximum values from forward pass.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape);

    /// <summary>
    /// Computes the mean along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the mean.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The tensor containing mean values.</returns>
    Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims);

    /// <summary>
    /// Computes the backward pass for reduce mean.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes);

    #endregion

    #region Spatial Operations

    /// <summary>
    /// Performs nearest-neighbor upsampling on a 4D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="scaleH">The height scaling factor.</param>
    /// <param name="scaleW">The width scaling factor.</param>
    /// <returns>The upsampled tensor.</returns>
    Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW);

    /// <summary>
    /// Computes the backward pass for upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="scaleH">The height scaling factor used in forward pass.</param>
    /// <param name="scaleW">The width scaling factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW);

    /// <summary>
    /// Performs pixel shuffle (depth-to-space) operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="upscaleFactor">The factor to upscale spatial dimensions.</param>
    /// <returns>The rearranged tensor with increased spatial dimensions.</returns>
    Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor);

    /// <summary>
    /// Computes the backward pass for pixel shuffle.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="upscaleFactor">The upscale factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor);

    /// <summary>
    /// Crops a region from a 4D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="top">The top offset for cropping.</param>
    /// <param name="left">The left offset for cropping.</param>
    /// <param name="height">The height of the cropped region.</param>
    /// <param name="width">The width of the cropped region.</param>
    /// <returns>The cropped tensor.</returns>
    Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width);

    /// <summary>
    /// Computes the backward pass for crop.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="top">The top offset used in forward pass.</param>
    /// <param name="left">The left offset used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left);

    /// <summary>
    /// Pads a 2D tensor with specified values.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="padTop">Padding for top edge.</param>
    /// <param name="padBottom">Padding for bottom edge.</param>
    /// <param name="padLeft">Padding for left edge.</param>
    /// <param name="padRight">Padding for right edge.</param>
    /// <param name="padValue">The value to use for padding.</param>
    /// <returns>The padded tensor.</returns>
    Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue);

    /// <summary>
    /// Computes the backward pass for padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="padTop">Padding used for top edge.</param>
    /// <param name="padLeft">Padding used for left edge.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape);

    /// <summary>
    /// Concatenates tensors along a specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The list of tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>The concatenated tensor.</returns>
    Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis);

    #endregion
}
