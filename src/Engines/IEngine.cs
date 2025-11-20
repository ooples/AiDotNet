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
/// </remarks>
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

    #endregion
}
