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
}
