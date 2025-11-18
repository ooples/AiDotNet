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
}
