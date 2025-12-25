using System;

namespace AiDotNet.Tensors.Interfaces;

/// <summary>
/// Defines vectorized (array/span-based) operations for numeric types that can be SIMD-optimized.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides batch operations on arrays or spans of numeric values. Implementations
/// can use hardware acceleration (SIMD via TensorPrimitives) when available, or fall back to
/// sequential loops for unsupported types.
/// </para>
/// <para>
/// <b>For Beginners:</b> While <see cref="INumericOperations{T}"/> handles single-value operations
/// (like adding two numbers), this interface handles operations on entire arrays at once.
/// </para>
/// <para>
/// Modern CPUs can perform the same operation on multiple values simultaneously using SIMD
/// (Single Instruction Multiple Data). For example, adding two arrays of 8 floats might
/// complete in a single CPU instruction instead of 8 separate additions.
/// </para>
/// <para>
/// <b>Performance Characteristics (with AVX2):</b>
/// - Element-wise operations (Add, Multiply): 5-10x speedup
/// - Reductions (Sum, Max, Min): 8-12x speedup
/// - Transcendentals (Exp, Log, Tanh): 3-6x speedup
/// - Dot product: 10-15x speedup on large vectors
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type for the operations.</typeparam>
public interface IVectorizedOperations<T>
{
    /// <summary>
    /// Performs element-wise addition: destination[i] = x[i] + y[i].
    /// </summary>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <param name="destination">The destination span for results.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);

    /// <summary>
    /// Performs element-wise subtraction: destination[i] = x[i] - y[i].
    /// </summary>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <param name="destination">The destination span for results.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);

    /// <summary>
    /// Performs element-wise multiplication: destination[i] = x[i] * y[i].
    /// </summary>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <param name="destination">The destination span for results.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);

    /// <summary>
    /// Performs element-wise division: destination[i] = x[i] / y[i].
    /// </summary>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <param name="destination">The destination span for results.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination);

    /// <summary>
    /// Computes the dot product (inner product) of two vectors: sum(x[i] * y[i]).
    /// </summary>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <returns>The dot product of the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    T Dot(ReadOnlySpan<T> x, ReadOnlySpan<T> y);

    /// <summary>
    /// Computes the sum of all elements in the span.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <returns>The sum of all elements.</returns>
    T Sum(ReadOnlySpan<T> x);

    /// <summary>
    /// Finds the maximum value in the span.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <returns>The maximum value.</returns>
    /// <exception cref="ArgumentException">Thrown when the span is empty.</exception>
    T Max(ReadOnlySpan<T> x);

    /// <summary>
    /// Finds the minimum value in the span.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <returns>The minimum value.</returns>
    /// <exception cref="ArgumentException">Thrown when the span is empty.</exception>
    T Min(ReadOnlySpan<T> x);

    /// <summary>
    /// Computes the exponential function element-wise: destination[i] = e^x[i].
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Exp(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the natural logarithm element-wise: destination[i] = ln(x[i]).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Log(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the hyperbolic tangent element-wise: destination[i] = tanh(x[i]).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Tanh(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the sigmoid function element-wise: destination[i] = 1 / (1 + e^(-x[i])).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Sigmoid(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the base-2 logarithm element-wise: destination[i] = log2(x[i]).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Log2(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the softmax function: destination[i] = exp(x[i] - max) / sum(exp(x - max)).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Softmax converts a vector of numbers into a probability distribution.
    /// All output values sum to 1, and each value represents the probability of that element.
    /// This is commonly used as the final layer in classification neural networks.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void SoftMax(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the cosine similarity between two vectors: dot(x, y) / (norm(x) * norm(y)).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine similarity measures how similar two vectors are based on
    /// their direction (angle), ignoring their magnitude. The result ranges from -1 (opposite)
    /// to 1 (identical direction), with 0 meaning orthogonal (perpendicular).
    /// </para>
    /// </remarks>
    /// <param name="x">The first source span.</param>
    /// <param name="y">The second source span.</param>
    /// <returns>The cosine similarity between the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    T CosineSimilarity(ReadOnlySpan<T> x, ReadOnlySpan<T> y);

    /// <summary>
    /// Fills the destination span with a constant value.
    /// </summary>
    /// <param name="destination">The destination span to fill.</param>
    /// <param name="value">The value to fill with.</param>
    void Fill(Span<T> destination, T value);

    /// <summary>
    /// Multiplies each element by a scalar: destination[i] = x[i] * scalar.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="scalar">The scalar value to multiply by.</param>
    /// <param name="destination">The destination span for results.</param>
    void MultiplyScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination);

    /// <summary>
    /// Divides each element by a scalar: destination[i] = x[i] / scalar.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="scalar">The scalar value to divide by.</param>
    /// <param name="destination">The destination span for results.</param>
    void DivideScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination);

    /// <summary>
    /// Adds a scalar to each element: destination[i] = x[i] + scalar.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="scalar">The scalar value to add.</param>
    /// <param name="destination">The destination span for results.</param>
    void AddScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination);

    /// <summary>
    /// Subtracts a scalar from each element: destination[i] = x[i] - scalar.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="scalar">The scalar value to subtract.</param>
    /// <param name="destination">The destination span for results.</param>
    void SubtractScalar(ReadOnlySpan<T> x, T scalar, Span<T> destination);

    /// <summary>
    /// Computes the square root of each element: destination[i] = sqrt(x[i]).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Sqrt(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the absolute value of each element: destination[i] = |x[i]|.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Abs(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Negates each element: destination[i] = -x[i].
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Negate(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Clips (clamps) each element to a range: destination[i] = clamp(x[i], min, max).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <param name="destination">The destination span for results.</param>
    void Clip(ReadOnlySpan<T> x, T min, T max, Span<T> destination);

    /// <summary>
    /// Computes the power of each element: destination[i] = x[i]^power.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="power">The power to raise each element to.</param>
    /// <param name="destination">The destination span for results.</param>
    void Pow(ReadOnlySpan<T> x, T power, Span<T> destination);

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    /// <param name="source">The source span.</param>
    /// <param name="destination">The destination span.</param>
    void Copy(ReadOnlySpan<T> source, Span<T> destination);

    /// <summary>
    /// Computes the floor of each element: destination[i] = floor(x[i]).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Floor returns the largest integer less than or equal to each value.
    /// For example, floor(3.7) = 3, floor(-2.3) = -3.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Floor(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the ceiling of each element: destination[i] = ceiling(x[i]).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ceiling returns the smallest integer greater than or equal to each value.
    /// For example, ceiling(3.2) = 4, ceiling(-2.7) = -2.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Ceiling(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the fractional part of each element: destination[i] = x[i] - floor(x[i]).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The fractional part is the portion after the decimal point.
    /// For example, frac(3.7) = 0.7, frac(-2.3) = 0.7 (not -0.3, since frac = x - floor(x)).
    /// This is useful in hash encoding and periodic functions.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Frac(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the sine of each element: destination[i] = sin(x[i]).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sine is a trigonometric function that maps angles to values between -1 and 1.
    /// Input values should be in radians. Used in positional encoding, spherical harmonics, and signal processing.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span (values in radians).</param>
    /// <param name="destination">The destination span for results.</param>
    void Sin(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes the cosine of each element: destination[i] = cos(x[i]).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine is a trigonometric function that maps angles to values between -1 and 1.
    /// Input values should be in radians. Used in positional encoding, spherical harmonics, and signal processing.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span (values in radians).</param>
    /// <param name="destination">The destination span for results.</param>
    void Cos(ReadOnlySpan<T> x, Span<T> destination);
}
