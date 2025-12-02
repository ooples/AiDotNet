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
}
