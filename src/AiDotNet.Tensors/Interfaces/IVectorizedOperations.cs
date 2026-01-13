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

    /// <summary>
    /// Computes fused multiply-add: destination[i] = x[i] + y[i] * scalar.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This operation multiplies each element of y by a scalar value,
    /// then adds it to the corresponding element of x. It's commonly used in matrix multiplication
    /// and neural network operations.
    /// </para>
    /// <para>
    /// <b>Performance:</b> When FMA (Fused Multiply-Add) hardware is available, this operation
    /// can be performed in a single instruction, providing better performance and precision
    /// than separate multiply and add operations.
    /// </para>
    /// </remarks>
    /// <param name="x">The first source span (values to add to).</param>
    /// <param name="y">The second source span (values to multiply).</param>
    /// <param name="scalar">The scalar value to multiply y by.</param>
    /// <param name="destination">The destination span for results.</param>
    void MultiplyAdd(ReadOnlySpan<T> x, ReadOnlySpan<T> y, T scalar, Span<T> destination);

    /// <summary>
    /// Converts elements from type T to float (FP32): destination[i] = (float)source[i].
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts an array of numbers from one type (like double or int)
    /// to 32-bit floating-point numbers. This is commonly used when preparing data for GPU processing,
    /// which typically operates on float32 for optimal performance.
    /// </para>
    /// <para>
    /// <b>Performance:</b> This operation can be SIMD-accelerated using TensorPrimitives.ConvertToSingle
    /// on .NET 8+, providing significant speedup over sequential conversion loops.
    /// </para>
    /// </remarks>
    /// <param name="source">The source span containing values of type T.</param>
    /// <param name="destination">The destination span for float results.</param>
    void ToFloatSpan(ReadOnlySpan<T> source, Span<float> destination);

    /// <summary>
    /// Converts elements from float (FP32) to type T: destination[i] = (T)source[i].
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts an array of 32-bit floating-point numbers back to
    /// another type (like double or int). This is commonly used when retrieving results from GPU
    /// processing and converting them back to the user's preferred type.
    /// </para>
    /// <para>
    /// <b>Performance:</b> This operation can be SIMD-accelerated on .NET 8+, providing significant
    /// speedup over sequential conversion loops.
    /// </para>
    /// </remarks>
    /// <param name="source">The source span containing float values.</param>
    /// <param name="destination">The destination span for values of type T.</param>
    void FromFloatSpan(ReadOnlySpan<float> source, Span<T> destination);

    /// <summary>
    /// Converts elements from type T to Half (FP16): destination[i] = (Half)source[i].
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts an array of numbers to 16-bit half-precision
    /// floating-point numbers. Half precision uses less memory and can be faster on GPUs that
    /// support it, at the cost of reduced precision and range.
    /// </para>
    /// <para>
    /// <b>Performance:</b> This operation can be SIMD-accelerated using TensorPrimitives.ConvertToHalf
    /// on .NET 8+, providing significant speedup over sequential conversion loops. Critical for
    /// mixed-precision GPU operations where FP16 loads with FP32 accumulation provides 2x speedup.
    /// </para>
    /// </remarks>
    /// <param name="source">The source span containing values of type T.</param>
    /// <param name="destination">The destination span for Half results.</param>
    void ToHalfSpan(ReadOnlySpan<T> source, Span<Half> destination);

    /// <summary>
    /// Converts elements from Half (FP16) to type T: destination[i] = (T)source[i].
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts an array of 16-bit half-precision floating-point
    /// numbers to another type (like float or double). This is commonly used when retrieving results
    /// from GPU processing that used half precision.
    /// </para>
    /// <para>
    /// <b>Performance:</b> This operation can be SIMD-accelerated on .NET 8+, providing significant
    /// speedup over sequential conversion loops.
    /// </para>
    /// </remarks>
    /// <param name="source">The source span containing Half values.</param>
    /// <param name="destination">The destination span for values of type T.</param>
    void FromHalfSpan(ReadOnlySpan<Half> source, Span<T> destination);

    /// <summary>
    /// Checks if all elements in the span are finite (not NaN or Infinity).
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="badIndex">The index of the first non-finite value found, or -1 if all are finite.</param>
    /// <returns>True if all elements are finite, false otherwise.</returns>
    bool AllFinite(ReadOnlySpan<T> x, out int badIndex);

    /// <summary>
    /// Checks if any element in the span is NaN or Infinity.
    /// </summary>
    /// <param name="x">The source span.</param>
    /// <param name="badIndex">When this method returns true, contains the index of the first non-finite value found; otherwise, -1.</param>
    /// <returns>True if any element is non-finite; otherwise, false.</returns>
    bool IsAnyNonFinite(ReadOnlySpan<T> x, out int badIndex);

    #region Vectorized Activation Functions

    /// <summary>
    /// Computes LeakyReLU element-wise: destination[i] = x[i] > 0 ? x[i] : alpha * x[i].
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LeakyReLU is a variant of ReLU that allows a small negative slope
    /// for negative inputs instead of zeroing them out. This helps prevent "dying neurons".
    /// </para>
    /// <para>
    /// <b>Performance:</b> This operation can be SIMD-accelerated using vectorized comparisons
    /// and conditional selection, providing 3-5x speedup over scalar loops.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="alpha">The negative slope coefficient (typically 0.01).</param>
    /// <param name="destination">The destination span for results.</param>
    void LeakyReLU(ReadOnlySpan<T> x, T alpha, Span<T> destination);

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise.
    /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GELU is a smooth activation function that approximates
    /// multiplying the input by a probability based on its value. It's used in
    /// Transformers (BERT, GPT) and provides smooth gradients for optimization.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Uses SIMD-optimized tanh, exp, and arithmetic operations
    /// for 2-4x speedup over scalar implementation.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void GELU(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes Mish activation element-wise: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mish is a smooth, self-regularized activation function that
    /// often outperforms ReLU in practice. It's unbounded above, bounded below, smooth,
    /// and non-monotonic.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Uses SIMD-optimized exp, log, and tanh operations
    /// for 2-3x speedup over scalar implementation.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Mish(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes Swish/SiLU activation element-wise: x * sigmoid(x) = x / (1 + exp(-x)).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Swish (also called SiLU - Sigmoid Linear Unit) is a smooth,
    /// non-monotonic activation function that often outperforms ReLU. It allows negative
    /// values to pass through, helping with gradient flow.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Uses SIMD-optimized sigmoid and multiplication operations
    /// for 2-4x speedup over scalar implementation.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void Swish(ReadOnlySpan<T> x, Span<T> destination);

    /// <summary>
    /// Computes ELU (Exponential Linear Unit) element-wise: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ELU is similar to ReLU for positive inputs but produces smooth
    /// negative values for negative inputs. This helps push mean activations closer to zero,
    /// speeding up learning.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Uses SIMD-optimized exp, comparisons, and conditional selection
    /// for 2-4x speedup over scalar implementation.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="alpha">The scale factor for negative values (typically 1.0).</param>
    /// <param name="destination">The destination span for results.</param>
    void ELU(ReadOnlySpan<T> x, T alpha, Span<T> destination);

    /// <summary>
    /// Computes ReLU (Rectified Linear Unit) element-wise: max(0, x).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ReLU is the most common activation function in deep learning.
    /// It outputs the input directly if positive, otherwise outputs zero.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Uses SIMD-optimized maximum operation for 5-10x speedup.
    /// </para>
    /// </remarks>
    /// <param name="x">The source span.</param>
    /// <param name="destination">The destination span for results.</param>
    void ReLU(ReadOnlySpan<T> x, Span<T> destination);

    #endregion
}
