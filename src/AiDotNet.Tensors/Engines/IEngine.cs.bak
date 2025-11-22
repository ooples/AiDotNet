using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

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
    /// Computes the base-2 logarithm of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-2 logarithms.</returns>
    /// <remarks>
    /// Used in information theory (entropy, bits), binary tree computations, and quantization.
    /// </remarks>
    Vector<T> Log2<T>(Vector<T> vector);

    /// <summary>
    /// Computes exp(x) - 1 for each element with higher precision for small values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing exp(x) - 1 for each element.</returns>
    /// <remarks>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// </remarks>
    Vector<T> ExpM1<T>(Vector<T> vector);

    /// <summary>
    /// Computes log(1 + x) for each element with higher precision for small values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing log(1 + x) for each element.</returns>
    /// <remarks>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// </remarks>
    Vector<T> Log1P<T>(Vector<T> vector);

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

    /// <summary>
    /// Negates each element of the vector (computes -x).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element negated.</returns>
    /// <remarks>
    /// Used for gradient reversal, adversarial training, and sign flipping operations.
    /// </remarks>
    Vector<T> Negate<T>(Vector<T> vector);

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
    /// Applies the softmax function to convert a vector of values into a probability distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector of logits.</param>
    /// <returns>A new vector where elements sum to 1 and represent probabilities.</returns>
    /// <remarks>
    /// <para>
    /// Softmax converts arbitrary real values into a probability distribution using:
    /// softmax(x)[i] = exp(x[i]) / sum(exp(x[j])) for all j
    /// </para>
    /// <para>
    /// For numerical stability, this is computed as:
    /// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    /// </para>
    /// <para><b>Common Uses:</b></para>
    /// <list type="bullet">
    /// <item><description>Final layer of classification networks (converts scores to probabilities)</description></item>
    /// <item><description>Attention mechanisms (weighs how much focus to give each element)</description></item>
    /// <item><description>Mixture-of-Experts routing (determines which experts to use)</description></item>
    /// <item><description>Reinforcement learning policy networks</description></item>
    /// </list>
    /// <para>
    /// GPU implementation provides 10-50x speedup for large vectors.
    /// Uses hardware-accelerated exp() and reduction operations.
    /// </para>
    /// </remarks>
    Vector<T> Softmax<T>(Vector<T> vector);

    /// <summary>
    /// Computes the cosine similarity between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A value between -1 and 1 representing the cosine of the angle between the vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Cosine similarity measures the cosine of the angle between two non-zero vectors:
    /// cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
    /// </para>
    /// <para><b>Return Values:</b></para>
    /// <list type="bullet">
    /// <item><description>1.0: Vectors point in exactly the same direction (identical)</description></item>
    /// <item><description>0.0: Vectors are orthogonal (perpendicular, no similarity)</description></item>
    /// <item><description>-1.0: Vectors point in opposite directions</description></item>
    /// </list>
    /// <para><b>Common Uses:</b></para>
    /// <list type="bullet">
    /// <item><description>Text similarity in NLP (document/sentence embeddings)</description></item>
    /// <item><description>Recommendation systems (user/item similarity)</description></item>
    /// <item><description>Image similarity (feature vector comparison)</description></item>
    /// <item><description>Attention mechanisms in transformers</description></item>
    /// </list>
    /// <para>
    /// GPU implementation provides 20-100x speedup by parallelizing dot product and norm computations.
    /// Returns zero if either vector has zero magnitude to avoid division by zero.
    /// </para>
    /// </remarks>
    T CosineSimilarity<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the product of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The product of all elements.</returns>
    /// <remarks>
    /// Used for geometric mean computation and product aggregation in statistics.
    /// </remarks>
    T Product<T>(Vector<T> vector);

    /// <summary>
    /// Computes the standard deviation of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The standard deviation of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Standard deviation measures the spread of values: sqrt(variance).
    /// Essential for batch normalization, layer normalization, and outlier detection.
    /// </para>
    /// </remarks>
    T StdDev<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Euclidean norm (L2 norm) of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The L2 norm: sqrt(sum(x[i]Ãƒâ€šÃ‚Â²)).</returns>
    /// <remarks>
    /// <para>
    /// L2 norm is the Euclidean length of the vector.
    /// Critical for:
    /// - Gradient clipping (clip by norm)
    /// - L2 regularization
    /// - Vector normalization (unit vectors)
    /// - Distance metrics
    /// </para>
    /// </remarks>
    T Norm<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The Euclidean distance: sqrt(sum((a[i] - b[i])Ãƒâ€šÃ‚Â²)).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Euclidean distance is the straight-line distance between two points.
    /// Used extensively in:
    /// - k-Nearest Neighbors (k-NN)
    /// - Clustering algorithms (k-means, DBSCAN)
    /// - Metric learning
    /// - Similarity search
    /// </para>
    /// </remarks>
    T Distance<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise minimum magnitude between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is the value with minimum absolute value.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// Used for magnitude-based comparisons and weight pruning strategies.
    /// </remarks>
    Vector<T> MinMagnitude<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise maximum magnitude between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is the value with maximum absolute value.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// Used for gradient clipping by magnitude and weight analysis.
    /// </remarks>
    Vector<T> MaxMagnitude<T>(Vector<T> a, Vector<T> b);

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

    #region Specialized Operations

    /// <summary>
    /// Clamps each element of a vector to the specified range [min, max].
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A new vector with each element clamped to [min, max].</returns>
    /// <remarks>
    /// <para>
    /// Clamp ensures all values are within bounds: result[i] = max(min, min(max, vector[i])).
    /// Critical for:
    /// - Gradient clipping (prevent exploding gradients)
    /// - Activation function bounds (e.g., ReLU6)
    /// - Numerical stability
    /// - Value range enforcement
    /// </para>
    /// </remarks>
    Vector<T> Clamp<T>(Vector<T> vector, T min, T max);

    /// <summary>
    /// Performs linear interpolation between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The start vector.</param>
    /// <param name="b">The end vector.</param>
    /// <param name="t">The interpolation weight (0 to 1).</param>
    /// <returns>A new vector with interpolated values: a + t * (b - a).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Linear interpolation blends two vectors based on weight t.
    /// Used for:
    /// - Exponential moving average (EMA) in optimizers
    /// - Model weight interpolation
    /// - Smooth transitions between states
    /// - Temporal blending
    /// </para>
    /// </remarks>
    Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, T t);

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing 1/x for each element.</returns>
    /// <remarks>
    /// <para>
    /// Reciprocal is used for:
    /// - Division optimization (multiply by reciprocal)
    /// - Normalization operations
    /// - Inverse scaling
    /// </para>
    /// </remarks>
    Vector<T> Reciprocal<T>(Vector<T> vector);

    /// <summary>
    /// Computes the reciprocal square root (1/sqrt(x)) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing 1/sqrt(x) for each element.</returns>
    /// <remarks>
    /// <para>
    /// Reciprocal square root is CRITICAL for normalization efficiency.
    /// Essential for:
    /// - Layer normalization: x / sqrt(variance + epsilon)
    /// - Batch normalization: (x - mean) / sqrt(variance + epsilon)
    /// - RMS normalization (RMSNorm used in LLaMA, GPT-NeoX)
    /// - Fast inverse square root (Quake III algorithm)
    /// </para>
    /// <para>
    /// GPU/SIMD implementations provide hardware-accelerated rsqrt instruction,
    /// which is significantly faster than computing sqrt followed by division.
    /// </para>
    /// </remarks>
    Vector<T> ReciprocalSqrt<T>(Vector<T> vector);

    #endregion

    #region Trigonometric Operations

    /// <summary>
    /// Computes the sine of each element (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <returns>A new vector containing sine values.</returns>
    /// <remarks>
    /// <para>
    /// Sine is used extensively in:
    /// - Positional encodings for transformers (sin/cos for position embedding)
    /// - Signal processing and wave functions
    /// - Fourier transforms
    /// - Periodic activation functions
    /// </para>
    /// </remarks>
    Vector<T> Sin<T>(Vector<T> vector);

    /// <summary>
    /// Computes the cosine of each element (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <returns>A new vector containing cosine values.</returns>
    /// <remarks>
    /// <para>
    /// Cosine is used extensively in:
    /// - Positional encodings for transformers (sin/cos for position embedding)
    /// - Cosine annealing learning rate schedules
    /// - Attention mechanisms
    /// - Signal processing
    /// </para>
    /// </remarks>
    Vector<T> Cos<T>(Vector<T> vector);

    /// <summary>
    /// Computes both sine and cosine of each element simultaneously (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <param name="sinResult">Output vector containing sine values.</param>
    /// <param name="cosResult">Output vector containing cosine values.</param>
    /// <remarks>
    /// <para>
    /// SinCos computes both sin and cos simultaneously, which is more efficient
    /// than calling Sin() and Cos() separately.
    /// Critical for:
    /// - Positional encodings in transformers (need both sin and cos)
    /// - Complex number operations
    /// - Rotary position embeddings (RoPE)
    /// </para>
    /// <para>
    /// Hardware implementations can compute both with ~1.5x the cost of a single
    /// sin/cos operation, rather than 2x.
    /// </para>
    /// </remarks>
    void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult);

    /// <summary>
    /// Computes the sine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write sine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Sin loop.
    /// </para>
    /// </remarks>
    void Sin(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the sine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write sine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Sin loop.
    /// </para>
    /// </remarks>
    void Sin(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the cosine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write cosine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Cos loop.
    /// </para>
    /// </remarks>
    void Cos(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the cosine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write cosine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Cos loop.
    /// </para>
    /// </remarks>
    void Cos(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Exponential Operations

    /// <summary>
    /// Computes e raised to the power of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (exponents).</param>
    /// <param name="destination">The destination span to write exp values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Exp loop.
    /// </para>
    /// </remarks>
    void Exp(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes e raised to the power of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (exponents).</param>
    /// <param name="destination">The destination span to write exp values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Exp loop.
    /// </para>
    /// </remarks>
    void Exp(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the natural logarithm of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be positive).</param>
    /// <param name="destination">The destination span to write log values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Log loop.
    /// </para>
    /// </remarks>
    void Log(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the natural logarithm of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be positive).</param>
    /// <param name="destination">The destination span to write log values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Log loop.
    /// </para>
    /// </remarks>
    void Log(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes exp(x) - 1 for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write exp(x) - 1 values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// On .NET 5.0+, uses Math.ExpM1 for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void ExpM1(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes exp(x) - 1 for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write exp(x) - 1 values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// On .NET 5.0+, uses Math.ExpM1 for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void ExpM1(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes log(1 + x) for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write log(1 + x) values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// On .NET 5.0+, uses Math.Log1P for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void Log1P(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes log(1 + x) for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write log(1 + x) values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// On .NET 5.0+, uses Math.Log1P for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void Log1P(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the tangent of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write tangent values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Tan(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the tangent of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write tangent values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Tan(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse sine of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (values in range [-1, 1]).</param>
    /// <returns>A new vector containing arcsin values in range [-π/2, π/2].</returns>
    /// <remarks>
    /// <para>
    /// Inverse sine (arcsin) is the inverse of the sine function.
    /// Input domain: [-1, 1]
    /// Output range: [-π/2, π/2] radians
    /// </para>
    /// </remarks>
    Vector<T> Asin<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arcsin values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asin(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arcsin values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asin(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse cosine of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (values in range [-1, 1]).</param>
    /// <returns>A new vector containing arccos values in range [0, π].</returns>
    /// <remarks>
    /// <para>
    /// Inverse cosine (arccos) is the inverse of the cosine function.
    /// Input domain: [-1, 1]
    /// Output range: [0, π] radians
    /// </para>
    /// </remarks>
    Vector<T> Acos<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arccos values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acos(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arccos values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acos(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse tangent of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing arctan values in range (-π/2, π/2).</returns>
    /// <remarks>
    /// <para>
    /// Inverse tangent (arctan) is the inverse of the tangent function.
    /// Input domain: (-Inf, Inf)
    /// Output range: (-π/2, π/2) radians
    /// </para>
    /// </remarks>
    Vector<T> Atan<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write arctan values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atan(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write arctan values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atan(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the square root of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be non-negative).</param>
    /// <param name="destination">The destination span to write sqrt values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Sqrt(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the square root of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be non-negative).</param>
    /// <param name="destination">The destination span to write sqrt values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Sqrt(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the absolute value of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write absolute values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Abs(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the absolute value of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write absolute values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Abs(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Hyperbolic Operations

    /// <summary>
    /// Computes the hyperbolic sine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing sinh values.</returns>
    /// <remarks>
    /// Hyperbolic sine: sinh(x) = (e^x - e^-x) / 2.
    /// Used in some activation functions and mathematical transformations.
    /// </remarks>
    Vector<T> Sinh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the hyperbolic cosine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing cosh values.</returns>
    /// <remarks>
    /// Hyperbolic cosine: cosh(x) = (e^x + e^-x) / 2.
    /// Component of tanh gradient and used in hyperbolic geometry.
    /// </remarks>
    Vector<T> Cosh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing asinh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic sine: asinh(x) = log(x + sqrt(x^2 + 1)).
    /// Also called the hyperbolic area sine function. Domain: (-inf, inf), Range: (-inf, inf).
    /// </remarks>
    Vector<T> Asinh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing acosh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic cosine: acosh(x) = log(x + sqrt(x^2 - 1)).
    /// Also called the hyperbolic area cosine function. Domain: [1, inf), Range: [0, inf).
    /// </remarks>
    Vector<T> Acosh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing atanh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic tangent: atanh(x) = 0.5 * log((1 + x) / (1 - x)).
    /// Also called the hyperbolic area tangent function. Domain: (-1, 1), Range: (-inf, inf).
    /// </remarks>
    Vector<T> Atanh<T>(Vector<T> vector);

    void Sinh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Sinh(System.ReadOnlySpan<double> x, System.Span<double> destination);
    void Cosh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Cosh(System.ReadOnlySpan<double> x, System.Span<double> destination);
    void Tanh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Tanh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write asinh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asinh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write asinh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asinh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be >= 1).</param>
    /// <param name="destination">The destination span to write acosh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acosh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be >= 1).</param>
    /// <param name="destination">The destination span to write acosh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acosh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be in range (-1, 1)).</param>
    /// <param name="destination">The destination span to write atanh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atanh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be in range (-1, 1)).</param>
    /// <param name="destination">The destination span to write atanh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atanh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Additional Mathematical Operations

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    void Reciprocal(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    void Reciprocal(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the cube root of each element.
    /// </summary>
    void Cbrt(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the cube root of each element.
    /// </summary>
    void Cbrt(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the base-2 logarithm of each element.
    /// </summary>
    void Log2(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the base-2 logarithm of each element.
    /// </summary>
    void Log2(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the base-10 logarithm of each element.
    /// </summary>
    void Log10(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the base-10 logarithm of each element.
    /// </summary>
    void Log10(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Rounding Operations

    /// <summary>
    /// Rounds each element to the nearest integer.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded to nearest integer.</returns>
    /// <remarks>
    /// <para>
    /// Rounding is used for:
    /// - Quantization (neural network compression)
    /// - Discretization for inference
    /// - Integer conversion
    /// </para>
    /// </remarks>
    Vector<T> Round<T>(Vector<T> vector);

    /// <summary>
    /// Computes the floor (round down) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded down to integer.</returns>
    /// <remarks>
    /// Floor rounds toward negative infinity. Used for integer conversion and binning operations.
    /// </remarks>
    Vector<T> Floor<T>(Vector<T> vector);

    /// <summary>
    /// Computes the ceiling (round up) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded up to integer.</returns>
    /// <remarks>
    /// Ceiling rounds toward positive infinity. Used for integer conversion and binning operations.
    /// </remarks>
    Vector<T> Ceiling<T>(Vector<T> vector);

    /// <summary>
    /// Truncates each element toward zero (removes fractional part).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with fractional parts removed.</returns>
    /// <remarks>
    /// Truncate rounds toward zero. Used for integer conversion and quantization schemes.
    /// </remarks>
    Vector<T> Truncate<T>(Vector<T> vector);

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
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float).
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
    /// Sigmoid activation function: ÃƒÂÃ†â€™(x) = 1 / (1 + e^-x).
    /// Commonly used for binary classification and gate functions in LSTMs/GRUs.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float).
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
    /// GELU activation: x * ÃƒÅ½Ã‚Â¦(x) where ÃƒÅ½Ã‚Â¦ is the standard Gaussian cumulative distribution.
    /// Commonly used in transformers (BERT, GPT) and modern architectures.
    /// Approximation: 0.5 * x * (1 + tanh(ÃƒÂ¢Ã‹â€ Ã…Â¡(2/ÃƒÂÃ¢â€šÂ¬) * (x + 0.044715 * xÃƒâ€šÃ‚Â³)))
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
    /// Matrix multiplication is O(nÃƒâ€šÃ‚Â³) - highly computationally intensive.
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
    /// <returns>An MÃƒÆ’Ã¢â‚¬â€N matrix containing the outer product.</returns>
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
    /// <param name="a">The first tensor [B, M, K] - B batches of MÃƒÆ’Ã¢â‚¬â€K matrices.</param>
    /// <param name="b">The second tensor [B, K, N] - B batches of KÃƒÆ’Ã¢â‚¬â€N matrices.</param>
    /// <returns>The result tensor [B, M, N] - B batches of MÃƒÆ’Ã¢â‚¬â€N matrices.</returns>
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
