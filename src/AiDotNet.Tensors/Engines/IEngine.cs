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
    /// Computes 2^x (base-2 exponential) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-2 exponentials.</returns>
    /// <remarks>
    /// Used in information theory (entropy calculations), binary computations, and scientific applications
    /// that work with powers of 2 (common in signal processing and computer graphics).
    /// </remarks>
    Vector<T> Exp2<T>(Vector<T> vector);

    /// <summary>
    /// Computes 10^x (base-10 exponential) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-10 exponentials.</returns>
    /// <remarks>
    /// Used in scientific and engineering applications, decibel calculations (dB), and pH computations.
    /// Common in physics, chemistry, and signal processing where base-10 is the standard scale.
    /// </remarks>
    Vector<T> Exp10<T>(Vector<T> vector);

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
    /// <returns>The L2 norm: sqrt(sum(x[i]ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)).</returns>
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
    /// <returns>The Euclidean distance: sqrt(sum((a[i] - b[i])ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)).</returns>
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
    /// <returns>A new vector containing arcsin values in range [-Ãâ‚¬/2, Ãâ‚¬/2].</returns>
    /// <remarks>
    /// <para>
    /// Inverse sine (arcsin) is the inverse of the sine function.
    /// Input domain: [-1, 1]
    /// Output range: [-Ãâ‚¬/2, Ãâ‚¬/2] radians
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
    /// <returns>A new vector containing arccos values in range [0, Ãâ‚¬].</returns>
    /// <remarks>
    /// <para>
    /// Inverse cosine (arccos) is the inverse of the cosine function.
    /// Input domain: [-1, 1]
    /// Output range: [0, Ãâ‚¬] radians
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
    /// <returns>A new vector containing arctan values in range (-Ãâ‚¬/2, Ãâ‚¬/2).</returns>
    /// <remarks>
    /// <para>
    /// Inverse tangent (arctan) is the inverse of the tangent function.
    /// Input domain: (-Inf, Inf)
    /// Output range: (-Ãâ‚¬/2, Ãâ‚¬/2) radians
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
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â speedup for float).
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
    /// Sigmoid activation function: ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢(x) = 1 / (1 + e^-x).
    /// Commonly used for binary classification and gate functions in LSTMs/GRUs.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â speedup for float).
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
    /// GELU activation: x * ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦(x) where ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ is the standard Gaussian cumulative distribution.
    /// Commonly used in transformers (BERT, GPT) and modern architectures.
    /// Approximation: 0.5 * x * (1 + tanh(ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã¢â‚¬Â¹ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡(2/ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬) * (x + 0.044715 * xÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³)))
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
    /// Matrix multiplication is O(nÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³) - highly computationally intensive.
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
    /// <returns>An MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrix containing the outer product.</returns>
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
    /// Reshapes a tensor to a new shape with the same total number of elements.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to reshape.</param>
    /// <param name="newShape">The new shape dimensions.</param>
    /// <returns>A new tensor with the specified shape and the same data.</returns>
    /// <exception cref="ArgumentException">Thrown when the total number of elements doesn't match.</exception>
    Tensor<T> Reshape<T>(Tensor<T> tensor, int[] newShape);

    /// <summary>
    /// Performs batched matrix multiplication on 3D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor [B, M, K] - B batches of MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂK matrices.</param>
    /// <param name="b">The second tensor [B, K, N] - B batches of KÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrices.</param>
    /// <returns>The result tensor [B, M, N] - B batches of MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrices.</returns>
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
    /// Adds two tensors with broadcasting support, following NumPy/PyTorch broadcasting rules.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor. Can have different shape if broadcastable.</param>
    /// <returns>A new tensor containing the element-wise sum with broadcasting.</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are not broadcastable.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations with Broadcasting</b></para>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be added together by automatically
    /// expanding dimensions of size 1 to match the other tensor. This is essential for operations
    /// like adding per-channel bias in convolutional layers.
    /// </para>
    /// <para>
    /// For example, adding shapes [batch, channels, H, W] + [1, channels, 1, 1] broadcasts
    /// the bias across batch and spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Subtracts two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor to subtract (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise difference with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be subtracted by automatically
    /// expanding the smaller tensor. This is commonly used in operations like normalizing
    /// by subtracting a mean: [batch, features] - [1, features] broadcasts the mean across the batch.
    /// </para>
    /// <para>
    /// For example, subtracting shapes [batch, channels, H, W] - [1, channels, 1, 1] broadcasts
    /// the bias subtraction across batch and spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastSubtract<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Divides two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The dividend tensor.</param>
    /// <param name="b">The divisor tensor (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise quotient with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be divided by automatically
    /// expanding the smaller tensor. This is commonly used in normalization operations
    /// like dividing by a sum: [batch, features] / [batch, 1] broadcasts the divisor across features.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastDivide<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise product with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically
    /// expanding the smaller tensor. For example, [B,H,W,C] * [B,1,1,C] broadcasts the [B,1,1,C]
    /// tensor across the spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds multiple tensors element-wise in a single optimized operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to add together.</param>
    /// <returns>A new tensor containing the element-wise sum of all inputs.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than 2 tensors provided or shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.stack + torch.sum pattern, this avoids intermediate allocations
    /// by computing all additions in a single pass. Essential for residual networks and
    /// skip connections that combine multiple feature maps.
    /// </para>
    /// <para>
    /// Performance: O(n*elements) where n is number of tensors, but with single output allocation
    /// instead of n-1 intermediate allocations from chained binary additions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAddMany<T>(params Tensor<T>[] tensors);

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
    /// Multiplies multiple tensors element-wise in a single optimized operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to multiply together.</param>
    /// <returns>A new tensor containing the element-wise product of all inputs.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than 2 tensors provided or shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.stack + torch.prod pattern, this avoids intermediate allocations
    /// by computing all multiplications in a single pass. Useful for gating mechanisms
    /// and attention computations that combine multiple masks or weights.
    /// </para>
    /// <para>
    /// Performance: O(n*elements) where n is number of tensors, but with single output allocation
    /// instead of n-1 intermediate allocations from chained binary multiplications.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMultiplyMany<T>(params Tensor<T>[] tensors);

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

    #region Tensor Comparison Operations

    /// <summary>
    /// Compares each element of a tensor to a scalar value for equality.
    /// Returns a tensor where each element is 1 if equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where equal, 0 where not equal.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.eq(), this enables vectorized comparison operations.
    /// Essential for masking operations in neural networks.
    /// </para>
    /// </remarks>
    Tensor<T> TensorEquals<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares two tensors element-wise for equality.
    /// Returns a tensor where each element is 1 if equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where equal, 0 where not equal.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorEquals<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for inequality.
    /// Returns a tensor where each element is 1 if not equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where not equal, 0 where equal.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.ne(), this enables vectorized inequality comparison.
    /// Essential for masking layers where we need to identify non-padding values.
    /// </para>
    /// </remarks>
    Tensor<T> TensorNotEquals<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares two tensors element-wise for inequality.
    /// Returns a tensor where each element is 1 if not equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where not equal, 0 where equal.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorNotEquals<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of tensor a to corresponding element of tensor b for greater than.
    /// Returns a tensor where each element is 1 if a > b, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where a > b, 0 otherwise.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorGreaterThan<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for greater than.
    /// Returns a tensor where each element is 1 if element > value, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where greater, 0 otherwise.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorGreaterThan<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares each element of tensor a to corresponding element of tensor b for less than.
    /// Returns a tensor where each element is 1 if a &lt; b, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where a &lt; b, 0 otherwise.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorLessThan<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for less than.
    /// Returns a tensor where each element is 1 if element &lt; value, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where less, 0 otherwise.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorLessThan<T>(Tensor<T> tensor, T value);

    #endregion

    #region Tensor Element-wise Math Operations

    /// <summary>
    /// Computes the element-wise natural logarithm of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with the natural logarithm of each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Computes log(x) for each element. Used in:
    /// - Cross-entropy loss calculation
    /// - Log-probability computations
    /// - Attention entropy regularization
    /// </para>
    /// </remarks>
    Tensor<T> TensorLog<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise exponential of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with exp(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in softmax computation, probability distributions, and exponential scaling.
    /// </para>
    /// </remarks>
    Tensor<T> TensorExp<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise square root of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with sqrt(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in normalization layers, RMSProp/Adam optimizers, and standard deviation calculations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSqrt<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise absolute value of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with abs(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in L1 regularization, MAE loss, and gradient clipping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAbs<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise negation of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with -x for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// </remarks>
    Tensor<T> TensorNegate<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to a scalar exponent.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor (base values).</param>
    /// <param name="exponent">The scalar exponent to raise each element to.</param>
    /// <returns>A tensor with pow(x, exponent) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in attention sharpening (Neural Turing Machines), gamma correction,
    /// polynomial features, and various normalization operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPower<T>(Tensor<T> tensor, T exponent);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to another tensor (element-wise exponents).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="bases">The input tensor of base values.</param>
    /// <param name="exponents">The tensor of exponents (must have same shape as bases).</param>
    /// <returns>A tensor with pow(bases[i], exponents[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used for element-wise power operations where exponents vary per element.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPower<T>(Tensor<T> bases, Tensor<T> exponents);

    /// <summary>
    /// Computes the element-wise floor of a tensor (largest integer less than or equal to each element).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with floor(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in hash encoding for 3D AI (NeRF, Gaussian Splatting), grid-based calculations,
    /// and index computation. Essential for converting continuous coordinates to discrete grid indices.
    /// </para>
    /// </remarks>
    Tensor<T> TensorFloor<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise ceiling of a tensor (smallest integer greater than or equal to each element).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with ceiling(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ceil mode pooling, index calculations, and grid-based spatial computations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCeiling<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise fractional part of a tensor (x - floor(x)).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with frac(x) = x - floor(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for hash encoding in neural radiance fields (NeRF) and Instant-NGP.
    /// The fractional part is used to interpolate between discrete grid corners for
    /// smooth, differentiable spatial encoding. Also used in periodic functions and texture mapping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorFrac<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise sine of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor with angles in radians.</param>
    /// <returns>A tensor with sin(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for positional encoding in transformers and neural radiance fields.
    /// Positional encoding uses sin(position * frequency) to create smooth,
    /// periodic spatial features that help models understand relative positions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSin<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise cosine of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor with angles in radians.</param>
    /// <returns>A tensor with cos(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for positional encoding in transformers and neural radiance fields.
    /// Positional encoding uses cos(position * frequency) alongside sin to create
    /// unique, differentiable position representations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCos<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs trilinear interpolation on a 3D grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="grid">The 3D feature grid of shape [D, H, W, C] where D=depth, H=height, W=width, C=channels.</param>
    /// <param name="positions">The 3D coordinates to sample at, shape [N, 3] where each row is (z, y, x) in range [0, D-1], [0, H-1], [0, W-1].</param>
    /// <returns>Interpolated values of shape [N, C].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: 3D Spatial Operations</b></para>
    /// <para>
    /// Essential for hash encoding in neural radiance fields (NeRF) and Instant-NGP.
    /// Trilinear interpolation samples from a discrete 3D grid using continuous coordinates,
    /// computing weighted averages of the 8 surrounding voxel corners. The fractional part
    /// of each coordinate determines the interpolation weights.
    /// 
    /// Formula for weights at position (z, y, x):
    /// - fz, fy, fx = fractional parts of z, y, x
    /// - Corners weighted by (1-fz)*(1-fy)*(1-fx), fz*(1-fy)*(1-fx), etc.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTrilinearInterpolate<T>(Tensor<T> grid, Tensor<T> positions);

    /// <summary>
    /// Computes the backward pass for trilinear interpolation, returning gradients for the grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the upstream layer [numPositions, channels].</param>
    /// <param name="grid">The original 3D grid [depth, height, width, channels].</param>
    /// <param name="positions">The positions at which interpolation was performed [numPositions, 3].</param>
    /// <returns>The gradient with respect to the grid [depth, height, width, channels].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: 3D Spatial Operations</b></para>
    /// <para>
    /// Essential for training neural radiance fields (NeRF) and Instant-NGP with backpropagation.
    /// The backward pass scatters gradients to the 8 surrounding voxel corners using the same
    /// trilinear interpolation weights computed during the forward pass.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTrilinearInterpolateBackward<T>(Tensor<T> gradOutput, Tensor<T> grid, Tensor<T> positions);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to a scalar exponent.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor (base).</param>
    /// <param name="exponent">The scalar exponent.</param>
    /// <returns>A tensor with pow(x, exponent) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in polynomial features, learning rate scheduling, and custom activations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPow<T>(Tensor<T> tensor, T exponent);

    /// <summary>
    /// Computes the element-wise maximum of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor with max(a[i], b[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ReLU activation, gradient clipping, and element-wise maximum operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMax<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes the element-wise maximum of a tensor and a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor with max(x, value) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ReLU activation (max(0, x)), clamping lower bounds, and preventing log(0).
    /// </para>
    /// </remarks>
    Tensor<T> TensorMax<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Computes the element-wise minimum of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor with min(a[i], b[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMin<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes the element-wise minimum of a tensor and a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor with min(x, value) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in clamping upper bounds and gradient clipping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMin<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Clamps tensor values to a specified range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="min">The minimum value (lower bound).</param>
    /// <param name="max">The maximum value (upper bound).</param>
    /// <returns>A tensor with values clamped to [min, max].</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Equivalent to min(max(x, min), max). Used for gradient clipping and value normalization.
    /// </para>
    /// </remarks>
    Tensor<T> TensorClamp<T>(Tensor<T> tensor, T min, T max);

    /// <summary>
    /// Computes the sum of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar sum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to a scalar. For axis-wise reduction, use ReduceSum.
    /// </para>
    /// </remarks>
    T TensorSum<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the sum along specified axes (axis reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axes">The axes along which to sum. Null or empty for full reduction.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The reduced tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in batch/layer normalization, attention weight computation, and loss calculation.
    /// </para>
    /// </remarks>
    Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>
    /// Computes the maximum value of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar maximum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to find the maximum value. Used in:
    /// - Attention weight analysis (max weight indicates peakiness)
    /// - Gradient clipping (finding max gradient magnitude)
    /// - Numerical stability (finding scale factors)
    /// - Normalization (max-normalization)
    /// </para>
    /// </remarks>
    T TensorMaxValue<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the minimum value of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar minimum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to find the minimum value. Used in:
    /// - Range normalization (min-max scaling)
    /// - Gradient analysis
    /// - Numerical stability checks
    /// </para>
    /// </remarks>
    T TensorMinValue<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the mean (average) of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar mean of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to compute mean. Used in:
    /// - Layer normalization (computing mean for centering)
    /// - Batch statistics
    /// - Loss averaging
    /// </para>
    /// </remarks>
    T TensorMean<T>(Tensor<T> tensor);

    #endregion

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
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="kernel">The convolution kernel used in forward pass.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor.</returns>
    Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the kernel (weights).
    /// </summary>
    Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Performs a Locally Connected 2D convolution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="weights">The weights tensor [out_h, out_w, out_channels, in_channels, kernel_h, kernel_w].</param>
    /// <param name="bias">Optional bias tensor [out_channels].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the input tensor.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the weights.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the bias.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput);

    /// <summary>
    /// Transposes a 2D tensor (matrix represented as tensor).
    /// </summary>
    Tensor<T> TensorTranspose<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs matrix multiplication supporting tensors of any rank (PyTorch-style batched matmul).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor. Can be 2D [M, N] or higher rank [..., M, N].</param>
    /// <param name="b">The second tensor. Can be 2D [N, P] or higher rank [..., N, P].</param>
    /// <returns>The result tensor with appropriately broadcasted batch dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This follows PyTorch's torch.matmul semantics for batched matrix multiplication:
    /// </para>
    /// <para><b>Supported combinations:</b></para>
    /// <list type="bullet">
    ///   <item>2D x 2D: [M, N] @ [N, P] = [M, P] (standard matrix multiplication)</item>
    ///   <item>3D x 2D: [B, M, N] @ [N, P] = [B, M, P] (batch matmul, weights broadcasted)</item>
    ///   <item>ND x 2D: [..., M, N] @ [N, P] = [..., M, P] (any batch dims, weights broadcasted)</item>
    ///   <item>3D x 3D: [B, M, N] @ [B, N, P] = [B, M, P] (batched matrix multiplication)</item>
    /// </list>
    /// <para><b>For Transformers:</b></para>
    /// <para>
    /// Input [batch, seq, features] @ weights [features, output] = [batch, seq, output]
    /// </para>
    /// </remarks>
    Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Performs 2D max pooling with asymmetric pool size and stride, returning max indices for backpropagation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="maxIndices">Output: indices of max elements for backpropagation.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices);

    /// <summary>
    /// Computes the gradient of MaxPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="maxIndices">The max indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">The pool size used in forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs 2D average pooling with asymmetric pool size and stride.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride);

    /// <summary>
    /// Computes the gradient of AvgPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">The pool size used in forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
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
    /// Performs deformable 2D convolution (DCNv2) with learned spatial offsets and modulation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="offset">The learned offset tensor [batch, 2*kernel_h*kernel_w, out_h, out_w].
    /// Each kernel position has (dy, dx) offsets for bilinear sampling.</param>
    /// <param name="mask">Optional modulation mask [batch, kernel_h*kernel_w, out_h, out_w].
    /// Values in [0,1] modulate each kernel position. Null uses no modulation (DCNv1).</param>
    /// <param name="stride">The stride [strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-DCN: Deformable Convolution v2</b></para>
    /// <para>
    /// Deformable convolution learns spatial deformations to adapt the sampling grid.
    /// Unlike standard convolution which samples on a fixed grid, deformable convolution
    /// adds learned 2D offsets to each sampling position, enabling the network to:
    /// - Focus on relevant parts of objects regardless of their shape
    /// - Handle geometric transformations (rotation, scale, deformation)
    /// - Adapt receptive fields to object structure
    /// </para>
    /// <para><b>DCNv2 Features (this implementation):</b></para>
    /// <list type="bullet">
    ///   <item>Per-sample offsets learned via backpropagation</item>
    ///   <item>Modulation mask to weight each sampling position (attention mechanism)</item>
    ///   <item>Bilinear interpolation for sub-pixel sampling</item>
    /// </list>
    /// <para><b>Applications:</b></para>
    /// <list type="bullet">
    ///   <item>Object detection (DETR, deformable attention)</item>
    ///   <item>Video super-resolution (aligning features across frames)</item>
    ///   <item>Optical flow estimation (SpyNet, PWC-Net)</item>
    ///   <item>Semantic segmentation with geometric adaptation</item>
    /// </list>
    /// <para>
    /// GPU acceleration provides 20-100x speedup due to parallel bilinear sampling.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor.</returns>
    Tensor<T> DeformableConv2DBackwardInput<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the kernel (weights).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="kernelShape">The shape of the kernel tensor.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel tensor.</returns>
    Tensor<T> DeformableConv2DBackwardKernel<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the offset tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the offset tensor.</returns>
    /// <remarks>
    /// <para>
    /// This gradient enables learning of the spatial deformations.
    /// The gradient flows through the bilinear interpolation operation.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2DBackwardOffset<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the modulation mask.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the modulation mask tensor.</returns>
    /// <remarks>
    /// <para>
    /// This gradient enables learning of per-position attention weights (DCNv2).
    /// Returns zero tensor if mask was null in forward pass.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2DBackwardMask<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of GridSample with respect to the input (NHWC format).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, outH, outW, channels].</param>
    /// <param name="grid">The sampling grid from forward pass [batch, outH, outW, 2].</param>
    /// <param name="inputShape">The shape of the original input [batch, height, width, channels].</param>
    /// <returns>The gradient with respect to the input tensor [batch, height, width, channels].</returns>
    Tensor<T> GridSampleBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> grid, int[] inputShape);

    /// <summary>
    /// Computes the gradient of GridSample with respect to the grid (NHWC format).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, outH, outW, channels].</param>
    /// <param name="input">The original input tensor [batch, height, width, channels].</param>
    /// <param name="grid">The sampling grid from forward pass [batch, outH, outW, 2].</param>
    /// <returns>The gradient with respect to the grid tensor [batch, outH, outW, 2].</returns>
    Tensor<T> GridSampleBackwardGrid<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> grid);

    #endregion

    #region 3D Convolution and Pooling Operations

    /// <summary>
    /// Performs 3D convolution on a 5D tensor for volumetric data processing.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride of the convolution (applied equally to all spatial dimensions).</param>
    /// <param name="padding">The zero-padding to add to all sides (applied equally to all spatial dimensions).</param>
    /// <param name="dilation">The spacing between kernel elements (applied equally to all spatial dimensions).</param>
    /// <returns>The convolved tensor [batch, out_channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-030: Conv3D</b></para>
    /// <para>
    /// 3D convolution extends 2D convolution to volumetric data by applying learnable filters
    /// across depth, height, and width dimensions. This is essential for:
    /// - Voxel-based 3D object recognition (ModelNet, ShapeNet)
    /// - Medical imaging analysis (CT scans, MRI volumes)
    /// - Video understanding (treating time as the third spatial dimension)
    /// - Point cloud processing after voxelization
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_depth = floor((depth + 2*padding - dilation*(kernel_depth-1) - 1) / stride) + 1
    /// output_height = floor((height + 2*padding - dilation*(kernel_height-1) - 1) / stride) + 1
    /// output_width = floor((width + 2*padding - dilation*(kernel_width-1) - 1) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 100-1000x speedup for typical 3D CNN layers due to the
    /// cubic growth in computation compared to 2D convolutions.
    /// </para>
    /// <para><b>For Beginners:</b> Think of 3D convolution as sliding a small 3D cube (the kernel)
    /// through a larger 3D volume (the input), computing dot products at each position.
    /// This allows the network to learn 3D patterns like surfaces, edges, and volumetric shapes.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Performs 3D convolution with asymmetric stride, padding, and dilation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padD, padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid or stride/padding/dilation arrays have incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This overload allows different stride, padding, and dilation values for each spatial dimension,
    /// providing more flexibility for architectures that need asymmetric operations.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv3D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, out_depth, out_height, out_width].</param>
    /// <param name="kernel">The convolution kernel used in forward pass [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="inputShape">The shape of the original input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padD, padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor [batch, in_channels, depth, height, width].</returns>
    /// <remarks>
    /// <para>
    /// This is used during backpropagation to compute how the loss changes with respect to the input.
    /// The operation is mathematically a transposed convolution (deconvolution) of the gradient
    /// with the kernel rotated 180 degrees in each spatial dimension.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv3D with respect to the kernel (weights).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, out_depth, out_height, out_width].</param>
    /// <param name="input">The original input tensor from forward pass [batch, in_channels, depth, height, width].</param>
    /// <param name="kernelShape">The shape of the kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padD, padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</returns>
    /// <remarks>
    /// <para>
    /// This is used during backpropagation to compute how the loss changes with respect to the weights.
    /// The operation is mathematically a convolution between the input and the output gradient.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Performs 3D max pooling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (applied equally to all spatial dimensions).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 5D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-031: MaxPool3D</b></para>
    /// <para>
    /// Max pooling downsamples the spatial dimensions by taking the maximum value
    /// in each 3D pooling window. Commonly used in 3D CNNs for:
    /// - Reducing spatial dimensions of volumetric data
    /// - Providing translation invariance in 3D
    /// - Reducing computation in deeper layers
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_d = floor((depth + 2*padding - poolSize) / stride) + 1
    /// output_h = floor((height + 2*padding - poolSize) / stride) + 1
    /// output_w = floor((width + 2*padding - poolSize) / stride) + 1
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 3D max pooling with asymmetric pool size, stride, and padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    Tensor<T> MaxPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Performs 3D max pooling and returns the indices of the maximum values.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="maxIndices">Output parameter containing the 3D coordinates (d, h, w) of maximum values for each output position.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <remarks>
    /// <para>
    /// The maxIndices array stores the 3D coordinates (depth, height, width) within the input tensor
    /// where the maximum value was found for each output position. This is essential for the backward
    /// pass to route gradients correctly.
    /// Shape of maxIndices: [batch, channels, output_depth, output_height, output_width, 3]
    /// where the last dimension contains [max_d_index, max_h_index, max_w_index].
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,,] maxIndices);

    /// <summary>
    /// Computes the gradient of MaxPool3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="maxIndices">The indices of maximum values from the forward pass.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are routed only to the positions that had the maximum
    /// values in the forward pass (as indicated by maxIndices). All other positions receive zero gradient.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3DBackward<T>(Tensor<T> gradOutput, int[,,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs 3D average pooling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (applied equally to all spatial dimensions).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 5D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-032: AvgPool3D</b></para>
    /// <para>
    /// Average pooling downsamples the spatial dimensions by computing the average value
    /// in each 3D pooling window. Compared to max pooling:
    /// - Smoother gradients during backpropagation
    /// - Better for preserving overall magnitude information
    /// - Often used in later layers or for global pooling
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 3D average pooling with asymmetric pool size, stride, and padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    Tensor<T> AvgPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of AvgPool3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, each gradient value from the output is divided equally among
    /// all the input positions that contributed to that output (i.e., divided by the pool volume).
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Performs 3D nearest-neighbor upsampling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleD">The depth scaling factor.</param>
    /// <param name="scaleH">The height scaling factor.</param>
    /// <param name="scaleW">The width scaling factor.</param>
    /// <returns>The upsampled tensor with shape [batch, channels, depth*scaleD, height*scaleH, width*scaleW].</returns>
    /// <remarks>
    /// <para><b>US-GPU-035: Upsample3D</b></para>
    /// <para>
    /// 3D upsampling increases the spatial dimensions of volumetric data by replicating values.
    /// This is essential for decoder paths in encoder-decoder architectures like 3D U-Net.
    /// </para>
    /// </remarks>
    Tensor<T> Upsample3D<T>(Tensor<T> input, int scaleD, int scaleH, int scaleW);

    /// <summary>
    /// Computes the backward pass for 3D upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer [batch, channels, out_depth, out_height, out_width].</param>
    /// <param name="inputShape">The original input shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleD">The depth scaling factor used in forward pass.</param>
    /// <param name="scaleH">The height scaling factor used in forward pass.</param>
    /// <param name="scaleW">The width scaling factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are accumulated from all output positions that were
    /// derived from each input position (i.e., summed over the scaling block).
    /// </para>
    /// </remarks>
    Tensor<T> Upsample3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleD, int scaleH, int scaleW);

    /// <summary>
    /// Performs 3D transposed convolution (deconvolution) for learned upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, out_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <param name="outputPadding">Output padding for size adjustment [outPadD, outPadH, outPadW].</param>
    /// <returns>The upsampled tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-036: ConvTranspose3D</b></para>
    /// <para>
    /// Transposed 3D convolution learns upsampling filters, providing more flexibility than
    /// nearest-neighbor upsampling. Used in decoder paths of 3D U-Net and similar architectures.
    /// </para>
    /// </remarks>
    Tensor<T> ConvTranspose3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding);

    /// <summary>
    /// Computes the gradient of ConvTranspose3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="kernel">The kernel tensor used in forward pass.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ConvTranspose3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of ConvTranspose3D with respect to the kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernelShape">The shape of the kernel.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel.</returns>
    Tensor<T> ConvTranspose3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

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
    /// Applies Gumbel-Softmax activation to produce differentiable categorical samples.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor of logits.</param>
    /// <param name="temperature">Temperature parameter controlling the softness. Must be positive.</param>
    /// <param name="hard">If true, uses straight-through estimator for discrete outputs.</param>
    /// <param name="axis">The axis along which to apply Gumbel-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Gumbel-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// Gumbel-Softmax provides a differentiable approximation to categorical sampling.
    /// As temperature approaches 0, outputs approach one-hot categorical samples.
    /// When hard=true, uses straight-through estimator for discrete outputs with gradient pass-through.
    /// </para>
    /// </remarks>
    Tensor<T> GumbelSoftmax<T>(Tensor<T> input, double temperature = 1.0, bool hard = false, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Gumbel-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward Gumbel-Softmax pass.</param>
    /// <param name="temperature">Temperature parameter used in forward pass.</param>
    /// <param name="axis">The axis along which Gumbel-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> GumbelSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, double temperature, int axis = -1);

    /// <summary>
    /// Applies Taylor-Softmax activation using polynomial approximation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="order">The order of Taylor expansion. Default is 2.</param>
    /// <param name="axis">The axis along which to apply Taylor-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Taylor-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// TaylorSoftmax uses Taylor series approximation of exp(x):
    /// exp(x) Ã¢â€°Ë† 1 + x + xÃ‚Â²/2! + xÃ‚Â³/3! + ... + xÃ¢ÂÂ¿/n!
    /// Then normalizes like standard softmax.
    /// More computationally efficient than standard softmax for some hardware.
    /// </para>
    /// </remarks>
    Tensor<T> TaylorSoftmax<T>(Tensor<T> input, int order = 2, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Taylor-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="output">The output from the forward Taylor-Softmax pass.</param>
    /// <param name="order">The order of Taylor expansion used in forward pass.</param>
    /// <param name="axis">The axis along which Taylor-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> TaylorSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int order, int axis = -1);

    /// <summary>
    /// Applies Sparsemax activation to produce sparse probability distributions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply Sparsemax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Sparsemax applied.</returns>
    /// <remarks>
    /// <para>
    /// Sparsemax produces sparse probability distributions where some outputs are exactly zero.
    /// Unlike softmax which always gives positive probabilities to all classes, sparsemax
    /// can assign exactly zero to low-scoring classes.
    /// </para>
    /// </remarks>
    Tensor<T> Sparsemax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Sparsemax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward Sparsemax pass (used to determine support set).</param>
    /// <param name="axis">The axis along which Sparsemax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SparsemaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Applies Spherical-Softmax activation (L2-normalized softmax).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply Spherical-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Spherical-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// SphericalSoftmax = softmax(x / ||x||Ã¢â€šâ€š)
    /// First L2-normalizes the input, then applies softmax.
    /// This improves numerical stability for inputs with varying magnitudes.
    /// </para>
    /// </remarks>
    Tensor<T> SphericalSoftmax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Spherical-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="output">The output from the forward Spherical-Softmax pass.</param>
    /// <param name="axis">The axis along which Spherical-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SphericalSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1);

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
    /// Applies layer normalization to a tensor of any rank, normalizing over the last N dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [d0, d1, ..., dn] of any rank.</param>
    /// <param name="gamma">Scale parameter with shape matching the last N dimensions to normalize over.
    /// For example, if input is [batch, seq, embed] and gamma is [embed], normalizes over the last dimension.
    /// If gamma is [seq, embed], normalizes over the last two dimensions.</param>
    /// <param name="beta">Shift parameter with the same shape as gamma.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean with shape [d0, d1, ..., d(n-N)] where N is the number of normalized dimensions.</param>
    /// <param name="variance">Output: computed variance with the same shape as mean.</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// This follows the industry standard (PyTorch/TensorFlow) behavior for layer normalization:
    /// - Supports tensors of any rank (2D, 3D, 4D, etc.)
    /// - Normalizes over the last N dimensions as determined by gamma's shape
    /// - Each position in the preceding dimensions is normalized independently
    /// </para>
    /// <para><b>Examples:</b>
    /// - Input [32, 64] with gamma [64]: normalizes each of 32 samples over 64 features
    /// - Input [2, 10, 64] with gamma [64]: normalizes each of 20 positions over 64 features
    /// - Input [2, 10, 64] with gamma [10, 64]: normalizes each of 2 batches over 640 features
    /// </para>
    /// </remarks>
    Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for layer normalization on tensors of any rank.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer (same shape as forward output).</param>
    /// <param name="input">The original input tensor of any rank.</param>
    /// <param name="gamma">Scale parameter with shape matching normalized dimensions.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma (same shape as gamma).</param>
    /// <param name="gradBeta">Output: gradient with respect to beta (same shape as beta).</param>
    /// <returns>The gradient with respect to the input (same shape as input).</returns>
    /// <remarks>
    /// This backward pass supports the same any-rank tensor semantics as the forward pass.
    /// </remarks>
    Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    /// <summary>
    /// Applies group normalization to a tensor with shape [batch, channels, ...spatial].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width] or [batch, channels].</param>
    /// <param name="numGroups">The number of groups to divide channels into.</param>
    /// <param name="gamma">Scale parameter with shape [channels].</param>
    /// <param name="beta">Shift parameter with shape [channels].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean per group with shape [batch, numGroups].</param>
    /// <param name="variance">Output: computed variance per group with shape [batch, numGroups].</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    Tensor<T> GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for group normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="numGroups">The number of groups used during forward pass.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> GroupNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    #endregion

    #region Tensor Reduction Operations

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

    /// <summary>
    /// Computes the variance of tensor elements along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the variance.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The tensor containing variance values.</returns>
    Tensor<T> ReduceVariance<T>(Tensor<T> input, int[] axes, bool keepDims);

    /// <summary>
    /// Computes the backward pass for reduce variance.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="mean">The mean values computed during forward pass.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes);

    /// <summary>
    /// Computes the natural logarithm of variance of tensor elements along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the log variance.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <param name="epsilon">Small value for numerical stability (prevents log(0)).</param>
    /// <returns>The tensor containing log variance values.</returns>
    Tensor<T> ReduceLogVariance<T>(Tensor<T> input, int[] axes, bool keepDims, double epsilon = 1e-8);

    /// <summary>
    /// Computes the backward pass for reduce log variance.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="mean">The mean values computed during forward pass.</param>
    /// <param name="variance">The variance values computed during forward pass.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceLogVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> variance, int[] axes);

    #endregion

    #region Spatial Operations

    /// <summary>
    /// Performs nearest-neighbor upsampling on a tensor of any rank (at least 2D).
    /// The last two dimensions are treated as height and width for upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with at least 2 dimensions, where the last two are height and width.
    /// Supports: 2D [H, W], 3D [C, H, W], 4D [B, C, H, W], 5D+.</param>
    /// <param name="scaleH">The height scaling factor.</param>
    /// <param name="scaleW">The width scaling factor.</param>
    /// <returns>The upsampled tensor with scaled height and width dimensions.</returns>
    Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW);

    /// <summary>
    /// Computes the backward pass for upsampling on a tensor of any rank (at least 2D).
    /// The last two dimensions are treated as height and width.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape (any rank, at least 2D).</param>
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
    /// Generates a normalized affine grid for spatial transformer sampling.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="theta">Affine matrices of shape [batch, 2, 3].</param>
    /// <param name="outputHeight">Target grid height.</param>
    /// <param name="outputWidth">Target grid width.</param>
    /// <returns>Grid tensor of shape [batch, outputHeight, outputWidth, 2] in [-1, 1] normalized coords.</returns>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT: Layout Note</b> - This method and <see cref="GridSample{T}"/> use NHWC layout
    /// [batch, height, width, channels/coords], which differs from Conv2D, MaxPool2D, and other
    /// spatial operations that use NCHW layout [batch, channels, height, width].
    /// </para>
    /// <para>
    /// When using these methods with NCHW tensors, you must transpose:
    /// <code>
    /// // NCHW to NHWC before GridSample
    /// var inputNHWC = input.Transpose([0, 2, 3, 1]);
    /// var output = engine.GridSample(inputNHWC, grid);
    /// // NHWC to NCHW after GridSample
    /// var outputNCHW = output.Transpose([0, 3, 1, 2]);
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> AffineGrid<T>(Tensor<T> theta, int outputHeight, int outputWidth);

    /// <summary>
    /// Samples an input tensor using a normalized grid with bilinear interpolation.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="input">Input tensor [batch, height, width, channels] (NHWC format).</param>
    /// <param name="grid">Sampling grid [batch, outH, outW, 2] with coords in [-1, 1].</param>
    /// <returns>Sampled output tensor [batch, outH, outW, channels] (NHWC format).</returns>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT: Layout Note</b> - This method uses NHWC layout [batch, height, width, channels],
    /// which differs from Conv2D, MaxPool2D, and other spatial operations that use NCHW layout
    /// [batch, channels, height, width]. Ensure inputs are transposed appropriately.
    /// </para>
    /// <para>
    /// The grid coordinates are normalized to [-1, 1] range where (-1, -1) is the top-left corner
    /// and (1, 1) is the bottom-right corner of the input tensor.
    /// </para>
    /// </remarks>
    Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid);

    /// <summary>
    /// Performs complex-valued matrix multiplication using split real/imaginary tensors.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="aReal">Real part of left matrix [M, K].</param>
    /// <param name="aImag">Imag part of left matrix [M, K].</param>
    /// <param name="bReal">Real part of right matrix [K, N].</param>
    /// <param name="bImag">Imag part of right matrix [K, N].</param>
    /// <returns>(real, imag) tuple representing the product [M, N].</returns>
    (Tensor<T> real, Tensor<T> imag) ComplexMatMul<T>(Tensor<T> aReal, Tensor<T> aImag, Tensor<T> bReal, Tensor<T> bImag);

    /// <summary>
    /// Computes magnitude squared of a complex tensor given real/imag split.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="real">Real part tensor.</param>
    /// <param name="imag">Imag part tensor.</param>
    /// <returns>Tensor of magnitudes squared.</returns>
    Tensor<T> ComplexMagnitudeSquared<T>(Tensor<T> real, Tensor<T> imag);

    /// <summary>
    /// Normalizes a complex tensor (real/imag split) so that sum(|z|^2) = 1.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="real">Real part tensor.</param>
    /// <param name="imag">Imag part tensor.</param>
    /// <returns>(real, imag) tuple of normalized complex tensor.</returns>
    (Tensor<T> real, Tensor<T> imag) ComplexNormalize<T>(Tensor<T> real, Tensor<T> imag);

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

    /// <summary>
    /// Computes the sum of squares of all elements in a tensor (L2 norm squared).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar sum of squared elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Computes ÃŽÂ£(x_iÃ‚Â²) for all elements in the tensor. Used in:
    /// - L2 regularization loss computation
    /// - Frobenius norm calculation (sqrt of sum of squares)
    /// - Gradient magnitude computation
    /// - Weight decay penalties
    /// </para>
    /// <para>
    /// GPU acceleration provides significant speedup for large tensors.
    /// </para>
    /// </remarks>
    T TensorSumOfSquares<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs embedding lookup - gathers rows from an embedding table based on indices.
    /// </summary>
    /// <typeparam name="TValue">The numeric type of embedding values.</typeparam>
    /// <typeparam name="TIndex">The integer type for indices (must be unmanaged).</typeparam>
    /// <param name="embeddings">The embedding table tensor [vocab_size, embedding_dim].</param>
    /// <param name="indices">The indices tensor containing token IDs.</param>
    /// <returns>The gathered embeddings with shape [*indices.shape, embedding_dim].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: Embedding Operations</b></para>
    /// <para>
    /// Embedding lookup is a fundamental operation for NLP and sequence models:
    /// - Word/token embeddings in language models
    /// - Item embeddings in recommendation systems
    /// - Categorical feature embeddings
    /// </para>
    /// <para>
    /// For each index i in indices, retrieves embeddings[i, :] and places it in the output.
    /// GPU acceleration provides significant speedup for large vocabularies and batch sizes.
    /// </para>
    /// </remarks>
    Tensor<TValue> TensorEmbeddingLookup<TValue, TIndex>(Tensor<TValue> embeddings, Tensor<TIndex> indices)
        where TIndex : unmanaged;

    /// <summary>
    /// Performs embedding lookup backward pass - scatters gradients back to embedding table.
    /// </summary>
    /// <typeparam name="TValue">The numeric type of gradient and embedding values.</typeparam>
    /// <typeparam name="TIndex">The integer type for indices (must be unmanaged).</typeparam>
    /// <param name="gradOutput">The gradient from the next layer [*indices.shape, embedding_dim].</param>
    /// <param name="indices">The indices tensor containing token IDs.</param>
    /// <param name="vocabSize">The vocabulary size (number of rows in embedding table).</param>
    /// <param name="embeddingDim">The embedding dimension.</param>
    /// <returns>The gradient for the embedding table [vocab_size, embedding_dim].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: Embedding Operations</b></para>
    /// <para>
    /// Computes the gradient for embedding parameters by accumulating gradients for each index.
    /// For each index i, adds gradOutput[position] to embeddingGrad[i, :].
    /// Handles duplicate indices by accumulating their gradients.
    /// </para>
    /// </remarks>
    Tensor<TValue> TensorEmbeddingLookupBackward<TValue, TIndex>(Tensor<TValue> gradOutput, Tensor<TIndex> indices, int vocabSize, int embeddingDim)
        where TIndex : unmanaged;

    /// <summary>
    /// Computes the Radial Basis Function (RBF) kernel between input samples and centers.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, features].</param>
    /// <param name="centers">The RBF center positions with shape [numCenters, features].</param>
    /// <param name="epsilons">The epsilon values (1/(2*widthÃ‚Â²)) for each center with shape [numCenters].</param>
    /// <returns>The RBF kernel output with shape [batch, numCenters], computing exp(-epsilon * ||x - center||Ã‚Â²).</returns>
    /// <remarks>
    /// <para>
    /// Computes Gaussian RBF: K(x, c) = exp(-epsilon * ||x - c||Ã‚Â²) where:
    /// - x is an input sample
    /// - c is a center
    /// - epsilon = 1/(2*widthÃ‚Â²) controls the spread
    /// </para>
    /// <para><b>For Beginners:</b> RBF kernels measure similarity between points.
    /// Points close to a center produce values near 1, distant points produce values near 0.
    /// </para>
    /// </remarks>
    Tensor<T> RBFKernel<T>(Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons);

    /// <summary>
    /// Computes the backward pass for the RBF kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer with shape [batch, numCenters].</param>
    /// <param name="input">The original input tensor with shape [batch, features].</param>
    /// <param name="centers">The RBF center positions with shape [numCenters, features].</param>
    /// <param name="epsilons">The epsilon values with shape [numCenters].</param>
    /// <param name="output">The output from the forward pass with shape [batch, numCenters].</param>
    /// <returns>A tuple containing gradients for (input, centers, epsilons).</returns>
    (Tensor<T> gradInput, Tensor<T> gradCenters, Tensor<T> gradEpsilons) RBFKernelBackward<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons, Tensor<T> output);

    #endregion

    #region Tensor Shape Operations

    /// <summary>
    /// Repeats each element of a tensor along the specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to repeat.</param>
    /// <param name="repeats">The number of times to repeat each element.</param>
    /// <param name="axis">The axis along which to repeat. Default is 0.</param>
    /// <returns>A tensor with elements repeated along the specified axis.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.repeat(). For a 1D tensor [a, b, c] with repeats=2:
    /// Result: [a, a, b, b, c, c]
    /// </para>
    /// <para><b>For Beginners:</b> This is useful for creating masks or expanding data
    /// where each element needs to be duplicated multiple times.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRepeatElements<T>(Tensor<T> tensor, int repeats, int axis = 0);

    /// <summary>
    /// Tiles (repeats) a tensor along each axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to tile.</param>
    /// <param name="multiples">The number of times to tile along each axis.</param>
    /// <returns>A tensor that is the input tiled according to multiples.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.tile(). For a tensor [a, b] with multiples=[3]:
    /// Result: [a, b, a, b, a, b]
    /// </para>
    /// <para>
    /// For a 2D tensor [[1, 2], [3, 4]] with multiples=[2, 3]:
    /// Result: [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
    /// </para>
    /// </remarks>
    Tensor<T> TensorTile<T>(Tensor<T> tensor, int[] multiples);

    /// <summary>
    /// Extracts a slice from a tensor along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to slice.</param>
    /// <param name="start">The starting indices for each axis.</param>
    /// <param name="length">The length to extract along each axis.</param>
    /// <returns>A tensor containing the sliced portion.</returns>
    /// <remarks>
    /// <para>
    /// This operation extracts a contiguous region from the tensor.
    /// For a 1D tensor [a, b, c, d, e] with start=[1] and length=[3]:
    /// Result: [b, c, d]
    /// </para>
    /// </remarks>
    Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length);

    /// <summary>
    /// Sets a slice of a tensor to values from another tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The tensor to modify (in-place or returns new tensor).</param>
    /// <param name="source">The tensor containing values to set.</param>
    /// <param name="start">The starting indices where to place the source tensor.</param>
    /// <returns>A tensor with the slice set to the source values.</returns>
    /// <remarks>
    /// <para>
    /// This operation sets values in a region of the destination tensor.
    /// Useful for building tensors piece by piece without manual loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSetSlice<T>(Tensor<T> destination, Tensor<T> source, int[] start);

    /// <summary>
    /// Creates a tensor by selecting elements based on a condition mask.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="condition">A boolean-like tensor where non-zero means true.</param>
    /// <param name="x">Values to select where condition is true.</param>
    /// <param name="y">Values to select where condition is false.</param>
    /// <returns>A tensor with elements from x where condition is true, else from y.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.where() or torch.where().
    /// Result[i] = condition[i] != 0 ? x[i] : y[i]
    /// </para>
    /// <para>
    /// <b>Note:</b> Prefer the overload that accepts <c>Tensor&lt;bool&gt;</c> for explicit boolean conditions.
    /// This overload treats any non-zero value as true, which may lead to unexpected behavior with floating-point types.
    /// </para>
    /// </remarks>
    Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y);

    #endregion

    #region Loop Elimination Operations

    /// <summary>
    /// Copies data from a source tensor to a destination tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="destination">The destination tensor to copy to.</param>
    /// <remarks>
    /// <para>
    /// This operation performs an in-place copy of tensor data.
    /// Both tensors must have the same total number of elements.
    /// </para>
    /// </remarks>
    void TensorCopy<T>(Tensor<T> source, Tensor<T> destination);

    /// <summary>
    /// Fills a tensor with a constant value.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to fill.</param>
    /// <param name="value">The value to fill with.</param>
    /// <remarks>
    /// <para>
    /// This operation sets all elements of the tensor to the specified value.
    /// Useful for initialization without manual loops.
    /// </para>
    /// </remarks>
    void TensorFill<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Computes the outer product of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor (typically a vector [N]).</param>
    /// <param name="b">The second tensor (typically a vector [M]).</param>
    /// <returns>A tensor containing the outer product [N, M].</returns>
    /// <remarks>
    /// <para>
    /// For vectors a[N] and b[M], produces a matrix [N, M] where result[i,j] = a[i] * b[j].
    /// This is useful for computing weight gradients: dW = x^T * dy.
    /// </para>
    /// </remarks>
    Tensor<T> TensorOuterProduct<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes batched outer products for all items in a batch.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor [batch, N].</param>
    /// <param name="b">The second tensor [batch, M].</param>
    /// <returns>A tensor containing the batched outer products [batch, N, M].</returns>
    /// <remarks>
    /// <para>
    /// For each batch item, computes result[b, i, j] = a[b, i] * b[b, j].
    /// Useful for batched gradient computations without explicit loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBatchOuterProduct<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Permutes the dimensions of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to permute.</param>
    /// <param name="axes">The new order of dimensions (e.g., [0, 2, 1] swaps last two dims).</param>
    /// <returns>A tensor with permuted dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This generalizes transpose to arbitrary dimension reordering.
    /// Similar to numpy.transpose() or torch.permute().
    /// </para>
    /// </remarks>
    Tensor<T> TensorPermute<T>(Tensor<T> tensor, int[] axes);

    /// <summary>
    /// Expands dimensions by inserting a new axis of size 1.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to expand.</param>
    /// <param name="axis">The position where to insert the new axis.</param>
    /// <returns>A tensor with an additional dimension of size 1.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.expand_dims() or torch.unsqueeze().
    /// Useful for broadcasting operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorExpandDims<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Removes singleton dimensions from a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to squeeze.</param>
    /// <param name="axis">The axis to remove (must have size 1). Use -1 to remove all singleton dims.</param>
    /// <returns>A tensor with the specified singleton dimension removed.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.squeeze() or torch.squeeze().
    /// Removes dimensions of size 1.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSqueeze<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>
    /// Performs scatter-add: adds values to specific indices in a destination tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor to add values to.</param>
    /// <param name="indices">The indices where to add values (integer tensor).</param>
    /// <param name="updates">The values to add at the specified indices.</param>
    /// <param name="axis">The axis along which to scatter.</param>
    /// <returns>A tensor with values added at the specified indices.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.scatter_add() or tf.tensor_scatter_nd_add().
    /// Useful for sparse gradient accumulation in embeddings.
    /// </para>
    /// </remarks>
    Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis = 0);

    /// <summary>
    /// Gathers values from a tensor along an axis using indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">The source tensor to gather from.</param>
    /// <param name="indices">The indices specifying which values to gather (integer tensor).</param>
    /// <param name="axis">The axis along which to gather.</param>
    /// <returns>A tensor containing the gathered values.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.gather() or tf.gather().
    /// Useful for index-based lookups without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorGather<T>(Tensor<T> source, Tensor<int> indices, int axis = 0);

    /// <summary>
    /// Computes a cumulative sum along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the cumulative sum.</param>
    /// <returns>A tensor where each element is the sum of all previous elements along the axis.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.cumsum() or torch.cumsum().
    /// Useful for CRF forward-backward and other sequence operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes a log-sum-exp reduction along an axis (numerically stable).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute logsumexp.</param>
    /// <param name="keepDims">Whether to keep the reduced dimension.</param>
    /// <returns>A tensor with the log-sum-exp values.</returns>
    /// <remarks>
    /// <para>
    /// Computes log(sum(exp(x))) in a numerically stable way.
    /// Essential for CRF partition functions and attention mechanisms.
    /// </para>
    /// </remarks>
    Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims = false);

    /// <summary>
    /// Generates a tensor filled with random values from a uniform distribution [0, 1).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <returns>A tensor filled with random uniform values.</returns>
    /// <remarks>
    /// <para>
    /// Useful for weight initialization and dropout masks without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomUniform<T>(int[] shape);

    /// <summary>
    /// Generates a tensor filled with random values from a normal distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="mean">The mean of the distribution.</param>
    /// <param name="stddev">The standard deviation of the distribution.</param>
    /// <returns>A tensor filled with random normal values.</returns>
    /// <remarks>
    /// <para>
    /// Useful for Xavier/He initialization without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomNormal<T>(int[] shape, T mean, T stddev);

    /// <summary>
    /// Generates a tensor filled with random values from a uniform distribution within a specified range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="min">The minimum value (inclusive).</param>
    /// <param name="max">The maximum value (exclusive).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tensor filled with random values in [min, max).</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Random Operations</b></para>
    /// <para>
    /// Used for weight initialization with specific ranges (e.g., Xavier uniform [-limit, limit]),
    /// embedding initialization, and data augmentation. More flexible than TensorRandomUniform
    /// which only generates values in [0, 1).
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomUniformRange<T>(int[] shape, T min, T max, int? seed = null);

    /// <summary>
    /// Generates a dropout mask tensor where each element is either zero or a scale value.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0 to 1).</param>
    /// <param name="scale">The scale factor for non-dropped elements (typically 1/(1-dropoutRate)).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tensor containing the dropout mask.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Dropout Operations</b></para>
    /// <para>
    /// Used in dropout layers during training. Elements are randomly set to zero with probability
    /// dropoutRate, and remaining elements are scaled to maintain expected values.
    /// The mask can be multiplied element-wise with activations: output = input * mask.
    /// </para>
    /// </remarks>
    Tensor<T> TensorDropoutMask<T>(int[] shape, T dropoutRate, T scale, int? seed = null);

    /// <summary>
    /// Subtracts a tensor from a scalar value element-wise (scalar - tensor).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="scalar">The scalar value to subtract from.</param>
    /// <param name="tensor">The tensor to subtract.</param>
    /// <returns>A tensor with (scalar - x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Arithmetic Operations</b></para>
    /// <para>
    /// Used for computing (1 - p) in probability calculations, BCE loss gradients,
    /// and other operations where the subtraction order matters.
    /// More efficient than TensorNegate(TensorSubtractScalar(tensor, scalar)).
    /// </para>
    /// </remarks>
    Tensor<T> ScalarMinusTensor<T>(T scalar, Tensor<T> tensor);

    /// <summary>
    /// Creates an identity matrix as a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>A tensor [size, size] with 1s on diagonal and 0s elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Useful for initializing transformation matrices and attention masks.
    /// </para>
    /// </remarks>
    Tensor<T> TensorEye<T>(int size);

    /// <summary>
    /// Creates a diagonal tensor from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="diagonal">The values to place on the diagonal.</param>
    /// <returns>A tensor with the diagonal values and zeros elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.diag() or torch.diag().
    /// </para>
    /// </remarks>
    Tensor<T> TensorDiag<T>(Tensor<T> diagonal);

    /// <summary>
    /// Extracts the diagonal from a matrix tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input matrix tensor.</param>
    /// <returns>A 1D tensor containing the diagonal elements.</returns>
    /// <remarks>
    /// <para>
    /// Extracts diagonal[i] = tensor[i, i] without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorDiagonal<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies einsum (Einstein summation) notation for flexible tensor contractions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="subscripts">The einsum subscript notation (e.g., "ij,jk->ik" for matmul).</param>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The result of the einsum operation.</returns>
    /// <remarks>
    /// <para>
    /// Einsum is a powerful notation for expressing tensor operations.
    /// Common patterns:
    /// - "ij,jk->ik": matrix multiplication
    /// - "bij,bjk->bik": batched matrix multiplication
    /// - "bijk,bkl->bijl": batched tensor contraction
    /// - "bi,bj->bij": batched outer product
    /// </para>
    /// </remarks>
    Tensor<T> TensorEinsum<T>(string subscripts, params Tensor<T>[] tensors);

    /// <summary>
    /// Adds a scalar to all elements of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar to add.</param>
    /// <returns>A tensor with the scalar added to all elements.</returns>
    Tensor<T> TensorAddScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Subtracts a scalar from all elements of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar to subtract.</param>
    /// <returns>A tensor with the scalar subtracted from all elements.</returns>
    Tensor<T> TensorSubtractScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Divides all elements of a tensor by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar divisor.</param>
    /// <returns>A tensor with all elements divided by the scalar.</returns>
    Tensor<T> TensorDivideScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Applies the hyperbolic tangent derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tanhOutput">The output of tanh (not the input).</param>
    /// <returns>A tensor containing 1 - tanh(x)^2 for each element.</returns>
    /// <remarks>
    /// <para>
    /// Given y = tanh(x), the derivative is 1 - y^2.
    /// This takes the tanh output directly to avoid recomputation.
    /// </para>
    /// </remarks>
    Tensor<T> TanhDerivative<T>(Tensor<T> tanhOutput);

    /// <summary>
    /// Applies the sigmoid derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="sigmoidOutput">The output of sigmoid (not the input).</param>
    /// <returns>A tensor containing sigmoid(x) * (1 - sigmoid(x)) for each element.</returns>
    /// <remarks>
    /// <para>
    /// Given y = sigmoid(x), the derivative is y * (1 - y).
    /// This takes the sigmoid output directly to avoid recomputation.
    /// </para>
    /// </remarks>
    Tensor<T> SigmoidDerivative<T>(Tensor<T> sigmoidOutput);

    /// <summary>
    /// Applies the ReLU derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The original input to ReLU.</param>
    /// <returns>A tensor containing 1 where input > 0, else 0.</returns>
    Tensor<T> ReLUDerivative<T>(Tensor<T> input);

    /// <summary>
    /// Creates a triangular mask tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="size">The size of the square mask.</param>
    /// <param name="upper">If true, creates upper triangular; otherwise lower triangular.</param>
    /// <param name="diagonal">Offset from the main diagonal (0 = include diagonal).</param>
    /// <returns>A tensor with 1s in the triangular region and 0s elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Useful for causal attention masks without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTriangularMask<T>(int size, bool upper = false, int diagonal = 0);

    /// <summary>
    /// Applies element-wise squash function for capsule networks.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the norm.</param>
    /// <returns>A tensor with vectors squashed to have magnitude in [0, 1).</returns>
    /// <remarks>
    /// <para>
    /// Implements squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)
    /// Used in capsule networks to ensure output vectors have bounded magnitude.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSquash<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>
    /// Computes the backward pass for squash function.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient of the loss with respect to squash output.</param>
    /// <param name="input">The original input to squash.</param>
    /// <param name="output">The output of squash.</param>
    /// <param name="axis">The axis along which squash was computed.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Computes the L2 norm along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the norm.</param>
    /// <param name="keepDims">Whether to keep the reduced dimension.</param>
    /// <returns>A tensor containing the L2 norms.</returns>
    Tensor<T> TensorNorm<T>(Tensor<T> tensor, int axis, bool keepDims = false);

    /// <summary>
    /// Normalizes vectors along an axis to unit length.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to normalize.</param>
    /// <param name="epsilon">Small value to prevent division by zero.</param>
    /// <returns>A tensor with vectors normalized to unit length.</returns>
    Tensor<T> TensorNormalize<T>(Tensor<T> tensor, int axis, T epsilon);

    /// <summary>
    /// Clips tensor values to a range. This is an alias for <see cref="TensorClamp{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <returns>A tensor with values clipped to [minValue, maxValue].</returns>
    /// <remarks>
    /// <para>
    /// This method provides the same functionality as <see cref="TensorClamp{T}"/>.
    /// Both "clip" and "clamp" are common names for the same operation (min(max(x, min), max)).
    /// </para>
    /// </remarks>
    Tensor<T> TensorClip<T>(Tensor<T> tensor, T minValue, T maxValue);

    /// <summary>
    /// Creates a tensor by concatenating tensors along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>A tensor containing the concatenated tensors.</returns>
    Tensor<T> TensorConcatenate<T>(Tensor<T>[] tensors, int axis = 0);

    /// <summary>
    /// Splits a tensor into multiple tensors along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to split.</param>
    /// <param name="numSplits">The number of equal splits.</param>
    /// <param name="axis">The axis along which to split.</param>
    /// <returns>An array of tensors.</returns>
    Tensor<T>[] TensorSplit<T>(Tensor<T> tensor, int numSplits, int axis = 0);

    /// <summary>
    /// Creates a one-hot encoded tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the output tensor elements.</typeparam>
    /// <param name="indices">The indices tensor (must be integer values).</param>
    /// <param name="depth">The number of classes (size of one-hot dimension).</param>
    /// <returns>A tensor with one-hot encoding.</returns>
    /// <remarks>
    /// <para>
    /// For indices [0, 2, 1] and depth 3:
    /// [[1,0,0], [0,0,1], [0,1,0]]
    /// </para>
    /// <para>
    /// Note: This is a breaking API change. Indices must now be Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<T> TensorOneHot<T>(Tensor<int> indices, int depth);

    /// <summary>
    /// Computes argmax along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of input tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to find the maximum index.</param>
    /// <returns>A tensor containing the integer indices of maximum values.</returns>
    /// <remarks>
    /// <para>
    /// Note: This is a breaking API change. Return type is now Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<int> TensorArgMax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes argmin along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of input tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to find the minimum index.</param>
    /// <returns>A tensor containing the integer indices of minimum values.</returns>
    /// <remarks>
    /// <para>
    /// Note: This is a breaking API change. Return type is now Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<int> TensorArgMin<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes binary cross-entropy loss element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted probabilities (0-1).</param>
    /// <param name="targets">The target values (0 or 1).</param>
    /// <param name="epsilon">Small value for numerical stability.</param>
    /// <returns>A tensor containing the loss for each element.</returns>
    Tensor<T> TensorBinaryCrossEntropy<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon);

    /// <summary>
    /// Computes the backward pass for binary cross-entropy.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted probabilities.</param>
    /// <param name="targets">The target values.</param>
    /// <param name="epsilon">Small value for numerical stability.</param>
    /// <returns>The gradient with respect to predictions.</returns>
    Tensor<T> TensorBinaryCrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon);

    /// <summary>
    /// Creates coordinate meshgrids from 1D coordinate arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">1D array of x coordinates [width].</param>
    /// <param name="y">1D array of y coordinates [height].</param>
    /// <returns>A tuple of (X, Y) grids, each with shape [height, width].</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.meshgrid() or torch.meshgrid().
    /// Creates 2D coordinate grids from 1D coordinate arrays.
    /// Useful for spatial transformer networks and coordinate-based operations.
    /// </para>
    /// </remarks>
    (Tensor<T> X, Tensor<T> Y) TensorMeshgrid<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Extracts a slice along a specific axis from a 3D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor [dim0, dim1, dim2].</param>
    /// <param name="axis">The axis to slice (0, 1, or 2).</param>
    /// <param name="index">The index to extract along the axis.</param>
    /// <returns>A 2D tensor with the specified slice.</returns>
    /// <remarks>
    /// <para>
    /// For a 3D tensor [H, W, C], slicing along axis 2 with index i gives [H, W] at channel i.
    /// This is useful for extracting channels from multi-channel tensors without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSliceAxis<T>(Tensor<T> tensor, int axis, int index);

    /// <summary>
    /// Creates a tensor filled with values from a linear range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="start">The starting value of the range.</param>
    /// <param name="end">The ending value of the range (exclusive).</param>
    /// <param name="count">The number of values in the range.</param>
    /// <returns>A 1D tensor with linearly spaced values from start to end.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.linspace() or torch.linspace().
    /// Useful for creating coordinate ranges without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorLinspace<T>(T start, T end, int count);

    /// <summary>
    /// Batched matrix multiplication for 3D tensors. This is an alias for <see cref="BatchMatMul{T}"/>.
    /// Computes batched matrix multiply: result[b] = a[b] @ b[b] for each batch.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">First tensor of shape [batch, M, K].</param>
    /// <param name="b">Second tensor of shape [batch, K, N] or [K, N] for broadcasting.</param>
    /// <returns>Result tensor of shape [batch, M, N].</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.bmm() or np.matmul() with 3D tensors.
    /// Essential for RNN/LSTM/GRU vectorization where we compute all timesteps at once.
    /// If b is 2D [K, N], it broadcasts across the batch dimension.
    /// </para>
    /// <para>
    /// This method provides the same functionality as <see cref="BatchMatMul{T}"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBatchMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Sets a slice of a tensor along a specific axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor to modify.</param>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="axis">The axis along which to set the slice.</param>
    /// <param name="index">The index at which to set the slice.</param>
    /// <remarks>
    /// <para>
    /// Inverse of TensorSliceAxis. Sets destination[..., index, ...] = source along the specified axis.
    /// Essential for building output tensors without loops.
    /// </para>
    /// </remarks>
    void TensorSetSliceAxis<T>(Tensor<T> destination, Tensor<T> source, int axis, int index);

    /// <summary>
    /// Applies softmax along a specified axis. This is an alias for <see cref="Softmax{T}(Tensor{T}, int)"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to apply softmax.</param>
    /// <returns>A tensor with softmax applied along the specified axis.</returns>
    /// <remarks>
    /// <para>
    /// Computes softmax(x)_i = exp(x_i) / sum(exp(x_j)) along the specified axis.
    /// Numerically stable implementation subtracts max before exp.
    /// </para>
    /// <para>
    /// This method provides the same functionality as <see cref="Softmax{T}(Tensor{T}, int)"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSoftmax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes the backward pass for softmax. This is an alias for <see cref="SoftmaxBackward{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="softmaxOutput">The output from the forward softmax pass.</param>
    /// <param name="outputGradient">The gradient flowing back.</param>
    /// <param name="axis">The axis along which softmax was applied.</param>
    /// <returns>The gradient with respect to the softmax input.</returns>
    /// <remarks>
    /// <para>
    /// This method provides the same functionality as <see cref="SoftmaxBackward{T}"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSoftmaxBackward<T>(Tensor<T> softmaxOutput, Tensor<T> outputGradient, int axis);

    /// <summary>
    /// Computes log-softmax along a specified axis (more numerically stable than log(softmax(x))).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to apply log-softmax.</param>
    /// <returns>A tensor with log-softmax applied along the specified axis.</returns>
    Tensor<T> TensorLogSoftmax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Top-K selection along an axis, returning both values and indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="k">Number of top elements to select.</param>
    /// <param name="axis">The axis along which to select.</param>
    /// <param name="indices">Output tensor containing indices of top-k elements.</param>
    /// <returns>A tensor containing the top-k values.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.topk(). Essential for MoE gating without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTopK<T>(Tensor<T> tensor, int k, int axis, out Tensor<int> indices);

    /// <summary>
    /// Scatter operation: sets values at specified indices along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor (modified in place or returns new tensor).</param>
    /// <param name="indices">Integer indices where to scatter values.</param>
    /// <param name="source">Values to scatter.</param>
    /// <param name="axis">The axis along which to scatter.</param>
    /// <returns>The tensor with scattered values.</returns>
    Tensor<T> TensorScatter<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> source, int axis);

    /// <summary>
    /// Index select: gathers slices from a tensor along an axis using integer indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The source tensor.</param>
    /// <param name="indices">Integer indices to select.</param>
    /// <param name="axis">The axis along which to select.</param>
    /// <returns>Selected slices concatenated along the axis.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.index_select(). Essential for embedding lookups and expert selection.
    /// </para>
    /// </remarks>
    Tensor<T> TensorIndexSelect<T>(Tensor<T> tensor, Tensor<int> indices, int axis);

    /// <summary>
    /// Stacks multiple tensors along a new axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">Array of tensors to stack (must have same shape).</param>
    /// <param name="axis">The axis along which to stack.</param>
    /// <returns>A new tensor with an additional dimension.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.stack() or np.stack().
    /// Essential for building batch tensors without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorStack<T>(Tensor<T>[] tensors, int axis);

    /// <summary>
    /// Unstacks a tensor along an axis into multiple tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to unstack.</param>
    /// <param name="axis">The axis along which to unstack.</param>
    /// <returns>An array of tensors.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.unbind() or tf.unstack().
    /// Inverse of TensorStack.
    /// </para>
    /// </remarks>
    Tensor<T>[] TensorUnstack<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Applies a function element-wise to a tensor (vectorized map).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="func">The function to apply to each element.</param>
    /// <returns>A tensor with the function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This enables custom element-wise operations without explicit loops.
    /// The implementation should be parallelized internally.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMap<T>(Tensor<T> tensor, Func<T, T> func);

    /// <summary>
    /// Masked fill: fills tensor elements with a value where mask is true.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="mask">Boolean mask tensor (same shape as input or broadcastable).</param>
    /// <param name="value">The value to fill where mask is true.</param>
    /// <returns>A tensor with masked positions filled.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.masked_fill(). Essential for attention masking.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<bool> mask, T value);

    /// <summary>
    /// Where operation: selects elements from two tensors based on a condition.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="condition">Boolean condition tensor.</param>
    /// <param name="x">Tensor to select from where condition is true.</param>
    /// <param name="y">Tensor to select from where condition is false.</param>
    /// <returns>A tensor with selected elements.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.where() or np.where().
    /// Essential for conditional operations without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorWhere<T>(Tensor<bool> condition, Tensor<T> x, Tensor<T> y);

    #endregion

    #region Neural Radiance Fields Operations

    /// <summary>
    /// Computes positional encoding for Neural Radiance Fields.
    /// Applies sin/cos frequency encoding: [sin(2^0*Ãâ‚¬*x), cos(2^0*Ãâ‚¬*x), ..., sin(2^(L-1)*Ãâ‚¬*x), cos(2^(L-1)*Ãâ‚¬*x)]
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions tensor of shape [N, D] where D is typically 3 (x,y,z).</param>
    /// <param name="numFrequencies">Number of frequency levels (L in the paper, typically 10 for positions, 4 for directions).</param>
    /// <returns>Encoded tensor of shape [N, D * 2 * numFrequencies].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Positional encoding transforms low-dimensional coordinates into
    /// high-dimensional features that help neural networks learn high-frequency details.
    /// </para>
    /// <para>
    /// Without positional encoding, neural networks tend to learn smooth, blurry functions.
    /// The sin/cos encoding at multiple frequencies enables sharp, detailed reconstructions.
    /// </para>
    /// <para>
    /// Reference: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    /// by Mildenhall et al., ECCV 2020
    /// </para>
    /// </remarks>
    Tensor<T> PositionalEncoding<T>(Tensor<T> positions, int numFrequencies);

    /// <summary>
    /// Computes the backward pass for positional encoding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Original input positions tensor of shape [N, D].</param>
    /// <param name="encodedGradient">Gradient of loss with respect to encoded output [N, D * 2 * numFrequencies].</param>
    /// <param name="numFrequencies">Number of frequency levels used in forward pass.</param>
    /// <returns>Gradient with respect to input positions [N, D].</returns>
    Tensor<T> PositionalEncodingBackward<T>(Tensor<T> positions, Tensor<T> encodedGradient, int numFrequencies);

    /// <summary>
    /// Performs volume rendering along rays using alpha compositing.
    /// Computes: C(r) = ÃŽÂ£ T_i * ÃŽÂ±_i * c_i where T_i = ÃŽÂ (1 - ÃŽÂ±_j) for j &lt; i
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rgbSamples">RGB color samples of shape [numRays, numSamples, 3].</param>
    /// <param name="densitySamples">Density/sigma samples of shape [numRays, numSamples, 1].</param>
    /// <param name="tValues">Distance values along rays of shape [numRays, numSamples].</param>
    /// <returns>Rendered colors of shape [numRays, 3].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Volume rendering accumulates color along a ray by considering
    /// how much light is blocked (transmittance) at each sample point.
    /// </para>
    /// <para>
    /// The alpha value at each point represents how much light is absorbed there.
    /// Transmittance tracks how much light reaches each point from the camera.
    /// Final color is the weighted sum of all sample colors.
    /// </para>
    /// </remarks>
    Tensor<T> VolumeRendering<T>(Tensor<T> rgbSamples, Tensor<T> densitySamples, Tensor<T> tValues);

    /// <summary>
    /// Computes the backward pass for volume rendering.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rgbSamples">RGB color samples from forward pass [numRays, numSamples, 3].</param>
    /// <param name="densitySamples">Density samples from forward pass [numRays, numSamples, 1].</param>
    /// <param name="tValues">Distance values from forward pass [numRays, numSamples].</param>
    /// <param name="outputGradient">Gradient of loss with respect to rendered colors [numRays, 3].</param>
    /// <param name="rgbGradient">Output: Gradient with respect to RGB samples.</param>
    /// <param name="densityGradient">Output: Gradient with respect to density samples.</param>
    void VolumeRenderingBackward<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues,
        Tensor<T> outputGradient,
        out Tensor<T> rgbGradient,
        out Tensor<T> densityGradient);

    /// <summary>
    /// Samples points uniformly along rays for volume rendering.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rayOrigins">Ray origin points of shape [numRays, 3].</param>
    /// <param name="rayDirections">Ray direction vectors of shape [numRays, 3] (should be normalized).</param>
    /// <param name="nearBound">Near clipping distance.</param>
    /// <param name="farBound">Far clipping distance.</param>
    /// <param name="numSamples">Number of samples per ray.</param>
    /// <param name="stratified">If true, adds jitter to sample positions for anti-aliasing.</param>
    /// <returns>Tuple of (sample positions [numRays * numSamples, 3], sample directions [numRays * numSamples, 3], t values [numRays, numSamples]).</returns>
    (Tensor<T> positions, Tensor<T> directions, Tensor<T> tValues) SampleRayPoints<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        T nearBound,
        T farBound,
        int numSamples,
        bool stratified = true);

    /// <summary>
    /// Performs importance sampling based on coarse network density predictions.
    /// Samples more points where density is high (hierarchical sampling).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tValuesCoarse">Coarse sample t values [numRays, numCoarseSamples].</param>
    /// <param name="weightsCoarse">Rendering weights from coarse samples [numRays, numCoarseSamples].</param>
    /// <param name="numFineSamples">Number of additional fine samples to generate.</param>
    /// <returns>Fine sample t values [numRays, numFineSamples] concentrated where density is high.</returns>
    Tensor<T> ImportanceSampling<T>(Tensor<T> tValuesCoarse, Tensor<T> weightsCoarse, int numFineSamples);

    /// <summary>
    /// Generates camera rays for each pixel in an image.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="cameraPosition">Camera position in world coordinates [3].</param>
    /// <param name="cameraRotation">Camera rotation matrix [3, 3] (camera-to-world).</param>
    /// <param name="imageWidth">Image width in pixels.</param>
    /// <param name="imageHeight">Image height in pixels.</param>
    /// <param name="focalLength">Camera focal length.</param>
    /// <returns>Tuple of (ray origins [H*W, 3], ray directions [H*W, 3]).</returns>
    (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays<T>(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength);

    #endregion

    #region Gaussian Splatting Operations

    /// <summary>
    /// Projects 3D Gaussians to 2D screen space for rasterization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means3D">Gaussian center positions [N, 3].</param>
    /// <param name="covariances3D">3D covariance matrices [N, 3, 3] or [N, 6] (upper triangular).</param>
    /// <param name="viewMatrix">World-to-camera transformation [4, 4].</param>
    /// <param name="projMatrix">Camera projection matrix [4, 4].</param>
    /// <param name="imageWidth">Target image width.</param>
    /// <param name="imageHeight">Target image height.</param>
    /// <param name="means2D">Output: Projected 2D means [N, 2].</param>
    /// <param name="covariances2D">Output: Projected 2D covariances [N, 3] (a, b, c for axÃ‚Â² + 2bxy + cyÃ‚Â²).</param>
    /// <param name="depths">Output: Depth values for sorting [N].</param>
    /// <param name="visible">Output: Visibility mask (in frustum and valid) [N].</param>
    void ProjectGaussians3DTo2D<T>(
        Tensor<T> means3D,
        Tensor<T> covariances3D,
        Matrix<T> viewMatrix,
        Matrix<T> projMatrix,
        int imageWidth,
        int imageHeight,
        out Tensor<T> means2D,
        out Tensor<T> covariances2D,
        out Tensor<T> depths,
        out Tensor<bool> visible);

    /// <summary>
    /// Rasterizes 2D Gaussians onto an image using alpha blending.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means2D">Projected 2D Gaussian centers [N, 2].</param>
    /// <param name="covariances2D">2D covariance parameters [N, 3].</param>
    /// <param name="colors">Gaussian colors [N, C] where C is typically 3 (RGB) or more for SH.</param>
    /// <param name="opacities">Gaussian opacities [N].</param>
    /// <param name="depths">Depth values for sorting [N].</param>
    /// <param name="imageWidth">Output image width.</param>
    /// <param name="imageHeight">Output image height.</param>
    /// <param name="tileSize">Tile size for tiled rasterization (typically 16).</param>
    /// <returns>Rendered image [H, W, C].</returns>
    Tensor<T> RasterizeGaussians<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        int tileSize = 16);

    /// <summary>
    /// Computes the backward pass for Gaussian rasterization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means2D">Projected 2D Gaussian centers [N, 2].</param>
    /// <param name="covariances2D">2D covariance parameters [N, 3].</param>
    /// <param name="colors">Gaussian colors [N, C].</param>
    /// <param name="opacities">Gaussian opacities [N].</param>
    /// <param name="depths">Depth values [N].</param>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <param name="outputGradient">Gradient of loss with respect to rendered image [H, W, C].</param>
    /// <param name="tileSize">Tile size used in forward pass.</param>
    /// <param name="means2DGrad">Output: Gradient with respect to 2D means.</param>
    /// <param name="covariances2DGrad">Output: Gradient with respect to 2D covariances.</param>
    /// <param name="colorsGrad">Output: Gradient with respect to colors.</param>
    /// <param name="opacitiesGrad">Output: Gradient with respect to opacities.</param>
    void RasterizeGaussiansBackward<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        Tensor<T> outputGradient,
        int tileSize,
        out Tensor<T> means2DGrad,
        out Tensor<T> covariances2DGrad,
        out Tensor<T> colorsGrad,
        out Tensor<T> opacitiesGrad);

    /// <summary>
    /// Evaluates spherical harmonics for view-dependent color in Gaussian Splatting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shCoefficients">Spherical harmonics coefficients [N, (degree+1)Ã‚Â², C] where C=3 for RGB.</param>
    /// <param name="viewDirections">Normalized view directions [N, 3] or [1, 3] for broadcast.</param>
    /// <param name="degree">SH degree (0-3, where 0=constant, 3=full view dependence).</param>
    /// <returns>Evaluated colors [N, C].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spherical harmonics allow colors to change based on viewing angle,
    /// enabling realistic specular highlights and view-dependent effects.
    /// </para>
    /// <para>
    /// Degree 0: Constant color (1 coefficient per channel)
    /// Degree 1: Linear variation (4 coefficients per channel)
    /// Degree 2: Quadratic variation (9 coefficients per channel)
    /// Degree 3: Cubic variation (16 coefficients per channel)
    /// </para>
    /// </remarks>
    Tensor<T> EvaluateSphericalHarmonics<T>(Tensor<T> shCoefficients, Tensor<T> viewDirections, int degree);

    /// <summary>
    /// Computes the backward pass for spherical harmonics evaluation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shCoefficients">SH coefficients from forward pass.</param>
    /// <param name="viewDirections">View directions from forward pass.</param>
    /// <param name="degree">SH degree.</param>
    /// <param name="outputGradient">Gradient with respect to evaluated colors.</param>
    /// <returns>Gradient with respect to SH coefficients.</returns>
    Tensor<T> EvaluateSphericalHarmonicsBackward<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree,
        Tensor<T> outputGradient);

    /// <summary>
    /// Computes 3D covariance matrices from rotation quaternions and scale vectors.
    /// Covariance = R * S * S^T * R^T where R is rotation matrix from quaternion, S is diagonal scale.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rotations">Rotation quaternions [N, 4] (w, x, y, z).</param>
    /// <param name="scales">Scale vectors [N, 3].</param>
    /// <returns>Covariance matrices [N, 3, 3] or [N, 6] (upper triangular).</returns>
    Tensor<T> ComputeGaussianCovariance<T>(Tensor<T> rotations, Tensor<T> scales);

    /// <summary>
    /// Computes the backward pass for Gaussian covariance computation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rotations">Rotation quaternions from forward pass.</param>
    /// <param name="scales">Scale vectors from forward pass.</param>
    /// <param name="covarianceGradient">Gradient with respect to covariances.</param>
    /// <param name="rotationsGrad">Output: Gradient with respect to rotations.</param>
    /// <param name="scalesGrad">Output: Gradient with respect to scales.</param>
    void ComputeGaussianCovarianceBackward<T>(
        Tensor<T> rotations,
        Tensor<T> scales,
        Tensor<T> covarianceGradient,
        out Tensor<T> rotationsGrad,
        out Tensor<T> scalesGrad);

    #endregion

    #region Instant-NGP Operations

    /// <summary>
    /// Performs multiresolution hash encoding for Instant-NGP.
    /// Encodes 3D positions using a hierarchy of hash tables with trilinear interpolation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions [N, 3] normalized to [0, 1].</param>
    /// <param name="hashTables">List of hash tables, one per resolution level.</param>
    /// <param name="resolutions">Resolution at each level.</param>
    /// <param name="featuresPerLevel">Number of features stored per hash entry.</param>
    /// <returns>Encoded features [N, numLevels * featuresPerLevel].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hash encoding replaces expensive positional encoding with fast
    /// table lookups, enabling 100x faster training and 1000x faster rendering than NeRF.
    /// </para>
    /// <para>
    /// The key insight is that hash collisions are okay because:
    /// 1. Multiple positions mapping to the same entry often have similar features
    /// 2. The neural network learns to handle collisions during training
    /// 3. The speed benefit far outweighs the minor quality impact
    /// </para>
    /// <para>
    /// Reference: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
    /// by MÃƒÂ¼ller et al., ACM Transactions on Graphics 2022
    /// </para>
    /// </remarks>
    Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel);

    /// <summary>
    /// Computes the backward pass for multiresolution hash encoding.
    /// Accumulates gradients to the appropriate hash table entries.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions from forward pass.</param>
    /// <param name="hashTables">Hash tables from forward pass.</param>
    /// <param name="resolutions">Resolutions from forward pass.</param>
    /// <param name="featuresPerLevel">Features per level from forward pass.</param>
    /// <param name="outputGradient">Gradient with respect to encoded features.</param>
    /// <returns>Gradients for each hash table.</returns>
    Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel,
        Tensor<T> outputGradient);

    /// <summary>
    /// Updates occupancy grid for efficient ray sampling in Instant-NGP.
    /// Marks which voxels contain geometry based on density threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="occupancyGrid">Current occupancy grid [gridSize, gridSize, gridSize].</param>
    /// <param name="densities">Sampled density values.</param>
    /// <param name="positions">Positions where densities were sampled.</param>
    /// <param name="gridSize">Size of the occupancy grid.</param>
    /// <param name="threshold">Density threshold for occupancy.</param>
    /// <param name="decayFactor">EMA decay for updating occupancy values.</param>
    /// <returns>Updated occupancy grid.</returns>
    Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor);

    /// <summary>
    /// Samples rays while skipping empty space using occupancy grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rayOrigins">Ray origins [numRays, 3].</param>
    /// <param name="rayDirections">Ray directions [numRays, 3].</param>
    /// <param name="occupancyBitfield">Packed occupancy bits for fast lookup as a 1D tensor.</param>
    /// <param name="gridSize">Size of the occupancy grid.</param>
    /// <param name="sceneBoundsMin">Minimum scene bounds [3].</param>
    /// <param name="sceneBoundsMax">Maximum scene bounds [3].</param>
    /// <param name="nearBound">Near clipping distance.</param>
    /// <param name="farBound">Far clipping distance.</param>
    /// <param name="maxSamples">Maximum samples per ray.</param>
    /// <returns>Tuple of (sample positions, sample directions, valid mask, t values).</returns>
    (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues) SampleRaysWithOccupancy<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        Tensor<uint> occupancyBitfield,
        int gridSize,
        Vector<T> sceneBoundsMin,
        Vector<T> sceneBoundsMax,
        T nearBound,
        T farBound,
        int maxSamples);

    #endregion

    #region Mesh Convolution Operations

    /// <summary>
    /// Performs spiral convolution on mesh vertex features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiral convolution extends traditional convolution to irregular mesh surfaces.
    /// Unlike grid-based convolutions where neighbors are in fixed positions, mesh vertices have
    /// variable connectivity. Spiral convolution solves this by:
    /// 
    /// 1. Defining a consistent spiral ordering of neighbors around each vertex
    /// 2. Gathering features from neighbors in this canonical order
    /// 3. Applying learned weights to the ordered features
    /// 
    /// This creates translation-equivariant convolutions on arbitrary mesh topologies.
    /// </para>
    /// <para>
    /// <b>Mathematical Formulation:</b>
    /// For each vertex v with spiral neighbors S(v) = [nÃ¢â€šÂ, nÃ¢â€šâ€š, ..., nÃ¢â€šâ€“]:
    /// 
    /// gathered[v] = concat(features[nÃ¢â€šÂ], features[nÃ¢â€šâ€š], ..., features[nÃ¢â€šâ€“])
    /// output[v] = weights @ gathered[v] + bias
    /// 
    /// The spiral ordering ensures that the convolution is invariant to mesh parameterization.
    /// </para>
    /// <para>
    /// Reference: "Neural 3D Morphable Models: Spiral Convolutional Networks" by Bouritsas et al.
    /// Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.
    /// </para>
    /// </remarks>
    Tensor<T> SpiralConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        Tensor<T> biases);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to input features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <returns>Gradient with respect to input features [numVertices, inputChannels].</returns>
    /// <remarks>
    /// <para>
    /// The backward pass scatters gradients back to the original vertex positions according
    /// to the spiral indices. This uses atomic scatter-add operations for correctness when
    /// multiple spiral paths reference the same vertex.
    /// </para>
    /// </remarks>
    Tensor<T> SpiralConvBackwardInput<T>(
        Tensor<T> outputGradient,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        int inputChannels);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to weights.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <returns>Gradient with respect to weights [outputChannels, inputChannels * spiralLength].</returns>
    Tensor<T> SpiralConvBackwardWeights<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to biases.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <returns>Gradient with respect to biases [outputChannels].</returns>
    Tensor<T> SpiralConvBackwardBias<T>(Tensor<T> outputGradient);

    /// <summary>
    /// Performs diffusion convolution on mesh vertex features using the Laplacian operator.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="laplacian">Mesh Laplacian matrix [numVertices, numVertices].</param>
    /// <param name="weights">Diffusion weights [outputChannels, inputChannels].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <param name="diffusionTime">Diffusion time parameter controlling spatial extent.</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diffusion convolution uses heat diffusion on the mesh surface
    /// to define the convolution kernel. Features spread across the mesh according to the
    /// heat equation, which respects the intrinsic geometry of the surface.
    /// 
    /// The diffusion time controls how far features propagate:
    /// - Small time: local features, fine detail
    /// - Large time: global features, coarse structure
    /// </para>
    /// <para>
    /// <b>Mathematical Formulation:</b>
    /// output = exp(-t * L) @ input @ weights + bias
    /// 
    /// Where L is the mesh Laplacian and t is the diffusion time.
    /// The matrix exponential is computed using eigendecomposition or Taylor series.
    /// </para>
    /// <para>
    /// Reference: "DiffusionNet: Discretization Agnostic Learning on Surfaces" by Sharp et al.
    /// </para>
    /// </remarks>
    Tensor<T> DiffusionConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        Tensor<T> biases,
        T diffusionTime);

    /// <summary>
    /// Computes the backward pass for diffusion convolution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass.</param>
    /// <param name="laplacian">Mesh Laplacian matrix from forward pass.</param>
    /// <param name="weights">Diffusion weights from forward pass.</param>
    /// <param name="diffusionTime">Diffusion time from forward pass.</param>
    /// <returns>Tuple of (input gradient, weight gradient, bias gradient).</returns>
    (Tensor<T> inputGrad, Tensor<T> weightGrad, Tensor<T> biasGrad) DiffusionConvBackward<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        T diffusionTime);

    /// <summary>
    /// Computes the mesh Laplacian matrix from vertex positions and face indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3] for triangular mesh.</param>
    /// <param name="laplacianType">Type of Laplacian operator to compute.</param>
    /// <returns>Laplacian matrix [numVertices, numVertices].</returns>
    /// <remarks>
    /// <para>
    /// <b>Laplacian Types:</b>
    /// - <see cref="LaplacianType.Uniform"/>: Simple adjacency-based, ignores geometry
    /// - <see cref="LaplacianType.Cotangent"/>: Geometry-aware, preserves angles
    /// - <see cref="LaplacianType.Normalized"/>: Cotangent normalized by vertex areas
    /// </para>
    /// </remarks>
    Tensor<T> ComputeMeshLaplacian<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        LaplacianType laplacianType = LaplacianType.Cotangent);

    /// <summary>
    /// Generates spiral indices for mesh vertices based on connectivity.
    /// </summary>
    /// <typeparam name="T">The numeric type for vertex positions.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3].</param>
    /// <param name="spiralLength">Number of neighbors in each spiral.</param>
    /// <returns>Spiral indices [numVertices, spiralLength].</returns>
    /// <remarks>
    /// <para>
    /// The algorithm:
    /// 1. Build adjacency list from faces
    /// 2. For each vertex, find the initial reference direction
    /// 3. Sort neighbors by angle from reference in consistent winding order
    /// 4. Extend to spiral length by following the ring structure
    /// </para>
    /// </remarks>
    Tensor<int> GenerateSpiralIndices<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        int spiralLength);

    #endregion

    #region Advanced Vectorization Operations

    /// <summary>
    /// Computes pairwise squared Euclidean distances between two sets of points.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">First set of points [N, D] where N is number of points and D is dimensionality.</param>
    /// <param name="y">Second set of points [M, D] where M is number of points and D is dimensionality.</param>
    /// <returns>Distance matrix [N, M] where element [i,j] is squared distance between x[i] and y[j].</returns>
    /// <remarks>
    /// Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y for efficiency.
    /// This avoids the O(N*M*D) explicit subtraction and enables GPU parallelization.
    /// </remarks>
    Tensor<T> PairwiseDistanceSquared<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Computes pairwise Euclidean distances between two sets of points.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">First set of points [N, D].</param>
    /// <param name="y">Second set of points [M, D].</param>
    /// <returns>Distance matrix [N, M] where element [i,j] is Euclidean distance between x[i] and y[j].</returns>
    Tensor<T> PairwiseDistance<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Returns the k largest or smallest elements along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="k">Number of top elements to return.</param>
    /// <param name="axis">Axis along which to find top-k. Default -1 means last axis.</param>
    /// <param name="largest">If true, return k largest elements; if false, return k smallest.</param>
    /// <returns>Tuple of (values, indices) where values contains the top-k elements and indices their positions.</returns>
    (Tensor<T> values, Tensor<int> indices) TopK<T>(Tensor<T> input, int k, int axis = -1, bool largest = true);

    /// <summary>
    /// Returns indices that would sort the tensor along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="axis">Axis along which to sort. Default -1 means last axis.</param>
    /// <param name="descending">If true, sort in descending order.</param>
    /// <returns>Tensor of indices that would sort the input.</returns>
    Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>
    /// Gathers elements from input tensor along an axis using indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Source tensor to gather from.</param>
    /// <param name="indices">Indices specifying which elements to gather.</param>
    /// <param name="axis">Axis along which to gather.</param>
    /// <returns>Tensor of gathered elements.</returns>
    Tensor<T> Gather<T>(Tensor<T> input, Tensor<int> indices, int axis);

    /// <summary>
    /// Scatters values into a new tensor at positions specified by indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor providing the shape and initial values.</param>
    /// <param name="indices">Indices where to scatter values.</param>
    /// <param name="values">Values to scatter.</param>
    /// <param name="axis">Axis along which to scatter.</param>
    /// <returns>New tensor with scattered values.</returns>
    Tensor<T> Scatter<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis);

    /// <summary>
    /// Scatters values by adding them to existing values at positions specified by indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor providing the shape and initial values.</param>
    /// <param name="indices">Indices where to add values.</param>
    /// <param name="values">Values to add at specified indices.</param>
    /// <param name="axis">Axis along which to scatter-add.</param>
    /// <returns>New tensor with values added at specified positions.</returns>
    Tensor<T> ScatterAdd<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis);

    /// <summary>
    /// Computes the hyperbolic cosine of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with hyperbolic cosine applied element-wise.</returns>
    Tensor<T> TensorCosh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the hyperbolic sine of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with hyperbolic sine applied element-wise.</returns>
    Tensor<T> TensorSinh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the outer product of two 1D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">First 1D tensor of length N.</param>
    /// <param name="b">Second 1D tensor of length M.</param>
    /// <returns>2D tensor [N, M] where result[i,j] = a[i] * b[j].</returns>
    Tensor<T> TensorOuter<T>(Tensor<T> a, Tensor<T> b);

    #endregion
}
