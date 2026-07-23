namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Wave kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Wave kernel is a specialized kernel function that produces wave-like patterns in the similarity
/// space. It is based on the sinc function (sin(x)/x) and creates oscillating similarity values as the
/// distance between points increases.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Wave kernel is special because it creates a pattern of similarity that rises and falls like a wave
/// as points get farther apart. This is different from most kernels where similarity only decreases with distance.
/// </para>
/// <para>
/// Think of it like this: If you drop two stones in a pond, the Wave kernel is like measuring how the
/// ripples from each stone interact with each other. Sometimes the ripples add up (high similarity) and
/// sometimes they cancel out (low similarity), creating a wave-like pattern.
/// </para>
/// <para>
/// The formula for the Wave kernel is:
/// k(x, y) = sin(||x-y||/s) / (||x-y||/s)
/// where:
/// - ||x-y|| is the Euclidean distance between vectors x and y
/// - s (sigma) is a parameter that controls the width of the waves
/// </para>
/// <para>
/// Common uses include:
/// - Signal processing applications
/// - Time series analysis
/// - Problems where periodic patterns are important
/// - Specialized machine learning tasks where oscillating similarity is beneficial
/// </para>
/// </remarks>
public class WaveKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The sigma parameter that controls the width of the waves in the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter controls how quickly the waves in the similarity pattern
    /// oscillate. A smaller sigma means more rapid oscillations, while a larger sigma means
    /// more gradual changes in similarity.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that allows the kernel to perform mathematical
    /// operations regardless of what numeric type (like double, float, decimal) you're using.
    /// You don't need to interact with this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Wave kernel with the specified sigma parameter.
    /// </summary>
    /// <param name="sigma">
    /// The parameter that controls the width of the waves. If not specified, defaults to 1.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Wave kernel for use. You can optionally
    /// provide a value for sigma, which controls how quickly the similarity oscillates as
    /// points get farther apart.
    /// </para>
    /// <para>
    /// If you don't specify a sigma value, it will default to 1.0, which works well for many cases.
    /// </para>
    /// <para>
    /// When might you want to adjust sigma?
    /// - Use a smaller sigma (e.g., 0.1) when you want more rapid oscillations in similarity
    /// - Use a larger sigma (e.g., 10.0) when you want more gradual changes in similarity
    /// </para>
    /// <para>
    /// The Wave kernel is particularly useful for:
    /// - Data with periodic patterns
    /// - Signal processing tasks
    /// - Time series analysis
    /// </para>
    /// </remarks>
    public WaveKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Wave kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Wave kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the difference between the two vectors
    /// 2. Calculating the Euclidean distance (straight-line distance) between them
    /// 3. Normalizing this distance by dividing by sigma
    /// 4. If the normalized distance is zero (meaning the points are identical), returning 1
    /// 5. Otherwise, calculating sin(normalized distance) / normalized distance
    /// </para>
    /// <para>
    /// The result is a similarity measure that:
    /// - Equals 1 when the vectors are identical
    /// - Oscillates between positive and negative values as distance increases
    /// - Gradually decreases in amplitude as distance increases
    /// </para>
    /// <para>
    /// What is Euclidean distance? It's the straight-line distance between two points, calculated
    /// using the Pythagorean theorem. For example, the Euclidean distance between points (1,2) and
    /// (4,6) is v((4-1)² + (6-2)²) = v(9 + 16) = v25 = 5.
    /// </para>
    /// <para>
    /// What makes this kernel special is its oscillating behavior, which can be useful for capturing
    /// periodic patterns in your data.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));
        var normalizedDistance = _numOps.Divide(distance, _sigma);

        if (_numOps.Equals(normalizedDistance, _numOps.Zero))
        {
            return _numOps.One;
        }

        return _numOps.Divide(MathHelper.Sin(normalizedDistance), normalizedDistance);
    }
}
