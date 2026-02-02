namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Automatic Relevance Determination (ARD) kernel with per-dimension length scales.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The ARD (Automatic Relevance Determination) kernel is a powerful extension
/// of the standard RBF kernel that uses a different length scale for each input dimension.
///
/// In mathematical terms: k(x, x') = σ² × exp(-0.5 × Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)
///
/// Where:
/// - σ² is the overall variance (signal variance)
/// - lᵢ is the length scale for dimension i
/// </para>
/// <para>
/// Why is ARD important?
///
/// 1. **Feature Selection**: The length scales tell you which features are important:
///    - Small lᵢ → Feature i is very relevant (small changes matter a lot)
///    - Large lᵢ → Feature i is less relevant (can be ignored)
///    - Very large lᵢ → Feature i is essentially irrelevant
///
/// 2. **Dimensionality Handling**: In high-dimensional problems, not all features matter equally.
///    ARD automatically figures out which ones are important.
///
/// 3. **Interpretability**: After training, inspect the length scales to understand
///    which features drive your predictions.
///
/// 4. **Regularization**: By learning length scales, the model avoids overfitting
///    to irrelevant features.
///
/// How it works:
/// - Start with some initial length scales
/// - Optimize them by maximizing the log marginal likelihood
/// - Features with large optimized length scales can be pruned
/// </para>
/// </remarks>
public class ARDKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The length scales for each input dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each length scale controls how important that dimension is.
    ///
    /// - Small length scale (e.g., 0.1): The function changes rapidly with this feature
    /// - Large length scale (e.g., 10.0): The function changes slowly with this feature
    ///
    /// After optimization, you can interpret these values:
    /// - Features with small length scales are important predictors
    /// - Features with large length scales contribute little to predictions
    /// </para>
    /// </remarks>
    private readonly double[] _lengthScales;

    /// <summary>
    /// The signal variance (overall scale of the kernel).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The signal variance (σ²) scales the entire kernel output.
    /// It controls the overall magnitude of the function values:
    ///
    /// - Large σ²: Function values can vary widely
    /// - Small σ²: Function values stay close to the mean
    ///
    /// This is often learned during hyperparameter optimization.
    /// </para>
    /// </remarks>
    private readonly double _variance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ARD kernel with specified length scales.
    /// </summary>
    /// <param name="lengthScales">Array of length scales, one per input dimension.</param>
    /// <param name="variance">The signal variance (overall scale). Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an ARD kernel with explicit per-dimension length scales.
    ///
    /// Example for 3D input:
    /// var kernel = new ARDKernel&lt;double&gt;(new[] { 1.0, 2.0, 0.5 }, variance: 1.0);
    ///
    /// This means:
    /// - Dimension 0: Medium relevance (length scale 1.0)
    /// - Dimension 1: Lower relevance (length scale 2.0, changes less important)
    /// - Dimension 2: High relevance (length scale 0.5, changes very important)
    ///
    /// If you don't know the right length scales, start with equal values (e.g., all 1.0)
    /// and optimize them using gradient descent on the log marginal likelihood.
    /// </para>
    /// </remarks>
    public ARDKernel(double[] lengthScales, double variance = 1.0)
    {
        if (lengthScales is null || lengthScales.Length == 0)
            throw new ArgumentException("Length scales array must not be empty.", nameof(lengthScales));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        foreach (var ls in lengthScales)
        {
            if (ls <= 0)
                throw new ArgumentException("All length scales must be positive.", nameof(lengthScales));
        }

        _lengthScales = (double[])lengthScales.Clone();
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Initializes a new ARD kernel with uniform length scales.
    /// </summary>
    /// <param name="numDimensions">The number of input dimensions.</param>
    /// <param name="lengthScale">The uniform length scale for all dimensions. Default is 1.0.</param>
    /// <param name="variance">The signal variance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an ARD kernel where all dimensions start with the same length scale.
    ///
    /// This is a good starting point when you don't know which features are more important.
    /// The length scales can then be optimized during training.
    ///
    /// Example: new ARDKernel&lt;double&gt;(5, 1.0, 1.0) creates a kernel for 5-dimensional input
    /// with all length scales initially set to 1.0.
    /// </para>
    /// </remarks>
    public ARDKernel(int numDimensions, double lengthScale = 1.0, double variance = 1.0)
    {
        if (numDimensions < 1)
            throw new ArgumentException("Number of dimensions must be at least 1.", nameof(numDimensions));
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _lengthScales = new double[numDimensions];
        for (int i = 0; i < numDimensions; i++)
        {
            _lengthScales[i] = lengthScale;
        }
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the ARD kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value with per-dimension scaling.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths or don't match the configured dimensions.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes similarity with different scales for each dimension.
    ///
    /// The calculation:
    /// 1. For each dimension i: compute (x1[i] - x2[i])² / lengthScale[i]²
    /// 2. Sum all these scaled squared differences
    /// 3. Apply: variance × exp(-0.5 × sum)
    ///
    /// Dimensions with small length scales contribute more to the "distance" and thus
    /// have more influence on whether two points are considered similar.
    ///
    /// Example:
    /// - If dimension 0 has lengthScale = 0.5 and dimension 1 has lengthScale = 2.0
    /// - A difference of 1.0 in dimension 0 contributes (1.0)²/(0.5)² = 4.0
    /// - A difference of 1.0 in dimension 1 contributes (1.0)²/(2.0)² = 0.25
    /// - Dimension 0 matters 16× more!
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");
        if (x1.Length != _lengthScales.Length)
            throw new ArgumentException($"Vector length ({x1.Length}) must match the number of configured dimensions ({_lengthScales.Length}).");

        double scaledSqDist = 0.0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            double scaledDiff = diff / _lengthScales[i];
            scaledSqDist += scaledDiff * scaledDiff;
        }

        double result = _variance * Math.Exp(-0.5 * scaledSqDist);
        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Gets the current length scales.
    /// </summary>
    /// <returns>A copy of the length scales array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to inspect which features are considered important.
    ///
    /// After hyperparameter optimization:
    /// - Small length scales → Important features
    /// - Large length scales → Less important features
    ///
    /// You can use this for feature selection: remove features with very large length scales.
    /// </para>
    /// </remarks>
    public double[] GetLengthScales() => (double[])_lengthScales.Clone();

    /// <summary>
    /// Gets the signal variance.
    /// </summary>
    /// <returns>The variance parameter.</returns>
    public double GetVariance() => _variance;
}
