namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Rational Quadratic kernel, equivalent to an infinite mixture of RBF kernels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Rational Quadratic (RQ) kernel is a powerful kernel that can be
/// thought of as an infinite mixture of RBF kernels with different length scales.
///
/// In mathematical terms:
/// k(x, x') = σ² × (1 + r²/(2αl²))^(-α)
///
/// Where:
/// - r = |x - x'| is the Euclidean distance
/// - l is the length scale
/// - α is the "scale mixture" parameter
/// - σ² is the variance
///
/// The α parameter controls how the kernel behaves:
/// - α → ∞: Approaches the RBF kernel (single length scale)
/// - Small α: More heavy-tailed (multiple length scales contribute)
/// </para>
/// <para>
/// Why use Rational Quadratic?
///
/// 1. **Multi-scale**: Naturally captures patterns at different scales
/// 2. **Robustness**: Less sensitive to length scale choice than RBF
/// 3. **Heavy tails**: Points far apart still have some correlation
/// 4. **Flexibility**: Interpolates between RBF and more flexible behaviors
///
/// When to use:
/// - Data has patterns at multiple scales
/// - You're not sure what length scale to use
/// - RBF seems too restrictive
/// </para>
/// </remarks>
public class RationalQuadraticKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The length scale parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Similar to RBF, this controls the "typical" length scale
    /// of variation. But unlike RBF, the RQ kernel effectively uses multiple scales
    /// centered around this value.
    /// </para>
    /// </remarks>
    private readonly double _lengthScale;

    /// <summary>
    /// The scale mixture parameter (α).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter controls the "weighting" of different length scales.
    ///
    /// - Large α (e.g., 100): Behaves like RBF (single length scale dominates)
    /// - Small α (e.g., 0.1): Many different length scales contribute
    /// - α = 1: A common default, balances single-scale and multi-scale behavior
    ///
    /// If you're unsure, start with α = 1 and let hyperparameter optimization find the best value.
    /// </para>
    /// </remarks>
    private readonly double _alpha;

    /// <summary>
    /// The signal variance.
    /// </summary>
    private readonly double _variance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Rational Quadratic kernel.
    /// </summary>
    /// <param name="lengthScale">The length scale parameter. Default is 1.0.</param>
    /// <param name="alpha">The scale mixture parameter. Default is 1.0.</param>
    /// <param name="variance">The signal variance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Rational Quadratic kernel with specified parameters.
    ///
    /// Starting points:
    /// - lengthScale: Based on the scale of your input data
    /// - alpha: Start with 1.0, decrease if you need more flexibility
    /// - variance: Based on the variance of your output data
    ///
    /// Example:
    /// var kernel = new RationalQuadraticKernel&lt;double&gt;(lengthScale: 1.0, alpha: 2.0);
    /// </para>
    /// </remarks>
    public RationalQuadraticKernel(double lengthScale = 1.0, double alpha = 1.0, double variance = 1.0)
    {
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive.", nameof(alpha));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _lengthScale = lengthScale;
        _alpha = alpha;
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the length scale.
    /// </summary>
    public double LengthScale => _lengthScale;

    /// <summary>
    /// Gets the scale mixture parameter (α).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the signal variance.
    /// </summary>
    public double Variance => _variance;

    /// <summary>
    /// Calculates the Rational Quadratic kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the covariance between two points.
    ///
    /// The calculation:
    /// 1. Compute squared distance: r² = Σᵢ(x1ᵢ - x2ᵢ)²
    /// 2. Compute: k = σ² × (1 + r²/(2αl²))^(-α)
    ///
    /// Compared to RBF [exp(-r²/(2l²))]:
    /// - For small r: Both give similar values
    /// - For large r: RQ decays slower (polynomial vs exponential)
    ///
    /// This slower decay means RQ allows more long-range correlations.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Compute squared Euclidean distance
        double r2 = 0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            r2 += diff * diff;
        }

        // Compute RQ kernel: sigma^2 * (1 + r^2/(2*alpha*l^2))^(-alpha)
        double scaledR2 = r2 / (2.0 * _alpha * _lengthScale * _lengthScale);
        double result = _variance * Math.Pow(1.0 + scaledR2, -_alpha);

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the gradient of the kernel with respect to the input.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The gradient with respect to x1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient tells us how the kernel value changes as we move x1.
    /// This is useful for:
    /// - Optimizing acquisition functions in Bayesian optimization
    /// - Computing gradients for GP-based models
    /// </para>
    /// </remarks>
    public Vector<T> CalculateGradient(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        int d = x1.Length;
        var gradient = new Vector<T>(d);

        // Compute squared distance
        double r2 = 0;
        for (int i = 0; i < d; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            r2 += diff * diff;
        }

        // Precompute common factors
        double l2 = _lengthScale * _lengthScale;
        double scaledR2 = r2 / (2.0 * _alpha * l2);
        double baseTerm = 1.0 + scaledR2;
        double factor = -_variance * Math.Pow(baseTerm, -_alpha - 1) / l2;

        for (int i = 0; i < d; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            gradient[i] = _numOps.FromDouble(factor * diff);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the gradient with respect to hyperparameters.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>Dictionary mapping hyperparameter names to gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how the kernel value changes when we adjust
    /// the hyperparameters (lengthScale, alpha, variance).
    ///
    /// Used for:
    /// - Gradient-based hyperparameter optimization
    /// - Maximizing the log marginal likelihood
    /// </para>
    /// </remarks>
    public Dictionary<string, double> CalculateHyperparameterGradients(Vector<T> x1, Vector<T> x2)
    {
        // Compute squared distance
        double r2 = 0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            r2 += diff * diff;
        }

        double l2 = _lengthScale * _lengthScale;
        double scaledR2 = r2 / (2.0 * _alpha * l2);
        double baseTerm = 1.0 + scaledR2;
        double baseTermPowAlpha = Math.Pow(baseTerm, -_alpha);
        double k = _variance * baseTermPowAlpha;

        var gradients = new Dictionary<string, double>();

        // dk/d(variance) = (1 + r^2/(2*alpha*l^2))^(-alpha)
        gradients["variance"] = baseTermPowAlpha;

        // dk/d(lengthScale) = sigma^2 * (-alpha) * (1 + ...)^(-alpha-1) * r^2 / (alpha * l^3)
        //                   = k * alpha * r^2 / (l^3 * (2*alpha*l^2 + r^2))
        double denom = 2.0 * _alpha * l2 + r2;
        if (Math.Abs(denom) > 1e-10)
        {
            gradients["lengthScale"] = k * _alpha * r2 / (_lengthScale * denom);
        }
        else
        {
            gradients["lengthScale"] = 0;
        }

        // dk/d(alpha) is more complex
        // dk/d(alpha) = sigma^2 * [ -log(baseTerm) * baseTerm^(-alpha) + baseTerm^(-alpha-1) * r^2/(2*alpha^2*l^2) ]
        double logBase = Math.Log(Math.Max(baseTerm, 1e-10));
        gradients["alpha"] = k * (scaledR2 / (_alpha * baseTerm) - logBase);

        return gradients;
    }

    /// <summary>
    /// Determines if this kernel approximates an RBF kernel.
    /// </summary>
    /// <returns>True if alpha is large enough that the kernel is effectively RBF.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When α is very large, the RQ kernel becomes indistinguishable
    /// from an RBF kernel. This method checks if that's the case.
    /// </para>
    /// </remarks>
    public bool IsEffectivelyRBF()
    {
        return _alpha > 100;
    }

    /// <summary>
    /// Returns the "effective number of length scales" in the mixture.
    /// </summary>
    /// <returns>A measure of length scale diversity.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives a rough idea of how many different length scales
    /// the kernel is effectively using.
    ///
    /// - Near 1: Behaves like single length scale (RBF-like)
    /// - Large values: Multiple length scales are important
    ///
    /// This is an approximation based on the alpha parameter.
    /// </para>
    /// </remarks>
    public double EffectiveLengthScales()
    {
        // Rough approximation: smaller alpha = more effective length scales
        return 1.0 + 1.0 / _alpha;
    }
}
