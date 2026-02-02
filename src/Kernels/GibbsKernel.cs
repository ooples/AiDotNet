namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Gibbs kernel with input-dependent length scales for non-stationary covariance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Gibbs kernel is a non-stationary kernel where the length scale
/// can vary depending on where you are in input space. This allows modeling functions that
/// are smooth in some regions and vary rapidly in others.
///
/// In mathematical terms:
/// k(x, x') = √(2×l(x)×l(x') / (l(x)² + l(x')²)) × exp(-r² / (l(x)² + l(x')²))
///
/// Where:
/// - l(x) is the length scale function evaluated at x
/// - r = |x - x'| is the Euclidean distance
///
/// Unlike stationary kernels (like RBF) where the same length scale applies everywhere,
/// the Gibbs kernel allows:
/// - Tight length scales where you want to capture fine details
/// - Long length scales where the function is smooth
/// </para>
/// <para>
/// Applications:
/// - Functions that transition between smooth and rough regions
/// - Spatially varying correlation structures
/// - Time series where dynamics change over time
/// - Heteroscedastic (non-constant variance) modeling
///
/// Example: Stock prices might be smooth during normal trading but vary rapidly during
/// market opens, news events, or high volatility periods.
/// </para>
/// </remarks>
public class GibbsKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The function that maps input locations to length scales.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function determines the length scale at any point.
    /// You can design it to reflect your prior beliefs about where the function
    /// should be smooth vs. rapidly varying.
    ///
    /// Example length scale functions:
    /// - Constant: l(x) = c (reduces to standard RBF)
    /// - Linear: l(x) = a + b×|x| (length scale grows with x)
    /// - Periodic: l(x) = c + d×sin(x) (periodic variation)
    /// - Step: l(x) = c1 if x &lt; threshold else c2
    /// </para>
    /// </remarks>
    private readonly Func<Vector<T>, double> _lengthScaleFunction;

    /// <summary>
    /// The signal variance (overall scale of the kernel).
    /// </summary>
    private readonly double _variance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Gibbs kernel with a custom length scale function.
    /// </summary>
    /// <param name="lengthScaleFunction">Function mapping input points to length scales.</param>
    /// <param name="variance">The signal variance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gibbs kernel with a custom length scale function.
    ///
    /// The length scale function should:
    /// - Always return positive values
    /// - Be defined for all possible input points
    /// - Reflect your beliefs about where the function varies rapidly vs. slowly
    ///
    /// Example:
    /// // Length scale increases with x (smooth for large x)
    /// Func&lt;Vector&lt;double&gt;, double&gt; lFunc = x =&gt; 0.1 + 0.5 * Math.Abs(x[0]);
    /// var kernel = new GibbsKernel&lt;double&gt;(lFunc);
    /// </para>
    /// </remarks>
    public GibbsKernel(Func<Vector<T>, double> lengthScaleFunction, double variance = 1.0)
    {
        if (lengthScaleFunction is null) throw new ArgumentNullException(nameof(lengthScaleFunction));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _lengthScaleFunction = lengthScaleFunction;
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the signal variance.
    /// </summary>
    public double Variance => _variance;

    /// <summary>
    /// Calculates the Gibbs kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value with input-dependent length scales.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the covariance between two points where the
    /// "distance" is scaled differently depending on where each point is.
    ///
    /// The calculation:
    /// 1. Get length scale at x1: l1 = l(x1)
    /// 2. Get length scale at x2: l2 = l(x2)
    /// 3. Compute effective length scale: l_eff² = l1² + l2²
    /// 4. Compute squared distance: r² = |x1 - x2|²
    /// 5. Apply formula: σ² × √(2×l1×l2/l_eff²) × exp(-r²/l_eff²)
    ///
    /// When l1 = l2 = l (constant), this simplifies to the standard RBF kernel.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Get length scales at each point
        double l1 = _lengthScaleFunction(x1);
        double l2 = _lengthScaleFunction(x2);

        if (l1 <= 0 || l2 <= 0)
            throw new InvalidOperationException("Length scale function must return positive values.");

        double l1Sq = l1 * l1;
        double l2Sq = l2 * l2;
        double lEffSq = l1Sq + l2Sq;

        // Compute squared Euclidean distance
        double r2 = 0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            r2 += diff * diff;
        }

        // Normalization factor: sqrt(2 * l1 * l2 / (l1^2 + l2^2))
        double normFactor = Math.Sqrt(2.0 * l1 * l2 / lEffSq);

        // Exponential factor: exp(-r^2 / (l1^2 + l2^2))
        double expFactor = Math.Exp(-r2 / lEffSq);

        double result = _variance * normFactor * expFactor;
        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Gets the length scale at a given point.
    /// </summary>
    /// <param name="x">The input point.</param>
    /// <returns>The length scale at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the length scale function evaluated at x.
    /// Useful for visualizing how the length scale varies across input space.
    /// </para>
    /// </remarks>
    public double GetLengthScale(Vector<T> x)
    {
        return _lengthScaleFunction(x);
    }

    /// <summary>
    /// Creates a Gibbs kernel with a linear length scale function.
    /// </summary>
    /// <param name="baseScale">The base length scale at origin.</param>
    /// <param name="slope">How much length scale increases per unit distance from origin.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A new Gibbs kernel with linear length scale.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gibbs kernel where length scale grows linearly
    /// with distance from the origin.
    ///
    /// l(x) = baseScale + slope × |x|
    ///
    /// This is useful when you expect:
    /// - Fine details near the origin
    /// - Smoother behavior far from the origin
    ///
    /// Example: Modeling a function that is complex near zero but simple for large x.
    /// </para>
    /// </remarks>
    public static GibbsKernel<T> WithLinearLengthScale(
        double baseScale = 0.1,
        double slope = 0.1,
        double variance = 1.0)
    {
        if (baseScale <= 0)
            throw new ArgumentException("Base scale must be positive.", nameof(baseScale));
        if (slope < 0)
            throw new ArgumentException("Slope must be non-negative.", nameof(slope));

        var numOps = MathHelper.GetNumericOperations<T>();

        Func<Vector<T>, double> lFunc = x =>
        {
            double norm = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double val = numOps.ToDouble(x[i]);
                norm += val * val;
            }
            return baseScale + slope * Math.Sqrt(norm);
        };

        return new GibbsKernel<T>(lFunc, variance);
    }

    /// <summary>
    /// Creates a Gibbs kernel with a sinusoidal length scale function.
    /// </summary>
    /// <param name="baseScale">The average length scale.</param>
    /// <param name="amplitude">The amplitude of sinusoidal variation.</param>
    /// <param name="frequency">The frequency of sinusoidal variation.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A new Gibbs kernel with sinusoidal length scale.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gibbs kernel where length scale varies sinusoidally.
    ///
    /// l(x) = baseScale + amplitude × sin(2π × frequency × x[0])
    ///
    /// This is useful when you expect:
    /// - Periodic changes in function smoothness
    /// - Regions of high detail alternating with smooth regions
    ///
    /// Example: Modeling a signal that's smooth during certain phases and noisy during others.
    /// </para>
    /// </remarks>
    public static GibbsKernel<T> WithSinusoidalLengthScale(
        double baseScale = 0.5,
        double amplitude = 0.3,
        double frequency = 1.0,
        double variance = 1.0)
    {
        if (baseScale <= amplitude)
            throw new ArgumentException("Base scale must be greater than amplitude to ensure positive length scales.");

        var numOps = MathHelper.GetNumericOperations<T>();

        Func<Vector<T>, double> lFunc = x =>
        {
            double xVal = numOps.ToDouble(x[0]);
            return baseScale + amplitude * Math.Sin(2.0 * Math.PI * frequency * xVal);
        };

        return new GibbsKernel<T>(lFunc, variance);
    }

    /// <summary>
    /// Creates a Gibbs kernel with a step function length scale.
    /// </summary>
    /// <param name="scale1">Length scale for region 1.</param>
    /// <param name="scale2">Length scale for region 2.</param>
    /// <param name="threshold">The x value where the transition occurs.</param>
    /// <param name="transitionWidth">Width of smooth transition (0 for hard step).</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A new Gibbs kernel with step function length scale.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gibbs kernel where length scale changes abruptly
    /// (or smoothly) between two regions.
    ///
    /// Uses first dimension only:
    /// l(x) ≈ scale1  if x[0] &lt; threshold
    /// l(x) ≈ scale2  if x[0] > threshold
    ///
    /// With smooth transition if transitionWidth > 0.
    ///
    /// This is useful when you expect:
    /// - Different dynamics in different regions
    /// - A known changepoint in the data
    ///
    /// Example: Before/after an intervention, regime change, or structural break.
    /// </para>
    /// </remarks>
    public static GibbsKernel<T> WithStepLengthScale(
        double scale1 = 0.1,
        double scale2 = 1.0,
        double threshold = 0.0,
        double transitionWidth = 0.1,
        double variance = 1.0)
    {
        if (scale1 <= 0 || scale2 <= 0)
            throw new ArgumentException("Scales must be positive.");
        if (transitionWidth < 0)
            throw new ArgumentException("Transition width must be non-negative.");

        var numOps = MathHelper.GetNumericOperations<T>();

        Func<Vector<T>, double> lFunc = x =>
        {
            double xVal = numOps.ToDouble(x[0]);

            if (transitionWidth <= 0)
            {
                // Hard step
                return xVal < threshold ? scale1 : scale2;
            }
            else
            {
                // Smooth transition using sigmoid
                double z = (xVal - threshold) / transitionWidth;
                double sigmoid = 1.0 / (1.0 + Math.Exp(-z));
                return scale1 + (scale2 - scale1) * sigmoid;
            }
        };

        return new GibbsKernel<T>(lFunc, variance);
    }

    /// <summary>
    /// Creates a Gibbs kernel with length scale learned from data.
    /// </summary>
    /// <param name="lengthScaleValues">Array of length scale values at grid points.</param>
    /// <param name="gridMin">Minimum value of the interpolation grid.</param>
    /// <param name="gridMax">Maximum value of the interpolation grid.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A new Gibbs kernel with interpolated length scale.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gibbs kernel where length scale is specified
    /// at grid points and interpolated between them.
    ///
    /// This allows flexible non-parametric specification of how length scale varies.
    /// The length scale values can be learned by optimizing the GP marginal likelihood.
    ///
    /// Steps:
    /// 1. Define a grid over your input range
    /// 2. Specify or learn length scale at each grid point
    /// 3. Kernel interpolates between grid points
    /// </para>
    /// </remarks>
    public static GibbsKernel<T> WithInterpolatedLengthScale(
        double[] lengthScaleValues,
        double gridMin,
        double gridMax,
        double variance = 1.0)
    {
        if (lengthScaleValues is null) throw new ArgumentNullException(nameof(lengthScaleValues));
        if (lengthScaleValues.Length < 2)
            throw new ArgumentException("Need at least 2 length scale values.", nameof(lengthScaleValues));
        if (gridMax <= gridMin)
            throw new ArgumentException("gridMax must be greater than gridMin.");

        foreach (double ls in lengthScaleValues)
        {
            if (ls <= 0)
                throw new ArgumentException("All length scale values must be positive.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        double[] values = (double[])lengthScaleValues.Clone();
        int numPoints = values.Length;
        double gridStep = (gridMax - gridMin) / (numPoints - 1);

        Func<Vector<T>, double> lFunc = x =>
        {
            double xVal = numOps.ToDouble(x[0]);

            // Clamp to grid range
            xVal = Math.Max(gridMin, Math.Min(gridMax, xVal));

            // Find interpolation position
            double pos = (xVal - gridMin) / gridStep;
            int i0 = (int)Math.Floor(pos);
            int i1 = Math.Min(i0 + 1, numPoints - 1);
            i0 = Math.Max(0, i0);

            // Linear interpolation
            double t = pos - i0;
            return values[i0] * (1 - t) + values[i1] * t;
        };

        return new GibbsKernel<T>(lFunc, variance);
    }
}
