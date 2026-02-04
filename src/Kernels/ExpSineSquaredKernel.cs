namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Exp-Sine-Squared (Periodic) kernel for modeling repeating patterns.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Exp-Sine-Squared kernel is designed for data that has repeating patterns,
/// like daily temperature cycles, seasonal sales, or heartbeat signals.
///
/// In mathematical terms: k(x, x') = exp(-2 * sin²(π * |x - x'| / period) / lengthScale²)
///
/// Where:
/// - period: How often the pattern repeats (e.g., 24 hours for daily cycles)
/// - lengthScale: How smooth the pattern is within each period
/// </para>
/// <para>
/// Think of it like a wave function that repeats forever. If you have:
/// - Daily temperature data, period = 24 (hours)
/// - Yearly sales data, period = 12 (months) or 365 (days)
/// - Weekly patterns, period = 7 (days)
///
/// The kernel considers points at the same phase of the cycle as similar, regardless
/// of how many cycles apart they are. So Monday's data is similar to other Mondays,
/// even if they're weeks apart.
/// </para>
/// <para>
/// This kernel is commonly used in:
/// - Time series forecasting with seasonal patterns
/// - Signal processing with periodic components
/// - Any domain where you expect repeating behavior
/// </para>
/// </remarks>
public class ExpSineSquaredKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The period of the repeating pattern.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The period tells the kernel how often your pattern repeats.
    ///
    /// For example:
    /// - For hourly data with daily cycles: period = 24
    /// - For daily data with weekly cycles: period = 7
    /// - For monthly data with yearly cycles: period = 12
    ///
    /// The kernel will treat points that are exactly one period apart as having
    /// the same phase (and thus maximum similarity for that part of the pattern).
    ///
    /// If your period is wrong, the kernel won't capture the true periodicity in your data.
    /// You can often estimate the period from domain knowledge or by looking at your data.
    /// </para>
    /// </remarks>
    private readonly double _period;

    /// <summary>
    /// The length scale parameter that controls smoothness within each period.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The length scale controls how "smooth" the periodic pattern is.
    ///
    /// - A small length scale means the function can vary rapidly within each period
    /// - A large length scale means the function changes gradually within each period
    ///
    /// Think of it like the resolution of your periodic pattern:
    /// - Small length scale: Can capture sharp peaks and valleys
    /// - Large length scale: Smooths out fine details, captures broad trends
    ///
    /// If you set the length scale too small, your model might overfit to noise.
    /// If you set it too large, you might miss important features in your data.
    /// </para>
    /// </remarks>
    private readonly double _lengthScale;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Exp-Sine-Squared (Periodic) kernel.
    /// </summary>
    /// <param name="period">The period of the repeating pattern. Default is 1.0.</param>
    /// <param name="lengthScale">The length scale controlling smoothness. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Periodic kernel with your chosen parameters.
    ///
    /// The period should match the natural cycle in your data:
    /// - For daily patterns in hourly data: period = 24
    /// - For weekly patterns in daily data: period = 7
    /// - For yearly patterns in monthly data: period = 12
    ///
    /// The length scale is harder to choose - start with 1.0 and adjust based on
    /// whether your model is overfitting (try larger) or underfitting (try smaller).
    ///
    /// Both parameters can be optimized automatically using gradient descent on the
    /// log marginal likelihood if you're using Gaussian Processes.
    /// </para>
    /// </remarks>
    public ExpSineSquaredKernel(double period = 1.0, double lengthScale = 1.0)
    {
        if (period <= 0)
            throw new ArgumentException("Period must be positive.", nameof(period));
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));

        _period = period;
        _lengthScale = lengthScale;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Exp-Sine-Squared kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the periodic similarity between the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how similar two points are based on where
    /// they fall in the periodic pattern.
    ///
    /// The calculation:
    /// 1. Computes the distance between the two points
    /// 2. Converts this distance to a "phase" in the periodic cycle using sine
    /// 3. Applies the exponential function to get a similarity measure
    ///
    /// Points at the same phase of the cycle (e.g., both are Mondays) will have high similarity.
    /// Points at opposite phases (e.g., Monday vs. Thursday) will have lower similarity.
    ///
    /// The result is always between 0 and 1:
    /// - 1 = exactly the same phase (e.g., same point, or exactly one period apart)
    /// - 0 = opposite phases (though in practice, it rarely gets exactly to 0)
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Calculate the Euclidean distance between points
        T squaredDistance = _numOps.Zero;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            squaredDistance = _numOps.Add(squaredDistance, _numOps.Multiply(diff, diff));
        }

        double distance = Math.Sqrt(_numOps.ToDouble(squaredDistance));

        // Compute sin(π * d / period)
        double sineArg = Math.PI * distance / _period;
        double sineValue = Math.Sin(sineArg);

        // Compute exp(-2 * sin²(...) / lengthScale²)
        double exponent = -2.0 * sineValue * sineValue / (_lengthScale * _lengthScale);

        return _numOps.FromDouble(Math.Exp(exponent));
    }
}
