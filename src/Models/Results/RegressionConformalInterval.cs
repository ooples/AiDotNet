namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a conformal prediction interval for regression-style outputs.
/// </summary>
/// <typeparam name="TOutput">The output type (e.g., <c>Tensor&lt;T&gt;</c>, <c>Vector&lt;T&gt;</c>, or scalar type).</typeparam>
/// <remarks>
/// <para>
/// The interval is typically computed as <c>[prediction - q, prediction + q]</c> where <c>q</c> is a quantile of calibration residuals.
/// </para>
/// <para><b>For Beginners:</b> Instead of a single number, you get a range where the true value is likely to fall.</para>
/// </remarks>
public sealed class RegressionConformalInterval<TOutput>
{
    /// <summary>
    /// Gets the lower bound of the interval.
    /// </summary>
    public TOutput Lower { get; }

    /// <summary>
    /// Gets the upper bound of the interval.
    /// </summary>
    public TOutput Upper { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="RegressionConformalInterval{TOutput}"/> class.
    /// </summary>
    /// <param name="lower">Lower bound.</param>
    /// <param name="upper">Upper bound.</param>
    public RegressionConformalInterval(TOutput lower, TOutput upper)
    {
        Lower = lower;
        Upper = upper;
    }
}

