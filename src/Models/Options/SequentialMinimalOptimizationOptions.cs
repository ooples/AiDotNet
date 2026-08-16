namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the Sequential Minimal Optimization solver used to train support-vector
/// machines.
/// </summary>
public class SequentialMinimalOptimizationOptions
{
    /// <summary>
    /// Gets or sets the maximum number of multiplier pairs optimized.
    /// </summary>
    /// <value>The iteration limit, defaulting to 1000000.</value>
    /// <remarks>
    /// <para>
    /// Each iteration optimizes one pair exactly, so this counts successful steps rather than
    /// passes over the data. The default is deliberately generous: SMO's cost per step is tiny, and
    /// stopping early leaves multipliers that still violate the KKT conditions, which shows up as a
    /// silently worse decision boundary rather than an error.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000000;

    /// <summary>
    /// Gets or sets how far a multiplier may violate the KKT conditions before it is considered for
    /// optimization.
    /// </summary>
    /// <value>The KKT tolerance, defaulting to 1e-3.</value>
    /// <remarks>
    /// <para>
    /// This is the value Platt (1998) uses and LIBSVM defaults to. It is far looser than the
    /// tolerances elsewhere in this library for a good reason: the multipliers only need to be
    /// accurate enough to place the decision boundary correctly, and tightening it multiplies
    /// training time for a boundary that does not visibly move.
    /// </para>
    /// <para><b>For Beginners:</b> How close to perfect the solution has to be before training
    /// stops. Tighter is slower and rarely changes the predictions.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the smallest relative multiplier movement that counts as progress.
    /// </summary>
    /// <value>The step epsilon, defaulting to 1e-12.</value>
    /// <remarks>
    /// <para>
    /// A pair whose optimal step is smaller than this is rejected, which prevents the solver from
    /// looping forever on steps that are pure floating-point noise. It also decides ties when the
    /// objective's curvature along the segment is non-positive.
    /// </para>
    /// </remarks>
    public double StepEpsilon { get; set; } = 1e-12;
}
