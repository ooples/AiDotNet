namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for multi-fidelity AutoML search.
/// </summary>
/// <remarks>
/// <para>
/// Multi-fidelity search tries many configurations quickly with a smaller "budget" (for example, a subset of the
/// training data) and then promotes only the most promising trials to larger budgets.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of fully training every trial (slow), multi-fidelity does:
/// <list type="number">
/// <item><description>Train many candidates on a small amount of data.</description></item>
/// <item><description>Keep only the best candidates.</description></item>
/// <item><description>Train those candidates on more data.</description></item>
/// <item><description>Repeat until you reach full training.</description></item>
/// </list>
/// This usually finds strong models faster than running full training for every attempt.
/// </para>
/// </remarks>
public sealed class AutoMLMultiFidelityOptions
{
    /// <summary>
    /// Gets or sets the ordered list of training-data fractions to use as fidelity levels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Values must be in (0, 1]. The final level should be 1.0 to represent full-fidelity training.
    /// If the list does not include 1.0, multi-fidelity will append it automatically.
    /// </para>
    /// <para><b>For Beginners:</b> <c>0.25</c> means "train on 25% of the training data".</para>
    /// </remarks>
    public double[] TrainingFractions { get; set; } = { 0.25, 0.5, 1.0 };

    /// <summary>
    /// Gets or sets the reduction factor used when promoting trials between fidelity levels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 3 means "keep about 1/3 of the trials" when moving to the next fidelity level.
    /// </para>
    /// <para><b>For Beginners:</b> Higher values are more aggressive (fewer promotions).</para>
    /// </remarks>
    public double ReductionFactor { get; set; } = 3.0;
}
