using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on model uncertainty reaching a threshold.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion stops learning when the model becomes
/// sufficiently confident on the unlabeled data. If the model has low uncertainty
/// on remaining samples, labeling them is unlikely to help much.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Computes average uncertainty across the unlabeled pool</description></item>
/// <item><description>Compares against a threshold value</description></item>
/// <item><description>Optionally requires a fraction of samples to be confident</description></item>
/// </list>
///
/// <para><b>Key Parameters:</b></para>
/// <list type="bullet">
/// <item><description><b>Threshold:</b> Uncertainty below this is considered "confident"</description></item>
/// <item><description><b>Required Fraction:</b> What fraction of samples must be confident</description></item>
/// </list>
///
/// <para><b>Common Uncertainty Measures:</b></para>
/// <list type="bullet">
/// <item><description>Entropy: For multi-class classification (max = log(K) for K classes)</description></item>
/// <item><description>1 - max(p): For any classifier (range 0-1)</description></item>
/// </list>
/// </remarks>
public class UncertaintyThresholdCriterion<T> : IUncertaintyBasedCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _requiredConfidentFraction;
    private T _currentUncertainty;
    private T _confidentFraction;

    /// <inheritdoc/>
    public string Name => "Uncertainty Threshold";

    /// <inheritdoc/>
    public string Description =>
        $"Stops when {NumOps.ToDouble(_requiredConfidentFraction):P0} of samples have uncertainty below {NumOps.ToDouble(UncertaintyThreshold):F3}";

    /// <inheritdoc/>
    public T UncertaintyThreshold { get; set; }

    /// <inheritdoc/>
    public T CurrentAverageUncertainty => _currentUncertainty;

    /// <inheritdoc/>
    public T FractionConfident => _confidentFraction;

    /// <summary>
    /// Initializes a new UncertaintyThreshold criterion with default parameters.
    /// </summary>
    public UncertaintyThresholdCriterion()
        : this(uncertaintyThreshold: 0.1, requiredConfidentFraction: 0.95)
    {
    }

    /// <summary>
    /// Initializes a new UncertaintyThreshold criterion with specified parameters.
    /// </summary>
    /// <param name="uncertaintyThreshold">Uncertainty below this is considered confident.</param>
    /// <param name="requiredConfidentFraction">Fraction of samples that must be confident to stop.</param>
    public UncertaintyThresholdCriterion(double uncertaintyThreshold, double requiredConfidentFraction = 0.95)
    {
        UncertaintyThreshold = NumOps.FromDouble(uncertaintyThreshold > 0 ? uncertaintyThreshold : 0.1);
        _requiredConfidentFraction = NumOps.FromDouble(MathHelper.Clamp(requiredConfidentFraction, 0.5, 1.0));
        _currentUncertainty = NumOps.One;
        _confidentFraction = NumOps.Zero;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Need uncertainty history to evaluate
        if (context.UncertaintyHistory == null || context.UncertaintyHistory.Count == 0)
        {
            return false;
        }

        // Use the most recent uncertainty value
        _currentUncertainty = context.UncertaintyHistory[^1];

        // Estimate confident fraction from uncertainty history trend
        // If uncertainty is below threshold, assume high confidence
        if (NumOps.Compare(_currentUncertainty, UncertaintyThreshold) <= 0)
        {
            _confidentFraction = NumOps.One;
        }
        else
        {
            // Estimate fraction based on how close we are to threshold
            var ratio = NumOps.Divide(UncertaintyThreshold, _currentUncertainty);
            _confidentFraction = ratio;
        }

        return NumOps.Compare(_confidentFraction, _requiredConfidentFraction) >= 0;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        // Progress is fraction confident / required fraction
        if (NumOps.Compare(_requiredConfidentFraction, NumOps.Zero) <= 0)
        {
            return NumOps.One;
        }

        var progress = NumOps.Divide(_confidentFraction, _requiredConfidentFraction);
        var progressDouble = Math.Min(1.0, NumOps.ToDouble(progress));
        return NumOps.FromDouble(progressDouble);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _currentUncertainty = NumOps.One;
        _confidentFraction = NumOps.Zero;
    }
}
