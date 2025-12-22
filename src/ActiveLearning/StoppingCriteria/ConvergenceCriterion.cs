using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on learning curve convergence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion analyzes the learning curve (how performance
/// changes with more data) and stops when adding more data shows diminishing returns.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Fits a power law or logarithmic curve to performance history</description></item>
/// <item><description>Extrapolates expected improvement from more samples</description></item>
/// <item><description>Stops when expected improvement is below threshold</description></item>
/// </list>
///
/// <para><b>Key Insight:</b> Learning curves often follow a power law: performance = a - b*n^(-c)
/// where n is the number of samples. As n grows, improvement decreases.</para>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Want to optimize cost/benefit of labeling</description></item>
/// <item><description>Have enough history to estimate the curve</description></item>
/// <item><description>Model performance is expected to follow standard learning curves</description></item>
/// </list>
///
/// <para><b>Reference:</b> Figueroa et al. "Predicting sample size required for classification performance" (BMC 2012)</para>
/// </remarks>
public class ConvergenceCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _minHistory;
    private readonly T _minImprovement;
    private readonly int _lookAhead;

    private T _predictedImprovement;
    private bool _hasConverged;

    /// <inheritdoc/>
    public string Name => "Learning Curve Convergence";

    /// <inheritdoc/>
    public string Description =>
        $"Stops when expected improvement from {_lookAhead} more samples is below {NumOps.ToDouble(_minImprovement):P2}";

    /// <summary>
    /// Gets the predicted improvement from additional samples.
    /// </summary>
    public T PredictedImprovement => _predictedImprovement;

    /// <summary>
    /// Initializes a new Convergence criterion with default parameters.
    /// </summary>
    public ConvergenceCriterion()
        : this(minHistory: 10, minImprovement: 0.001, lookAhead: 100)
    {
    }

    /// <summary>
    /// Initializes a new Convergence criterion with specified parameters.
    /// </summary>
    /// <param name="minHistory">Minimum iterations before checking convergence.</param>
    /// <param name="minImprovement">Minimum expected improvement to continue.</param>
    /// <param name="lookAhead">Number of future samples to consider.</param>
    public ConvergenceCriterion(int minHistory, double minImprovement, int lookAhead)
    {
        _minHistory = minHistory > 5 ? minHistory : 10;
        _minImprovement = NumOps.FromDouble(minImprovement > 0 ? minImprovement : 0.001);
        _lookAhead = lookAhead > 0 ? lookAhead : 100;

        _predictedImprovement = NumOps.One;
        _hasConverged = false;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Need enough history to estimate learning curve
        if (context.AccuracyHistory == null || context.AccuracyHistory.Count < _minHistory)
        {
            return false;
        }

        // Fit learning curve and predict improvement
        _predictedImprovement = PredictImprovementFromCurve(context);
        _hasConverged = NumOps.Compare(_predictedImprovement, _minImprovement) < 0;

        return _hasConverged;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        // Progress is inverse of predicted improvement ratio
        // When improvement is close to min, we're close to stopping
        if (NumOps.Compare(_minImprovement, NumOps.Zero) <= 0)
        {
            return NumOps.One;
        }

        var ratio = NumOps.Divide(_minImprovement, NumOps.Add(_predictedImprovement, NumOps.FromDouble(1e-10)));
        var progress = Math.Min(1.0, NumOps.ToDouble(ratio));
        return NumOps.FromDouble(progress);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _predictedImprovement = NumOps.One;
        _hasConverged = false;
    }

    #region Private Methods

    private T PredictImprovementFromCurve(ActiveLearningContext<T> context)
    {
        var history = context.AccuracyHistory;
        int n = history.Count;

        // Simple approach: fit linear regression on log-log scale
        // log(1 - accuracy) = log(b) - c * log(n)
        // This captures power law decay

        // Compute current slope using recent points
        int windowSize = Math.Min(5, n / 2);
        if (windowSize < 2)
        {
            return NumOps.One; // Not enough data, assume large improvement expected
        }

        // Compute rate of improvement over last window
        T startPerf = history[n - windowSize];
        T endPerf = history[n - 1];
        T currentImprovement = NumOps.Subtract(endPerf, startPerf);

        // Compute per-sample improvement rate
        T perSampleRate = NumOps.Divide(currentImprovement, NumOps.FromDouble(windowSize));

        // Extrapolate improvement for next _lookAhead samples
        // Apply decay factor as we're on diminishing returns
        double currentN = context.TotalLabeled;
        double futureN = currentN + _lookAhead;

        // Decay factor based on power law: improvement ~ n^(-0.5)
        double decayFactor = Math.Sqrt(currentN / futureN);

        T scaledImprovement = NumOps.Multiply(
            NumOps.Multiply(perSampleRate, NumOps.FromDouble(_lookAhead)),
            NumOps.FromDouble(decayFactor));

        // Ensure non-negative
        if (NumOps.Compare(scaledImprovement, NumOps.Zero) < 0)
        {
            return NumOps.Zero;
        }

        return scaledImprovement;
    }

    #endregion
}
