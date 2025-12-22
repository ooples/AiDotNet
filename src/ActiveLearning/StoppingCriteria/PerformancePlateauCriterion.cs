using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on performance plateau detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion stops learning when the model's performance
/// (accuracy or validation score) stops improving significantly. Adding more labeled data
/// at this point is unlikely to help.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Monitors the performance metric over a sliding window</description></item>
/// <item><description>Computes improvement from start to end of window</description></item>
/// <item><description>Stops if improvement falls below threshold for patience iterations</description></item>
/// </list>
///
/// <para><b>Key Parameters:</b></para>
/// <list type="bullet">
/// <item><description><b>Window Size:</b> Number of iterations to average</description></item>
/// <item><description><b>Patience:</b> Consecutive flat windows before stopping</description></item>
/// <item><description><b>Threshold:</b> Minimum improvement to count as progress</description></item>
/// </list>
///
/// <para><b>Reference:</b> Early stopping is a common regularization technique in deep learning.</para>
/// </remarks>
public class PerformancePlateauCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _windowSize;
    private readonly int _patience;
    private readonly T _minImprovement;
    private readonly bool _useValidation;

    private int _patienceCounter;
    private T _bestPerformance;
    private bool _hasStarted;

    /// <inheritdoc/>
    public string Name => "Performance Plateau";

    /// <inheritdoc/>
    public string Description =>
        $"Stops when {(_useValidation ? "validation" : "training")} accuracy fails to improve by {NumOps.ToDouble(_minImprovement):P1} for {_patience} consecutive windows";

    /// <summary>
    /// Initializes a new PerformancePlateau criterion with default parameters.
    /// </summary>
    public PerformancePlateauCriterion()
        : this(windowSize: 5, patience: 3, minImprovement: 0.001, useValidation: true)
    {
    }

    /// <summary>
    /// Initializes a new PerformancePlateau criterion with specified parameters.
    /// </summary>
    /// <param name="windowSize">Number of iterations in the averaging window.</param>
    /// <param name="patience">Number of flat windows to tolerate before stopping.</param>
    /// <param name="minImprovement">Minimum relative improvement to count as progress.</param>
    /// <param name="useValidation">Whether to use validation or training accuracy.</param>
    public PerformancePlateauCriterion(int windowSize, int patience, double minImprovement, bool useValidation = true)
    {
        _windowSize = windowSize > 0 ? windowSize : 5;
        _patience = patience > 0 ? patience : 3;
        _minImprovement = NumOps.FromDouble(minImprovement > 0 ? minImprovement : 0.001);
        _useValidation = useValidation;

        _patienceCounter = 0;
        _bestPerformance = NumOps.MinValue;
        _hasStarted = false;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        var history = _useValidation ? context.ValidationAccuracyHistory : context.AccuracyHistory;

        if (history == null || history.Count < _windowSize)
        {
            return false;
        }

        // Get current window average
        var currentAvg = ComputeWindowAverage(history, history.Count - _windowSize, _windowSize);

        if (!_hasStarted)
        {
            _bestPerformance = currentAvg;
            _hasStarted = true;
            return false;
        }

        // Check for improvement
        var improvement = NumOps.Subtract(currentAvg, _bestPerformance);
        var relativeImprovement = NumOps.Compare(_bestPerformance, NumOps.Zero) > 0
            ? NumOps.Divide(improvement, _bestPerformance)
            : improvement;

        if (NumOps.Compare(relativeImprovement, _minImprovement) > 0)
        {
            // Significant improvement - reset patience and update best
            _bestPerformance = currentAvg;
            _patienceCounter = 0;
        }
        else
        {
            // No significant improvement - increment patience
            _patienceCounter++;
        }

        return _patienceCounter >= _patience;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        // Progress is how close we are to running out of patience
        var progress = (double)_patienceCounter / _patience;
        return NumOps.FromDouble(Math.Min(1.0, progress));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _patienceCounter = 0;
        _bestPerformance = NumOps.MinValue;
        _hasStarted = false;
    }

    #region Private Methods

    private T ComputeWindowAverage(List<T> history, int start, int length)
    {
        T sum = NumOps.Zero;
        int count = 0;

        for (int i = start; i < start + length && i < history.Count; i++)
        {
            if (i >= 0)
            {
                sum = NumOps.Add(sum, history[i]);
                count++;
            }
        }

        if (count == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(sum, NumOps.FromDouble(count));
    }

    #endregion
}
