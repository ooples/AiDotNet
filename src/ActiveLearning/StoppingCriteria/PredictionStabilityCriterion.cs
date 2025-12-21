using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on prediction stability across iterations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion stops learning when the model's predictions
/// on the unlabeled pool stop changing significantly. Stable predictions indicate the
/// model has converged and won't benefit much from more labeled data.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Stores predictions from the previous iteration</description></item>
/// <item><description>Compares current predictions to previous predictions</description></item>
/// <item><description>Measures change using various metrics (agreement, correlation, etc.)</description></item>
/// <item><description>Stops when stability exceeds threshold for patience iterations</description></item>
/// </list>
///
/// <para><b>Stability Measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Agreement Rate:</b> Fraction of samples with same predicted class</description></item>
/// <item><description><b>Probability Correlation:</b> Correlation of prediction probabilities</description></item>
/// <item><description><b>Mean Absolute Difference:</b> Average change in probabilities</description></item>
/// </list>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Model predictions are available for all unlabeled samples</description></item>
/// <item><description>Care about model behavior more than just accuracy</description></item>
/// </list>
/// </remarks>
public class PredictionStabilityCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _stabilityThreshold;
    private readonly int _patience;
    private readonly StabilityMeasure _measure;

    private int _stableIterations;
    private T _currentStability;

    /// <inheritdoc/>
    public string Name => $"Prediction Stability ({_measure})";

    /// <inheritdoc/>
    public string Description =>
        $"Stops when {_measure} stability exceeds {NumOps.ToDouble(_stabilityThreshold):P1} for {_patience} consecutive iterations";

    /// <summary>
    /// Gets the current stability measurement.
    /// </summary>
    public T CurrentStability => _currentStability;

    /// <summary>
    /// Gets the number of consecutive stable iterations.
    /// </summary>
    public int StableIterations => _stableIterations;

    /// <summary>
    /// Initializes a new PredictionStability criterion with default parameters.
    /// </summary>
    public PredictionStabilityCriterion()
        : this(stabilityThreshold: 0.99, patience: 3, measure: StabilityMeasure.Agreement)
    {
    }

    /// <summary>
    /// Initializes a new PredictionStability criterion with specified parameters.
    /// </summary>
    /// <param name="stabilityThreshold">Stability level required to consider stable.</param>
    /// <param name="patience">Number of stable iterations before stopping.</param>
    /// <param name="measure">The stability measure to use.</param>
    public PredictionStabilityCriterion(double stabilityThreshold, int patience, StabilityMeasure measure)
    {
        _stabilityThreshold = NumOps.FromDouble(MathHelper.Clamp(stabilityThreshold, 0.8, 1.0));
        _patience = patience > 0 ? patience : 3;
        _measure = measure;

        _stableIterations = 0;
        _currentStability = NumOps.Zero;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Need both current and previous predictions
        if (context.PreviousPredictions == null || context.CurrentPredictions == null)
        {
            return false;
        }

        // Compute stability
        _currentStability = ComputeStability(context.PreviousPredictions, context.CurrentPredictions);

        // Check if stable
        if (NumOps.Compare(_currentStability, _stabilityThreshold) >= 0)
        {
            _stableIterations++;
        }
        else
        {
            _stableIterations = 0;
        }

        return _stableIterations >= _patience;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        // Progress combines stability level and patience progress
        var stabilityProgress = NumOps.ToDouble(_currentStability);
        var patienceProgress = (double)_stableIterations / _patience;

        // Weight towards stability being the main indicator
        var progress = 0.7 * stabilityProgress + 0.3 * patienceProgress;
        return NumOps.FromDouble(Math.Min(1.0, progress));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _stableIterations = 0;
        _currentStability = NumOps.Zero;
    }

    #region Private Methods

    private T ComputeStability(Vector<T> previous, Vector<T> current)
    {
        return _measure switch
        {
            StabilityMeasure.Agreement => ComputeAgreementRate(previous, current),
            StabilityMeasure.Correlation => ComputeCorrelation(previous, current),
            StabilityMeasure.MeanAbsoluteDifference => ComputeMADStability(previous, current),
            _ => ComputeAgreementRate(previous, current)
        };
    }

    private T ComputeAgreementRate(Vector<T> previous, Vector<T> current)
    {
        // For classification: fraction of samples where argmax is the same
        // For regression: this treats values as discrete decisions
        int agreed = 0;
        int total = Math.Min(previous.Length, current.Length);

        if (total == 0)
        {
            return NumOps.One;
        }

        for (int i = 0; i < total; i++)
        {
            // Check if predictions are "close enough" to be considered agreeing
            var diff = NumOps.Abs(NumOps.Subtract(previous[i], current[i]));
            if (NumOps.Compare(diff, NumOps.FromDouble(0.5)) < 0)
            {
                agreed++;
            }
        }

        return NumOps.FromDouble((double)agreed / total);
    }

    private T ComputeCorrelation(Vector<T> previous, Vector<T> current)
    {
        int n = Math.Min(previous.Length, current.Length);
        if (n == 0)
        {
            return NumOps.One;
        }

        // Compute means
        T sumPrev = NumOps.Zero;
        T sumCurr = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            sumPrev = NumOps.Add(sumPrev, previous[i]);
            sumCurr = NumOps.Add(sumCurr, current[i]);
        }

        T meanPrev = NumOps.Divide(sumPrev, NumOps.FromDouble(n));
        T meanCurr = NumOps.Divide(sumCurr, NumOps.FromDouble(n));

        // Compute correlation
        T numerator = NumOps.Zero;
        T denomPrev = NumOps.Zero;
        T denomCurr = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T diffPrev = NumOps.Subtract(previous[i], meanPrev);
            T diffCurr = NumOps.Subtract(current[i], meanCurr);

            numerator = NumOps.Add(numerator, NumOps.Multiply(diffPrev, diffCurr));
            denomPrev = NumOps.Add(denomPrev, NumOps.Multiply(diffPrev, diffPrev));
            denomCurr = NumOps.Add(denomCurr, NumOps.Multiply(diffCurr, diffCurr));
        }

        T denominator = NumOps.Multiply(NumOps.Sqrt(denomPrev), NumOps.Sqrt(denomCurr));

        if (NumOps.Compare(denominator, NumOps.FromDouble(1e-10)) < 0)
        {
            return NumOps.One; // No variance means perfect stability
        }

        T correlation = NumOps.Divide(numerator, denominator);

        // Map correlation (-1 to 1) to stability (0 to 1)
        return NumOps.Divide(NumOps.Add(correlation, NumOps.One), NumOps.FromDouble(2.0));
    }

    private T ComputeMADStability(Vector<T> previous, Vector<T> current)
    {
        // Stability = 1 - mean absolute difference
        int n = Math.Min(previous.Length, current.Length);
        if (n == 0)
        {
            return NumOps.One;
        }

        T sumDiff = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            sumDiff = NumOps.Add(sumDiff, NumOps.Abs(NumOps.Subtract(previous[i], current[i])));
        }

        T mad = NumOps.Divide(sumDiff, NumOps.FromDouble(n));

        // Assume values are in [0,1] range, so MAD is in [0,1]
        // Stability = 1 - MAD
        return NumOps.Subtract(NumOps.One, mad);
    }

    #endregion
}

/// <summary>
/// Stability measurement methods for prediction stability criterion.
/// </summary>
public enum StabilityMeasure
{
    /// <summary>
    /// Fraction of samples with same predicted class/decision.
    /// </summary>
    Agreement,

    /// <summary>
    /// Pearson correlation between prediction probabilities.
    /// </summary>
    Correlation,

    /// <summary>
    /// One minus mean absolute difference of predictions.
    /// </summary>
    MeanAbsoluteDifference
}
