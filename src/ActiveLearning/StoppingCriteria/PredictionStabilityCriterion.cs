using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Active Learning Literature Survey", "https://minds.wisconsin.edu/handle/1793/60660", Year = 2009, Authors = "Burr Settles")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
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
        int total = Math.Min(previous.Length, current.Length);

        if (total == 0)
        {
            return NumOps.One;
        }

        // Check if predictions are "close enough" to be considered agreeing
        var engine = AiDotNetEngine.Current;
        var diff = engine.Abs(engine.Subtract(Prefix(previous, total), Prefix(current, total)));
        int agreed = engine.TensorCountNonzero(engine.TensorLessThan(Tensor<T>.FromVector(diff), NumOps.FromDouble(0.5)));

        return NumOps.FromDouble((double)agreed / total);
    }

    private T ComputeCorrelation(Vector<T> previous, Vector<T> current)
    {
        int n = Math.Min(previous.Length, current.Length);
        if (n == 0)
        {
            return NumOps.One;
        }

        var engine = AiDotNetEngine.Current;
        var prev = Prefix(previous, n);
        var curr = Prefix(current, n);

        // Compute means
        T meanPrev = NumOps.Divide(engine.Sum(prev), NumOps.FromDouble(n));
        T meanCurr = NumOps.Divide(engine.Sum(curr), NumOps.FromDouble(n));

        // Compute correlation
        var diffPrev = engine.TensorSubtractScalar(Tensor<T>.FromVector(prev), meanPrev).ToVector();
        var diffCurr = engine.TensorSubtractScalar(Tensor<T>.FromVector(curr), meanCurr).ToVector();
        T numerator = engine.DotProduct(diffPrev, diffCurr);
        T denomPrev = engine.DotProduct(diffPrev, diffPrev);
        T denomCurr = engine.DotProduct(diffCurr, diffCurr);

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

        var engine = AiDotNetEngine.Current;
        T sumDiff = engine.Sum(engine.Abs(engine.Subtract(Prefix(previous, n), Prefix(current, n))));

        T mad = NumOps.Divide(sumDiff, NumOps.FromDouble(n));

        // Assume values are in [0,1] range, so MAD is in [0,1]
        // Stability = 1 - MAD
        return NumOps.Subtract(NumOps.One, mad);
    }

    /// <summary>
    /// Returns the first <paramref name="length"/> elements of a vector.
    /// </summary>
    private static Vector<T> Prefix(Vector<T> vector, int length)
    {
        return vector.Length == length ? vector : vector.GetSubVector(0, length);
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
