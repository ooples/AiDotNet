namespace AiDotNet.Models;

public class PredictionStats<T>
{
    private readonly INumericOperations<T> _ops;

    // Prediction Intervals
    public T UpperPredictionInterval { get; private set; }
    public T LowerPredictionInterval { get; private set; }
    public T PredictionIntervalCoverage { get; private set; }

    // Confidence Intervals for predictions
    public T UpperConfidenceInterval { get; private set; }
    public T LowerConfidenceInterval { get; private set; }

    // Prediction Errors
    public T MeanPredictionError { get; private set; }
    public T MedianPredictionError { get; private set; }

    // Model Performance Metrics
    public T R2 { get; private set; }
    public T AdjustedR2 { get; private set; }
    public T ExplainedVarianceScore { get; private set; }

    // Confidence and Credible Levels
    public T UpperConfidenceLevel { get; private set; }
    public T LowerConfidenceLevel { get; private set; }
    public T UpperCredibleLevel { get; private set; }
    public T LowerCredibleLevel { get; private set; }

    // Distribution Fit
    public DistributionFitResult<T> BestDistributionFit { get; private set; } = new();

    public PredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel, INumericOperations<T>? ops = null)
    {
        _ops = ops ?? MathHelper.GetNumericOperations<T>();

        // Initialize all properties with _ops.Zero
        UpperPredictionInterval = _ops.Zero;
        LowerPredictionInterval = _ops.Zero;
        PredictionIntervalCoverage = _ops.Zero;
        UpperConfidenceInterval = _ops.Zero;
        LowerConfidenceInterval = _ops.Zero;
        MeanPredictionError = _ops.Zero;
        MedianPredictionError = _ops.Zero;
        R2 = _ops.Zero;
        AdjustedR2 = _ops.Zero;
        ExplainedVarianceScore = _ops.Zero;
        UpperConfidenceLevel = _ops.Zero;
        LowerConfidenceLevel = _ops.Zero;
        UpperCredibleLevel = _ops.Zero;
        LowerCredibleLevel = _ops.Zero;

        CalculatePredictionStats(actual, predicted, numberOfParameters, confidenceLevel);
    }

    private void CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel)
    {
        // Determine best fit distribution
        BestDistributionFit = StatisticsHelper<T>.DetermineBestFitDistribution(predicted);

        // Calculate prediction intervals
        (LowerPredictionInterval, UpperPredictionInterval) = StatisticsHelper<T>.CalculatePredictionIntervals(actual, predicted, confidenceLevel);
        PredictionIntervalCoverage = StatisticsHelper<T>.CalculatePredictionIntervalCoverage(actual, predicted, LowerPredictionInterval, UpperPredictionInterval);

        // Calculate confidence intervals for predictions
        (LowerConfidenceInterval, UpperConfidenceInterval) = StatisticsHelper<T>.CalculateConfidenceIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);

        // Calculate prediction errors
        MeanPredictionError = StatisticsHelper<T>.CalculateMeanPredictionError(actual, predicted);
        MedianPredictionError = StatisticsHelper<T>.CalculateMedianPredictionError(actual, predicted);

        // Calculate model performance metrics
        R2 = StatisticsHelper<T>.CalculateR2(actual, predicted);
        AdjustedR2 = StatisticsHelper<T>.CalculateAdjustedR2(R2, actual.Length, numberOfParameters);
        ExplainedVarianceScore = StatisticsHelper<T>.CalculateExplainedVarianceScore(actual, predicted);

        // Calculate credible intervals
        (LowerCredibleLevel, UpperCredibleLevel) = StatisticsHelper<T>.CalculateCredibleIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);
    }
}