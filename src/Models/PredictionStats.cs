public class PredictionStats<T>
{
    private readonly INumericOperations<T> _ops;

    // Intervals
    public (T Lower, T Upper) PredictionInterval { get; private set; }
    public (T Lower, T Upper) ConfidenceInterval { get; private set; }
    public (T Lower, T Upper) CredibleInterval { get; private set; }
    public (T Lower, T Upper) ToleranceInterval { get; private set; }
    public (T Lower, T Upper) ForecastInterval { get; private set; }
    public (T Lower, T Upper) BootstrapInterval { get; private set; }
    public (T Lower, T Upper) SimultaneousPredictionInterval { get; private set; }
    public (T Lower, T Upper) JackknifeInterval { get; private set; }
    public (T Lower, T Upper) PercentileInterval { get; private set; }

    public List<(T Quantile, T Lower, T Upper)> QuantileIntervals { get; private set; }
    public T PredictionIntervalCoverage { get; private set; }

    // Prediction Errors
    public T MeanPredictionError { get; private set; }
    public T MedianPredictionError { get; private set; }

    // Model Performance Metrics
    public T R2 { get; private set; }
    public T AdjustedR2 { get; private set; }
    public T ExplainedVarianceScore { get; private set; }

    // Distribution Fit
    public DistributionFitResult<T> BestDistributionFit { get; private set; } = new();

    public PredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel, INumericOperations<T>? ops = null)
    {
        _ops = ops ?? MathHelper.GetNumericOperations<T>();

        // Initialize all properties
        PredictionInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        ConfidenceInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        CredibleInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        ToleranceInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        ForecastInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        BootstrapInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        SimultaneousPredictionInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        JackknifeInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        PercentileInterval = (Lower: _ops.Zero, Upper: _ops.Zero);
        QuantileIntervals = [];
        PredictionIntervalCoverage = _ops.Zero;
        MeanPredictionError = _ops.Zero;
        MedianPredictionError = _ops.Zero;
        R2 = _ops.Zero;
        AdjustedR2 = _ops.Zero;
        ExplainedVarianceScore = _ops.Zero;

        CalculatePredictionStats(actual, predicted, numberOfParameters, confidenceLevel);
    }

    public static PredictionStats<T> Empty()
    {
        return new PredictionStats<T>(Vector<T>.Empty(), Vector<T>.Empty(), 0, MathHelper.GetNumericOperations<T>().Zero);
    }

    private void CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel)
    {
        BestDistributionFit = StatisticsHelper<T>.DetermineBestFitDistribution(predicted);

        MeanPredictionError = StatisticsHelper<T>.CalculateMeanPredictionError(actual, predicted);
        MedianPredictionError = StatisticsHelper<T>.CalculateMedianPredictionError(actual, predicted);

        R2 = StatisticsHelper<T>.CalculateR2(actual, predicted);
        AdjustedR2 = StatisticsHelper<T>.CalculateAdjustedR2(R2, actual.Length, numberOfParameters);
        ExplainedVarianceScore = StatisticsHelper<T>.CalculateExplainedVarianceScore(actual, predicted);

        PredictionInterval = StatisticsHelper<T>.CalculatePredictionIntervals(actual, predicted, confidenceLevel);
        PredictionIntervalCoverage = StatisticsHelper<T>.CalculatePredictionIntervalCoverage(actual, predicted, PredictionInterval.Lower, PredictionInterval.Upper);
        ConfidenceInterval = StatisticsHelper<T>.CalculateConfidenceIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);
        CredibleInterval = StatisticsHelper<T>.CalculateCredibleIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);
        ToleranceInterval = StatisticsHelper<T>.CalculateToleranceInterval(actual, predicted, confidenceLevel);
        ForecastInterval = StatisticsHelper<T>.CalculateForecastInterval(actual, predicted, confidenceLevel);
        QuantileIntervals = StatisticsHelper<T>.CalculateQuantileIntervals(actual, predicted, [_ops.FromDouble(0.25), _ops.FromDouble(0.5), _ops.FromDouble(0.75)]);
        BootstrapInterval = StatisticsHelper<T>.CalculateBootstrapInterval(actual, predicted, confidenceLevel);
        SimultaneousPredictionInterval = StatisticsHelper<T>.CalculateSimultaneousPredictionInterval(actual, predicted, confidenceLevel);
        JackknifeInterval = StatisticsHelper<T>.CalculateJackknifeInterval(actual, predicted);
        PercentileInterval = StatisticsHelper<T>.CalculatePercentileInterval(predicted, confidenceLevel);
    }
}