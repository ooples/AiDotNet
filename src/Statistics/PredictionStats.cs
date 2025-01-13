namespace AiDotNet.Statistics;

public class PredictionStats<T>
{
    private readonly INumericOperations<T> NumOps;

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
    public List<T> LearningCurve { get; private set; }
    public T Accuracy { get; private set; }
    public T Precision { get; private set; }
    public T Recall { get; private set; }
    public T F1Score { get; private set; }
    public T PearsonCorrelation { get; private set; }
    public T SpearmanCorrelation { get; private set; }
    public T KendallTau { get; private set; }
    public T DynamicTimeWarping { get; private set; }

    // Distribution Fit
    public DistributionFitResult<T> BestDistributionFit { get; private set; } = new();

    internal PredictionStats(PredictionStatsInputs<T> inputs)
    {
        NumOps = MathHelper.GetNumericOperations<T>();

        // Initialize all properties
        PredictionInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        ConfidenceInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        CredibleInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        ToleranceInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        ForecastInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        BootstrapInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        SimultaneousPredictionInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        JackknifeInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        PercentileInterval = (Lower: NumOps.Zero, Upper: NumOps.Zero);
        QuantileIntervals = [];
        PredictionIntervalCoverage = NumOps.Zero;
        MeanPredictionError = NumOps.Zero;
        MedianPredictionError = NumOps.Zero;
        R2 = NumOps.Zero;
        AdjustedR2 = NumOps.Zero;
        PearsonCorrelation = NumOps.Zero;
        SpearmanCorrelation = NumOps.Zero;
        KendallTau = NumOps.Zero;
        DynamicTimeWarping = NumOps.Zero;
        ExplainedVarianceScore = NumOps.Zero;
        LearningCurve = [];
        Accuracy = NumOps.Zero;
        Precision = NumOps.Zero;
        Recall = NumOps.Zero;
        F1Score = NumOps.Zero;

        CalculatePredictionStats(inputs.Actual, inputs.Predicted, inputs.NumberOfParameters, NumOps.FromDouble(inputs.ConfidenceLevel), inputs.LearningCurveSteps, 
            inputs.PredictionType);
    }

    public static PredictionStats<T> Empty()
    {
        return new PredictionStats<T>(new());
    }

    private void CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel, int learningCurveSteps, PredictionType predictionType)
    {
        BestDistributionFit = StatisticsHelper<T>.DetermineBestFitDistribution(predicted);

        MeanPredictionError = StatisticsHelper<T>.CalculateMeanPredictionError(actual, predicted);
        MedianPredictionError = StatisticsHelper<T>.CalculateMedianPredictionError(actual, predicted);

        R2 = StatisticsHelper<T>.CalculateR2(actual, predicted);
        AdjustedR2 = StatisticsHelper<T>.CalculateAdjustedR2(R2, actual.Length, numberOfParameters);
        ExplainedVarianceScore = StatisticsHelper<T>.CalculateExplainedVarianceScore(actual, predicted);
        LearningCurve = StatisticsHelper<T>.CalculateLearningCurve(actual, predicted, learningCurveSteps);
        PearsonCorrelation = StatisticsHelper<T>.CalculatePearsonCorrelationCoefficient(actual, predicted);
        SpearmanCorrelation = StatisticsHelper<T>.CalculateSpearmanRankCorrelationCoefficient(actual, predicted);
        KendallTau = StatisticsHelper<T>.CalculateKendallTau(actual, predicted);
        DynamicTimeWarping = StatisticsHelper<T>.CalculateDynamicTimeWarping(actual, predicted);

        PredictionInterval = StatisticsHelper<T>.CalculatePredictionIntervals(actual, predicted, confidenceLevel);
        PredictionIntervalCoverage = StatisticsHelper<T>.CalculatePredictionIntervalCoverage(actual, predicted, PredictionInterval.Lower, PredictionInterval.Upper);
        ConfidenceInterval = StatisticsHelper<T>.CalculateConfidenceIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);
        CredibleInterval = StatisticsHelper<T>.CalculateCredibleIntervals(predicted, confidenceLevel, BestDistributionFit.DistributionType);
        ToleranceInterval = StatisticsHelper<T>.CalculateToleranceInterval(actual, predicted, confidenceLevel);
        ForecastInterval = StatisticsHelper<T>.CalculateForecastInterval(actual, predicted, confidenceLevel);
        QuantileIntervals = StatisticsHelper<T>.CalculateQuantileIntervals(actual, predicted, [NumOps.FromDouble(0.25), NumOps.FromDouble(0.5), NumOps.FromDouble(0.75)]);
        BootstrapInterval = StatisticsHelper<T>.CalculateBootstrapInterval(actual, predicted, confidenceLevel);
        SimultaneousPredictionInterval = StatisticsHelper<T>.CalculateSimultaneousPredictionInterval(actual, predicted, confidenceLevel);
        JackknifeInterval = StatisticsHelper<T>.CalculateJackknifeInterval(actual, predicted);
        PercentileInterval = StatisticsHelper<T>.CalculatePercentileInterval(predicted, confidenceLevel);

        Accuracy = StatisticsHelper<T>.CalculateAccuracy(actual, predicted, predictionType);
        (Precision, Recall, F1Score) = StatisticsHelper<T>.CalculatePrecisionRecallF1(actual, predicted, predictionType);
    }

    public T GetMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.R2 => R2,
            MetricType.AdjustedR2 => AdjustedR2,
            MetricType.ExplainedVarianceScore => ExplainedVarianceScore,
            MetricType.MeanPredictionError => MeanPredictionError,
            MetricType.MedianPredictionError => MedianPredictionError,
            MetricType.Accuracy => Accuracy,
            MetricType.Precision => Precision,
            MetricType.Recall => Recall,
            MetricType.F1Score => F1Score,
            MetricType.PredictionIntervalCoverage => PredictionIntervalCoverage,
            MetricType.PearsonCorrelation => PearsonCorrelation,
            MetricType.SpearmanCorrelation => SpearmanCorrelation,
            MetricType.KendallTau => KendallTau,
            _ => throw new ArgumentException($"Unknown metric type: {metricType}")
        };
    }
}