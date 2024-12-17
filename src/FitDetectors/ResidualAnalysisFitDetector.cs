namespace AiDotNet.FitDetectors;

public class ResidualAnalysisFitDetector<T> : FitDetectorBase<T>
{
    private readonly ResidualAnalysisFitDetectorOptions _options;

    public ResidualAnalysisFitDetector(ResidualAnalysisFitDetectorOptions? options = null)
    {
        _options = options ?? new ResidualAnalysisFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var fitType = DetermineFitType(trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        var confidenceLevel = CalculateConfidenceLevel(trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        var recommendations = GenerateRecommendations(fitType, 
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    protected override FitType DetermineFitType(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        // Check for autocorrelation using Durbin-Watson statistic
        if (_numOps.LessThan(testErrorStats.DurbinWatsonStatistic, _numOps.FromDouble(1.5)) || 
            _numOps.GreaterThan(testErrorStats.DurbinWatsonStatistic, _numOps.FromDouble(2.5)))
        {
            return FitType.Unstable;
        }

        // Check MAPE for overall fit
        if (_numOps.GreaterThan(testErrorStats.MAPE, _numOps.FromDouble(_options.MapeThreshold)))
        {
            return FitType.Underfit;
        }

        // Analyze residuals across datasets
        var meanThreshold = _numOps.FromDouble(_options.MeanThreshold);
        var stdThreshold = _numOps.FromDouble(_options.StdThreshold);

        var trainingResidualMean = trainingErrorStats.MeanBiasError;
        var validationResidualMean = validationErrorStats.MeanBiasError;
        var testResidualMean = testErrorStats.MeanBiasError;

        var trainingResidualStd = trainingErrorStats.PopulationStandardError;
        var validationResidualStd = validationErrorStats.PopulationStandardError;
        var testResidualStd = testErrorStats.PopulationStandardError;

        if (_numOps.LessThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
            _numOps.LessThan(_numOps.Abs(validationResidualMean), meanThreshold) &&
            _numOps.LessThan(_numOps.Abs(testResidualMean), meanThreshold))
        {
            if (_numOps.LessThan(trainingResidualStd, stdThreshold) &&
                _numOps.LessThan(validationResidualStd, stdThreshold) &&
                _numOps.LessThan(testResidualStd, stdThreshold))
            {
                return FitType.Good;
            }
            else
            {
                return FitType.HighVariance;
            }
        }
        else if (_numOps.GreaterThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(validationResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(testResidualMean), meanThreshold))
        {
            return FitType.HighBias;
        }
        else if (_numOps.LessThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(validationResidualMean), meanThreshold))
        {
            return FitType.Overfit;
        }

        // Check for significant differences in R-squared values
        var r2Threshold = _numOps.FromDouble(_options.R2Threshold);
        if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(trainingPredictionStats.R2, validationPredictionStats.R2)), r2Threshold) ||
            _numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(trainingPredictionStats.R2, testPredictionStats.R2)), r2Threshold))
        {
            return FitType.Unstable;
        }

        return FitType.Good;
    }

    protected override T CalculateConfidenceLevel(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var trainingConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(trainingErrorStats.PopulationStandardError, trainingErrorStats.MeanBiasError));
        var validationConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(validationErrorStats.PopulationStandardError, validationErrorStats.MeanBiasError));
        var testConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(testErrorStats.PopulationStandardError, testErrorStats.MeanBiasError));

        var averageConfidence = _numOps.Divide(_numOps.Add(_numOps.Add(trainingConfidence, validationConfidence), testConfidence), _numOps.FromDouble(3));

        // Adjust confidence based on R-squared values
        var r2Adjustment = _numOps.Divide(_numOps.Add(_numOps.Add(trainingPredictionStats.R2, validationPredictionStats.R2), testPredictionStats.R2), _numOps.FromDouble(3));
        
        return _numOps.Multiply(averageConfidence, r2Adjustment);
    }
}