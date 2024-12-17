namespace AiDotNet.FitDetectors;

public class CrossValidationFitDetector<T> : FitDetectorBase<T>
{
    private readonly CrossValidationFitDetectorOptions _options;

    private readonly T _overfitThreshold;
    private readonly T _underfitThreshold;
    private readonly T _goodFitThreshold;

    public CrossValidationFitDetector(CrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new();
        _overfitThreshold = _numOps.FromDouble(_options.OverfitThreshold);
        _underfitThreshold = _numOps.FromDouble(_options.UnderfitThreshold);
        _goodFitThreshold = _numOps.FromDouble(_options.GoodFitThreshold);
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
        var fitType = DetermineFitType(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        var confidenceLevel = CalculateConfidenceLevel(
            trainingErrorStats, validationErrorStats, testErrorStats,
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
        var trainingR2 = trainingPredictionStats.R2;
        var validationR2 = validationPredictionStats.R2;
        var testR2 = testPredictionStats.R2;

        var r2Difference = _numOps.Abs(_numOps.Subtract(trainingR2, validationR2));

        if (_numOps.GreaterThan(trainingR2, _goodFitThreshold) &&
            _numOps.GreaterThan(validationR2, _goodFitThreshold) &&
            _numOps.GreaterThan(testR2, _goodFitThreshold))
        {
            return FitType.Good;
        }
        else if (_numOps.GreaterThan(r2Difference, _overfitThreshold) &&
                 _numOps.GreaterThan(trainingR2, validationR2))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(trainingR2, _underfitThreshold) &&
                 _numOps.LessThan(validationR2, _underfitThreshold) &&
                 _numOps.LessThan(testR2, _underfitThreshold))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(r2Difference, _overfitThreshold))
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
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
        var r2Consistency = _numOps.Divide(
            _numOps.Add(_numOps.Add(trainingPredictionStats.R2, validationPredictionStats.R2), testPredictionStats.R2),
            _numOps.FromDouble(3));

        var mseConsistency = _numOps.Divide(
            _numOps.Add(_numOps.Add(trainingErrorStats.MSE, validationErrorStats.MSE), testErrorStats.MSE),
            _numOps.FromDouble(3));

        var confidenceLevel = _numOps.Multiply(r2Consistency, _numOps.Subtract(_numOps.One, mseConsistency));
        var lessThan = _numOps.LessThan(_numOps.One, confidenceLevel) ? _numOps.One : confidenceLevel;
        return _numOps.GreaterThan(_numOps.Zero, lessThan) ? _numOps.Zero : lessThan;
    }

    protected override List<string> GenerateRecommendations(
        FitType fitType,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Good:
                recommendations.Add("The model shows good fit across all datasets. Consider deploying the model.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model shows signs of overfitting. Consider the following:");
                recommendations.Add("- Increase the amount of training data");
                recommendations.Add("- Apply regularization techniques");
                recommendations.Add("- Simplify the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("The model shows signs of underfitting. Consider the following:");
                recommendations.Add("- Increase model complexity");
                recommendations.Add("- Add more relevant features");
                recommendations.Add("- Reduce regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("The model shows high variance. Consider the following:");
                recommendations.Add("- Increase the amount of training data");
                recommendations.Add("- Apply feature selection techniques");
                recommendations.Add("- Use ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("The model performance is unstable across datasets. Consider the following:");
                recommendations.Add("- Investigate data quality and consistency");
                recommendations.Add("- Apply cross-validation techniques");
                recommendations.Add("- Use more robust feature selection methods");
                break;
        }

        if (_numOps.LessThan(trainingPredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Training R2 ({trainingPredictionStats.R2}) is below the good fit threshold. Consider improving model performance on training data.");
        }

        if (_numOps.LessThan(validationPredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Validation R2 ({validationPredictionStats.R2}) is below the good fit threshold. Focus on improving model generalization.");
        }

        if (_numOps.LessThan(testPredictionStats.R2, _goodFitThreshold))
        {
            recommendations.Add($"Test R2 ({testPredictionStats.R2}) is below the good fit threshold. Evaluate model performance on unseen data.");
        }

        return recommendations;
    }
}