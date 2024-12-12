public class DefaultFitDetector<T> : IFitDetector<T>
{
    private readonly INumericOperations<T> _numOps;

    public DefaultFitDetector()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public FitDetectorResult<T> DetectFit(
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
        var fitType = DetermineFitType(trainingPredictionStats, validationPredictionStats, testPredictionStats);
        var confidenceLevel = CalculateConfidenceLevel(trainingPredictionStats, validationPredictionStats, testPredictionStats);
        var recommendations = GenerateRecommendations(fitType);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    private FitType DetermineFitType(PredictionStats<T> training, PredictionStats<T> validation, PredictionStats<T> test)
    {
        T threshold09 = _numOps.FromDouble(0.9);
        T threshold07 = _numOps.FromDouble(0.7);
        T threshold05 = _numOps.FromDouble(0.5);
        T threshold02 = _numOps.FromDouble(0.2);

        if (_numOps.GreaterThan(training.R2, threshold09) && _numOps.GreaterThan(validation.R2, threshold09) && _numOps.GreaterThan(test.R2, threshold09))
            return FitType.Good;
        if (_numOps.GreaterThan(training.R2, threshold09) && _numOps.LessThan(validation.R2, threshold07))
            return FitType.Overfit;
        if (_numOps.LessThan(training.R2, threshold07) && _numOps.LessThan(validation.R2, threshold07))
            return FitType.Underfit;
        if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(training.R2, validation.R2)), threshold02))
            return FitType.HighVariance;
        if (_numOps.LessThan(training.R2, threshold05) && _numOps.LessThan(validation.R2, threshold05) && _numOps.LessThan(test.R2, threshold05))
            return FitType.HighBias;
        
        return FitType.Unstable;
    }

    private T CalculateConfidenceLevel(PredictionStats<T> training, PredictionStats<T> validation, PredictionStats<T> test)
    {
        return _numOps.Divide(_numOps.Add(_numOps.Add(training.R2, validation.R2), test.R2), _numOps.FromDouble(3));
    }

    private List<string> GenerateRecommendations(FitType fitType)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization");
                recommendations.Add("Try reducing model complexity");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider decreasing regularization");
                recommendations.Add("Try increasing model complexity");
                break;
            case FitType.HighVariance:
                recommendations.Add("Collect more training data");
                recommendations.Add("Try feature selection to reduce noise");
                break;
            case FitType.HighBias:
                recommendations.Add("Add more features");
                recommendations.Add("Increase model complexity");
                break;
            case FitType.Unstable:
                recommendations.Add("Check for data quality issues");
                recommendations.Add("Consider ensemble methods for more stable predictions");
                break;
        }

        return recommendations;
    }
}