public class DefaultFitDetector : IFitDetector
{
    public FitDetectorResult DetectFit(
        ErrorStats trainingErrorStats,
        ErrorStats validationErrorStats,
        ErrorStats testErrorStats,
        BasicStats trainingBasicStats,
        BasicStats validationBasicStats,
        BasicStats testBasicStats)
    {
        var fitType = DetermineFitType(trainingErrorStats, validationErrorStats, testErrorStats);
        var confidenceLevel = CalculateConfidenceLevel(trainingErrorStats, validationErrorStats, testErrorStats);
        var recommendations = GenerateRecommendations(fitType, trainingErrorStats, validationErrorStats, testErrorStats);

        return new FitDetectorResult
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    private FitType DetermineFitType(ErrorStats training, ErrorStats validation, ErrorStats test)
    {
        // Implement logic to determine fit type based on various metrics
        // This is a simplified example, you should expand on this
        if (training.R2 > 0.9 && validation.R2 > 0.9 && test.R2 > 0.9)
            return FitType.Good;
        if (training.R2 > 0.9 && validation.R2 < 0.7)
            return FitType.Overfit;
        if (training.R2 < 0.7 && validation.R2 < 0.7)
            return FitType.Underfit;
        if (Math.Abs(training.R2 - validation.R2) > 0.2)
            return FitType.HighVariance;
        if (training.R2 < 0.5 && validation.R2 < 0.5 && test.R2 < 0.5)
            return FitType.HighBias;
        
        return FitType.Unstable;
    }

    private double CalculateConfidenceLevel(ErrorStats training, ErrorStats validation, ErrorStats test)
    {
        // Implement logic to calculate confidence level
        // This is a placeholder implementation
        return (training.R2 + validation.R2 + test.R2) / 3;
    }

    private List<string> GenerateRecommendations(FitType fitType, ErrorStats training, ErrorStats validation, ErrorStats test)
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