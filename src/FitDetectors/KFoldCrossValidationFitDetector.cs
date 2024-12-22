using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class KFoldCrossValidationFitDetector<T> : FitDetectorBase<T>
{
    private readonly KFoldCrossValidationFitDetectorOptions _options;

    public KFoldCrossValidationFitDetector(KFoldCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new KFoldCrossValidationFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);

        var confidenceLevel = CalculateConfidenceLevel(evaluationData);

        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var avgTrainingR2 = evaluationData.TrainingPredictionStats.R2;
        var avgValidationR2 = evaluationData.ValidationPredictionStats.R2;
        var testR2 = evaluationData.TestPredictionStats.R2;

        var r2Difference = _numOps.Subtract(avgTrainingR2, avgValidationR2);
        var testDifference = _numOps.Subtract(avgValidationR2, testR2);

        if (_numOps.GreaterThan(r2Difference, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(avgValidationR2, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(_numOps.Abs(testDifference), _numOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (_numOps.GreaterThan(avgValidationR2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
                 _numOps.LessThan(_numOps.Abs(testDifference), _numOps.FromDouble(_options.StabilityThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var avgValidationR2 = evaluationData.ValidationPredictionStats.R2;
        var testR2 = evaluationData.TestPredictionStats.R2;

        var r2Difference = _numOps.Abs(_numOps.Subtract(avgValidationR2, testR2));
        var maxR2 = _numOps.GreaterThan(avgValidationR2, testR2) ? avgValidationR2 : testR2;

        var confidence = _numOps.Subtract(_numOps.One, _numOps.Divide(r2Difference, maxR2));
        return _numOps.GreaterThan(confidence, _numOps.Zero) ? confidence : _numOps.Zero;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on K-Fold Cross-Validation.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more training data");
                break;
            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("The model shows high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Using ensemble methods");
                recommendations.Add("- Applying feature selection techniques");
                break;
            case FitType.Unstable:
                recommendations.Add("The model performance is unstable. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust cross-validation techniques");
                break;
        }

        recommendations.Add($"Average Validation R2: {evaluationData.ValidationPredictionStats.R2:F4}, Test R2: {evaluationData.TestPredictionStats.R2:F4}");

        return recommendations;
    }
}