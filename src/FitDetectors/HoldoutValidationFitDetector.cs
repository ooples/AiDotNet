namespace AiDotNet.FitDetectors;

public class HoldoutValidationFitDetector<T> : FitDetectorBase<T>
{
    private readonly HoldoutValidationFitDetectorOptions _options;

    public HoldoutValidationFitDetector(HoldoutValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new HoldoutValidationFitDetectorOptions();
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
        var trainingMSE = evaluationData.TrainingErrorStats.MSE;
        var validationMSE = evaluationData.ValidationErrorStats.MSE;
        var testMSE = evaluationData.TestErrorStats.MSE;

        var trainingR2 = evaluationData.TrainingPredictionStats.R2;
        var validationR2 = evaluationData.ValidationPredictionStats.R2;
        var testR2 = evaluationData.TestPredictionStats.R2;

        if (_numOps.GreaterThan(_numOps.Subtract(trainingR2, validationR2), _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(validationR2, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(validationMSE, testMSE)), _numOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (_numOps.GreaterThan(validationR2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
                 _numOps.LessThan(_numOps.Abs(_numOps.Subtract(validationR2, testR2)), _numOps.FromDouble(_options.StabilityThreshold)))
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
        var validationR2 = evaluationData.ValidationPredictionStats.R2;
        var testR2 = evaluationData.TestPredictionStats.R2;

        var r2Difference = _numOps.Abs(_numOps.Subtract(validationR2, testR2));
        var maxR2 = _numOps.GreaterThan(validationR2, testR2) ? validationR2 : testR2;

        var confidence = _numOps.Subtract(_numOps.One, _numOps.Divide(r2Difference, maxR2));
        return _numOps.GreaterThan(confidence, _numOps.Zero) ? confidence : _numOps.Zero;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on holdout validation.");
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
                recommendations.Add("- Applying cross-validation techniques");
                break;
            case FitType.Unstable:
                recommendations.Add("The model performance is unstable. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Validation R2: {evaluationData.ValidationPredictionStats.R2:F4}, Test R2: {evaluationData.TestPredictionStats.R2:F4}");

        return recommendations;
    }
}