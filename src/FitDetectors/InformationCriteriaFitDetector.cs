using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class InformationCriteriaFitDetector<T> : FitDetectorBase<T>
{
    private readonly InformationCriteriaFitDetectorOptions _options;

    public InformationCriteriaFitDetector(InformationCriteriaFitDetectorOptions? options = null)
    {
        _options = options ?? new InformationCriteriaFitDetectorOptions();
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
        var trainingAic = evaluationData.TrainingErrorStats.AIC;
        var validationAic = evaluationData.ValidationErrorStats.AIC;
        var testAic = evaluationData.TestErrorStats.AIC;

        var trainingBic = evaluationData.TrainingErrorStats.BIC;
        var validationBic = evaluationData.ValidationErrorStats.BIC;
        var testBic = evaluationData.TestErrorStats.BIC;

        var aicDiff = Convert.ToDouble(_numOps.Subtract(_numOps.GreaterThan(validationAic, testAic) ? validationAic : testAic, trainingAic));
        var bicDiff = Convert.ToDouble(_numOps.Subtract(_numOps.GreaterThan(validationBic, testBic) ? validationBic : testBic, trainingBic));

        if (aicDiff < _options.AicThreshold && bicDiff < _options.BicThreshold)
        {
            return FitType.GoodFit;
        }
        else if (aicDiff > _options.OverfitThreshold || bicDiff > _options.OverfitThreshold)
        {
            return FitType.Overfit;
        }
        else if (aicDiff < -_options.UnderfitThreshold || bicDiff < -_options.UnderfitThreshold)
        {
            return FitType.Underfit;
        }
        else if (Math.Abs(Convert.ToDouble(_numOps.Subtract(validationAic, testAic))) > _options.HighVarianceThreshold ||
                 Math.Abs(Convert.ToDouble(_numOps.Subtract(validationBic, testBic))) > _options.HighVarianceThreshold)
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var trainingAic = evaluationData.TrainingErrorStats.AIC;
        var validationAic = evaluationData.ValidationErrorStats.AIC;
        var testAic = evaluationData.TestErrorStats.AIC;

        var trainingBic = evaluationData.TrainingErrorStats.BIC;
        var validationBic = evaluationData.ValidationErrorStats.BIC;
        var testBic = evaluationData.TestErrorStats.BIC;

        var aicConfidence = Math.Exp(-(Convert.ToDouble(_numOps.Subtract(validationAic, trainingAic)) / 2)) * 
            Math.Exp(-(Convert.ToDouble(_numOps.Subtract(testAic, trainingAic))) / 2);
        var bicConfidence = Math.Exp(-(Convert.ToDouble(_numOps.Subtract(validationBic, trainingBic)) / 2)) * 
            Math.Exp(-(Convert.ToDouble(_numOps.Subtract(testBic, trainingBic))) / 2);

        var averageConfidence = (aicConfidence + bicConfidence) / 2;

        return _numOps.FromDouble(averageConfidence);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on information criteria.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Information criteria suggest potential overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more training data");
                break;
            case FitType.Underfit:
                recommendations.Add("Information criteria indicate underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Information criteria show high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Using ensemble methods");
                recommendations.Add("- Applying cross-validation techniques");
                break;
            case FitType.Unstable:
                recommendations.Add("Information criteria indicate unstable performance. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"AIC threshold: {_options.AicThreshold}, BIC threshold: {_options.BicThreshold}");

        return recommendations;
    }
}