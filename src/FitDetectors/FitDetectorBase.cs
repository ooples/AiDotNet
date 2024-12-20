namespace AiDotNet.FitDetectors;

public abstract class FitDetectorBase<T> : IFitDetector<T>
{
    protected readonly INumericOperations<T> _numOps;

    protected FitDetectorBase()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public abstract FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData);

    protected abstract FitType DetermineFitType(ModelEvaluationData<T> evaluationData);

    protected abstract T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData);

    protected virtual List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Good:
                recommendations.Add("The model appears to be well-fitted. Consider deploying it and monitoring its performance on new data.");
                break;
            case FitType.HighVariance:
                recommendations.Add("The model shows high variance. Consider simplifying the model, using regularization techniques, or gathering more diverse training data.");
                break;
            case FitType.HighBias:
                recommendations.Add("The model shows high bias. Consider increasing model complexity, adding more relevant features, or using a different algorithm.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider using regularization, reducing model complexity, or gathering more training data.");
                break;
            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider increasing model complexity, reducing regularization, or adding more relevant features.");
                break;
            case FitType.Unstable:
                recommendations.Add("The model's performance is unstable across datasets. Consider using more robust feature selection, cross-validation techniques, or ensemble methods.");
                break;
        }

        return recommendations;
    }
}