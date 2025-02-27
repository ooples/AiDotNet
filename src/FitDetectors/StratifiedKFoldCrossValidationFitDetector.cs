using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class StratifiedKFoldCrossValidationFitDetector<T> : FitDetectorBase<T>
{
    private readonly StratifiedKFoldCrossValidationFitDetectorOptions _options;

    public StratifiedKFoldCrossValidationFitDetector(StratifiedKFoldCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new StratifiedKFoldCrossValidationFitDetectorOptions();
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
        var avgTrainingMetric = evaluationData.TrainingSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var avgValidationMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var testMetric = evaluationData.TestSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        var metricDifference = _numOps.Subtract(avgTrainingMetric, avgValidationMetric);
        var testDifference = _numOps.Subtract(avgValidationMetric, testMetric);

        if (_numOps.GreaterThan(metricDifference, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(avgValidationMetric, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(_numOps.Abs(testDifference), _numOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (_numOps.GreaterThan(avgValidationMetric, _numOps.FromDouble(_options.GoodFitThreshold)) &&
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
        var avgValidationMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var testMetric = evaluationData.TestSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        var metricDifference = _numOps.Abs(_numOps.Subtract(avgValidationMetric, testMetric));
        var maxMetric = _numOps.GreaterThan(avgValidationMetric, testMetric) ? avgValidationMetric : testMetric;

        var confidence = _numOps.Subtract(_numOps.One, _numOps.Divide(metricDifference, maxMetric));
        return _numOps.GreaterThan(confidence, _numOps.Zero) ? confidence : _numOps.Zero;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on Stratified K-Fold Cross-Validation.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more diverse training data");
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
                recommendations.Add("- Investigating data quality and class distribution");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust cross-validation techniques");
                break;
        }

        var primaryMetric = _options.PrimaryMetric;
        recommendations.Add($"Average Validation {primaryMetric}: {evaluationData.ValidationSet.PredictionStats.GetMetric(primaryMetric):F4}, Test {primaryMetric}: {evaluationData.TestSet.PredictionStats.GetMetric(primaryMetric):F4}");

        return recommendations;
    }
}