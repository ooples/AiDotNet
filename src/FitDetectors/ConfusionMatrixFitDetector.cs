namespace AiDotNet.FitDetectors;

public class ConfusionMatrixFitDetector<T> : FitDetectorBase<T>
{
    private readonly ConfusionMatrixFitDetectorOptions _options;

    public ConfusionMatrixFitDetector(ConfusionMatrixFitDetectorOptions options)
    {
        _options = options ?? new ConfusionMatrixFitDetectorOptions();
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
        var confusionMatrix = CalculateConfusionMatrix(evaluationData.ModelStats.Actual, evaluationData.ModelStats.Predicted);
        var metric = CalculatePrimaryMetric(confusionMatrix);

        if (_numOps.GreaterThanOrEquals(metric, _numOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (_numOps.GreaterThanOrEquals(metric, _numOps.FromDouble(_options.ModerateFitThreshold)))
        {
            return FitType.Moderate;
        }
        else
        {
            return FitType.PoorFit;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var confusionMatrix = CalculateConfusionMatrix(evaluationData.ModelStats.Actual, evaluationData.ModelStats.Predicted);
        var metric = CalculatePrimaryMetric(confusionMatrix);

        // Normalize the metric to a 0-1 range
        var normalizedMetric = _numOps.Divide(
            _numOps.Subtract(metric, _numOps.FromDouble(_options.ModerateFitThreshold)),
            _numOps.FromDouble(_options.GoodFitThreshold - _options.ModerateFitThreshold)
        );

        var lessThan = _numOps.LessThan(normalizedMetric, _numOps.One) ? normalizedMetric : _numOps.One;
        return _numOps.GreaterThan(lessThan, _numOps.Zero) ? lessThan : _numOps.Zero;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();
        var confusionMatrix = CalculateConfusionMatrix(evaluationData.ModelStats.Actual, evaluationData.ModelStats.Predicted);

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance based on the confusion matrix analysis.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model shows moderate performance. There's room for improvement.");
                recommendations.Add("Consider adjusting the classification threshold to optimize the trade-off between different types of errors.");
                recommendations.Add("Analyze feature importance and consider feature engineering or selection.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model's performance is suboptimal based on the confusion matrix analysis.");
                recommendations.Add("Review your feature set and consider adding more relevant features.");
                recommendations.Add("Experiment with different algorithms or ensemble methods.");
                break;
        }

        if (IsClassImbalanced(confusionMatrix))
        {
            recommendations.Add("The dataset appears to be imbalanced. Consider using techniques like oversampling, undersampling, or SMOTE to address this issue.");
        }

        return recommendations;
    }

    private ConfusionMatrix<T> CalculateConfusionMatrix(Vector<T> actual, Vector<T> predicted)
    {
        return StatisticsHelper<T>.CalculateConfusionMatrix(actual, predicted, _numOps.FromDouble(_options.ConfidenceThreshold));
    }

    private T CalculatePrimaryMetric(ConfusionMatrix<T> confusionMatrix)
    {
        return _options.PrimaryMetric switch
        {
            MetricType.Accuracy => confusionMatrix.Accuracy,
            MetricType.Precision => confusionMatrix.Precision,
            MetricType.Recall => confusionMatrix.Recall,
            MetricType.F1Score => confusionMatrix.F1Score,
            _ => throw new ArgumentException("Unsupported primary metric type."),
        };
    }

    private bool IsClassImbalanced(ConfusionMatrix<T> confusionMatrix)
    {
        T totalSamples = _numOps.Add(_numOps.Add(confusionMatrix.TruePositives, confusionMatrix.FalsePositives),
                                     _numOps.Add(confusionMatrix.TrueNegatives, confusionMatrix.FalseNegatives));
        T positiveRatio = _numOps.Divide(_numOps.Add(confusionMatrix.TruePositives, confusionMatrix.FalseNegatives), totalSamples);
        T negativeRatio = _numOps.Divide(_numOps.Add(confusionMatrix.TrueNegatives, confusionMatrix.FalsePositives), totalSamples);

        return _numOps.LessThan(positiveRatio, _numOps.FromDouble(_options.ClassImbalanceThreshold)) ||
               _numOps.LessThan(negativeRatio, _numOps.FromDouble(_options.ClassImbalanceThreshold));
    }
}