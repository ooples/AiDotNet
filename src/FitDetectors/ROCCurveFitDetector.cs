using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class ROCCurveFitDetector<T> : FitDetectorBase<T>
{
    private readonly ROCCurveFitDetectorOptions _options;
    private T Auc { get; set; }

    public ROCCurveFitDetector(ROCCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new ROCCurveFitDetectorOptions();
        Auc = _numOps.Zero;
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        var (fpr, tpr) = StatisticsHelper<T>.CalculateROCCurve(evaluationData.ModelStats.Actual, evaluationData.ModelStats.Predicted);
        Auc = StatisticsHelper<T>.CalculateAUC(fpr, tpr);

        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "AUC", Convert.ToDouble(Auc) },
                { "FPR", fpr },
                { "TPR", tpr }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        if (_numOps.GreaterThanOrEquals(Auc, _numOps.FromDouble(_options.GoodFitThreshold)))
            return FitType.GoodFit;
        else if (_numOps.GreaterThanOrEquals(Auc, _numOps.FromDouble(_options.ModerateFitThreshold)))
            return FitType.Moderate;
        else if (_numOps.GreaterThanOrEquals(Auc, _numOps.FromDouble(_options.PoorFitThreshold)))
            return FitType.PoorFit;
        else
            return FitType.VeryPoorFit;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        return _numOps.Multiply(Auc, _numOps.FromDouble(_options.ConfidenceScalingFactor));
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance. Consider fine-tuning for potential improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model performance is moderate. Consider feature engineering or trying different algorithms.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model performance is poor. Review feature selection, try different algorithms, or gather more data.");
                break;
            case FitType.VeryPoorFit:
                recommendations.Add("The model performance is very poor. Reassess the problem formulation, data quality, and chosen algorithm.");
                break;
        }

        if (_numOps.LessThan(Auc, _numOps.FromDouble(_options.BalancedDatasetThreshold)))
        {
            recommendations.Add("The dataset might be imbalanced. Consider using balanced accuracy, F1 score, or other metrics suitable for imbalanced data.");
        }

        return recommendations;
    }
}