using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class VIFFitDetector<T> : FitDetectorBase<T>
{
    private readonly VIFFitDetectorOptions _options;
    private readonly ModelStatsOptions _modelStatsOptions;

    public VIFFitDetector(VIFFitDetectorOptions? options = null, ModelStatsOptions? modelStatsOptions = null)
    {
        _options = options ?? new VIFFitDetectorOptions();
        _modelStatsOptions = modelStatsOptions ?? new ModelStatsOptions();
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
        var vifValues = StatisticsHelper<T>.CalculateVIF(evaluationData.ModelStats.CorrelationMatrix, _modelStatsOptions);
        var maxVIF = vifValues.Max() ?? _numOps.Zero;

        if (_numOps.GreaterThan(maxVIF, _numOps.FromDouble(_options.SevereMulticollinearityThreshold)))
        {
            return FitType.SevereMulticollinearity;
        }
        else if (_numOps.GreaterThan(maxVIF, _numOps.FromDouble(_options.ModerateMulticollinearityThreshold)))
        {
            return FitType.ModerateMulticollinearity;
        }
        else
        {
            var primaryMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
            if (_numOps.GreaterThan(primaryMetric, _numOps.FromDouble(_options.GoodFitThreshold)))
            {
                return FitType.GoodFit;
            }
            else
            {
                return FitType.PoorFit;
            }
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var vifValues = StatisticsHelper<T>.CalculateVIF(evaluationData.ModelStats.CorrelationMatrix, _modelStatsOptions);
        var maxVIF = vifValues.Max() ?? _numOps.Zero;
        var avgVIF = _numOps.Divide(vifValues.Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(vifValues.Count));

        var vifConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(avgVIF, maxVIF));
        var metricConfidence = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        return _numOps.Multiply(vifConfidence, metricConfidence);
    }

    protected override List<string> GenerateRecommendations(
        FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.SevereMulticollinearity:
                recommendations.Add("Severe multicollinearity detected. Consider:");
                recommendations.Add("- Removing highly correlated features");
                recommendations.Add("- Using regularization techniques (e.g., Ridge, Lasso)");
                recommendations.Add("- Applying dimensionality reduction methods (e.g., PCA)");
                break;
            case FitType.ModerateMulticollinearity:
                recommendations.Add("Moderate multicollinearity detected. Consider:");
                recommendations.Add("- Investigating feature correlations");
                recommendations.Add("- Combining or removing some correlated features");
                recommendations.Add("- Using regularization techniques");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit with acceptable multicollinearity levels.");
                recommendations.Add("Consider fine-tuning for even better performance.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model shows poor fit. Consider:");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Collecting more training data");
                break;
        }

        var primaryMetric = _options.PrimaryMetric;
        recommendations.Add($"Validation {primaryMetric}: {evaluationData.ValidationSet.PredictionStats.GetMetric(primaryMetric):F4}, Test {primaryMetric}: {evaluationData.TestSet.PredictionStats.GetMetric(primaryMetric):F4}");

        return recommendations;
    }
}