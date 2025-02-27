using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class PrecisionRecallCurveFitDetector<T> : FitDetectorBase<T>
{
    private readonly PrecisionRecallCurveFitDetectorOptions _options;
    private double Auc { get; set; }
    private double F1Score { get; set; }
    
    public PrecisionRecallCurveFitDetector(PrecisionRecallCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new PrecisionRecallCurveFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        (var auc, var f1Score) = StatisticsHelper<T>.CalculateAucF1Score(evaluationData);
        (Auc, F1Score) = (Convert.ToDouble(auc), Convert.ToDouble(f1Score));

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
                { "AUC", Auc },
                { "F1Score", F1Score }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        if (Auc > _options.AreaUnderCurveThreshold && F1Score > _options.F1ScoreThreshold)
        {
            return FitType.GoodFit;
        }
        else if (Auc < _options.AreaUnderCurveThreshold && F1Score < _options.F1ScoreThreshold)
        {
            return FitType.PoorFit;
        }
        else
        {
            return FitType.Moderate;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        // Calculate confidence level as a weighted average of AUC and F1 Score
        return _numOps.Add(_numOps.Multiply(_numOps.FromDouble(Auc), _numOps.FromDouble(_options.AucWeight)), _numOps.Multiply(_numOps.FromDouble(F1Score), _numOps.FromDouble(_options.F1ScoreWeight)));
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance based on the precision-recall curve analysis.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model shows moderate performance. There's room for improvement.");
                recommendations.Add("Try adjusting the classification threshold to optimize precision and recall trade-off.");
                recommendations.Add("Consider feature engineering or selection to improve model performance.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model's performance is suboptimal based on the precision-recall curve analysis.");
                recommendations.Add("Review your feature set and consider adding more relevant features.");
                recommendations.Add("Experiment with different algorithms or ensemble methods.");
                recommendations.Add("Check for class imbalance and consider using techniques like oversampling or undersampling.");
                break;
        }

        return recommendations;
    }
}