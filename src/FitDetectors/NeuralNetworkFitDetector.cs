using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class NeuralNetworkFitDetector<T> : FitDetectorBase<T>
{
    private readonly NeuralNetworkFitDetectorOptions _options;
    private double TrainingLoss { get; set; }
    private double ValidationLoss { get; set; }
    private double TestLoss { get; set; }
    private double OverfittingScore { get; set; }

    public NeuralNetworkFitDetector(NeuralNetworkFitDetectorOptions? options = null)
    {
        _options = options ?? new NeuralNetworkFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        TrainingLoss = Convert.ToDouble(evaluationData.TrainingErrorStats.MSE);
        ValidationLoss = Convert.ToDouble(evaluationData.ValidationErrorStats.MSE);
        TestLoss = Convert.ToDouble(evaluationData.TestErrorStats.MSE);
        OverfittingScore = CalculateOverfittingScore(evaluationData);

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
                { "TrainingLoss", TrainingLoss },
                { "ValidationLoss", ValidationLoss },
                { "TestLoss", TestLoss },
                { "OverfittingScore", OverfittingScore }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        if (ValidationLoss <= _options.GoodFitThreshold && OverfittingScore <= _options.OverfittingThreshold)
            return FitType.GoodFit;
        else if (ValidationLoss <= _options.ModerateFitThreshold && OverfittingScore <= _options.OverfittingThreshold * 1.5)
            return FitType.Moderate;
        else if (ValidationLoss <= _options.PoorFitThreshold || OverfittingScore <= _options.OverfittingThreshold * 2)
            return FitType.PoorFit;
        else
            return FitType.VeryPoorFit;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var lossConfidence = Math.Max(0, 1 - (ValidationLoss / _options.PoorFitThreshold));
        var overfittingConfidence = Math.Max(0, 1 - (OverfittingScore / (_options.OverfittingThreshold * 2)));

        var overallConfidence = (lossConfidence + overfittingConfidence) / 2;
        return _numOps.FromDouble(overallConfidence);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        if (fitType == FitType.GoodFit)
        {
            recommendations.Add("The neural network shows good fit. Consider fine-tuning for potential improvements.");
        }
        else if (fitType == FitType.Moderate)
        {
            recommendations.Add("The neural network shows moderate performance. Consider the following:");
            if (OverfittingScore > _options.OverfittingThreshold)
                recommendations.Add("- Implement regularization techniques to reduce overfitting.");
            recommendations.Add("- Experiment with different network architectures or hyperparameters.");
        }
        else if (fitType == FitType.PoorFit || fitType == FitType.VeryPoorFit)
        {
            recommendations.Add("The neural network shows poor fit. Consider the following:");
            if (TrainingLoss > _options.PoorFitThreshold)
                recommendations.Add("- Increase model capacity by adding more layers or neurons.");
            if (OverfittingScore > _options.OverfittingThreshold * 1.5)
                recommendations.Add("- Implement strong regularization techniques (e.g., dropout, L1/L2 regularization).");
            recommendations.Add("- Review and preprocess the input data for potential issues.");
            recommendations.Add("- Consider using a different type of neural network architecture.");
        }

        return recommendations;
    }

    private double CalculateOverfittingScore(ModelEvaluationData<T> evaluationData)
    {
        return Math.Max(0, (ValidationLoss - TrainingLoss) / TrainingLoss);
    }
}