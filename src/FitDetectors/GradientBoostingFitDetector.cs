namespace AiDotNet.FitDetectors;

public class GradientBoostingFitDetector<T> : FitDetectorBase<T>
{
    private readonly GradientBoostingFitDetectorOptions _options;

    public GradientBoostingFitDetector(GradientBoostingFitDetectorOptions? options = null)
    {
        _options = options ?? new GradientBoostingFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

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
                { "PerformanceMetrics", GetPerformanceMetrics(evaluationData) }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var trainError = evaluationData.TrainingSet.ErrorStats.MSE;
        var validationError = evaluationData.ValidationSet.ErrorStats.MSE;
        var errorDifference = _numOps.Subtract(validationError, trainError);

        if (_numOps.LessThan(errorDifference, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return _numOps.LessThan(validationError, _numOps.FromDouble(_options.GoodFitThreshold)) 
                ? FitType.GoodFit 
                : FitType.Moderate;
        }
        else
        {
            return _numOps.GreaterThan(errorDifference, _numOps.FromDouble(_options.SevereOverfitThreshold)) 
                ? FitType.VeryPoorFit 
                : FitType.PoorFit;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var trainError = evaluationData.TrainingSet.ErrorStats.MSE;
        var validationError = evaluationData.ValidationSet.ErrorStats.MSE;
        var errorDifference = _numOps.Subtract(validationError, trainError);

        // Calculate confidence based on how close the validation error is to the train error
        var relativeErrorDifference = _numOps.Divide(errorDifference, trainError);

        // Use an exponential decay function to map the relative error difference to a confidence level
        var confidence = _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-5), relativeErrorDifference));

        return MathHelper.Clamp(confidence, _numOps.Zero, _numOps.One);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows a good fit. Consider fine-tuning hyperparameters for potential improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model performance is moderate. Try adjusting the learning rate or increasing the number of estimators.");
                recommendations.Add("Consider feature engineering or selection to improve model performance.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model shows signs of overfitting. Implement regularization techniques like increasing min_samples_leaf or reducing max_depth.");
                recommendations.Add("Try using early stopping to prevent overfitting.");
                break;
            case FitType.VeryPoorFit:
                recommendations.Add("The model is severely overfitting. Drastically reduce model complexity by limiting max_depth and increasing min_samples_split.");
                recommendations.Add("Consider using a simpler model or gathering more training data.");
                break;
        }

        if (_numOps.LessThan(evaluationData.TrainingSet.ErrorStats.MSE, _numOps.FromDouble(0.01)))
        {
            recommendations.Add("The training error is suspiciously low. Verify that there's no data leakage in your preprocessing pipeline.");
        }

        return recommendations;
    }

    private Dictionary<string, T> GetPerformanceMetrics(ModelEvaluationData<T> evaluationData)
    {
        return new Dictionary<string, T>
        {
            { "TrainingMSE", evaluationData.TrainingSet.ErrorStats.MSE },
            { "ValidationMSE", evaluationData.ValidationSet.ErrorStats.MSE },
            { "TrainingR2", evaluationData.TrainingSet.PredictionStats.R2 },
            { "ValidationR2", evaluationData.ValidationSet.PredictionStats.R2 }
        };
    }
}