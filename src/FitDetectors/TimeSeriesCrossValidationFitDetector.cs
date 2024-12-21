namespace AiDotNet.FitDetectors;

public class TimeSeriesCrossValidationFitDetector<T> : FitDetectorBase<T>
{
    private readonly TimeSeriesCrossValidationFitDetectorOptions _options;

    public TimeSeriesCrossValidationFitDetector(TimeSeriesCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new TimeSeriesCrossValidationFitDetectorOptions();
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
        var trainingRMSE = evaluationData.TrainingErrorStats.RMSE;
        var validationRMSE = evaluationData.ValidationErrorStats.RMSE;
        var testRMSE = evaluationData.TestErrorStats.RMSE;

        var rmseRatio = _numOps.Divide(validationRMSE, trainingRMSE);
        var testTrainingRatio = _numOps.Divide(testRMSE, trainingRMSE);

        if (_numOps.GreaterThan(rmseRatio, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(trainingRMSE, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(validationRMSE, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(testRMSE, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(testTrainingRatio, _numOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (_numOps.GreaterThan(evaluationData.TrainingPredictionStats.R2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
                 _numOps.GreaterThan(evaluationData.ValidationPredictionStats.R2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
                 _numOps.GreaterThan(evaluationData.TestPredictionStats.R2, _numOps.FromDouble(_options.GoodFitThreshold)))
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
        var rmseStability = _numOps.Divide(
            _numOps.Abs(_numOps.Subtract(evaluationData.TestErrorStats.RMSE, evaluationData.ValidationErrorStats.RMSE)),
            evaluationData.ValidationErrorStats.RMSE
        );

        var r2Stability = _numOps.Divide(
            _numOps.Abs(_numOps.Subtract(evaluationData.TestPredictionStats.R2, evaluationData.ValidationPredictionStats.R2)),
            evaluationData.ValidationPredictionStats.R2
        );

        var stabilityScore = _numOps.Subtract(_numOps.One, _numOps.Add(rmseStability, r2Stability));
        var lessThan = _numOps.LessThan(_numOps.One, stabilityScore) ? _numOps.One : stabilityScore;
        return _numOps.GreaterThan(_numOps.Zero, lessThan) ? _numOps.Zero : lessThan;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Model appears to be overfitting. Consider:");
                recommendations.Add("- Increasing the size of your training data");
                recommendations.Add("- Simplifying the model (reduce complexity/parameters)");
                recommendations.Add("- Adding regularization techniques");
                recommendations.Add("- Adjusting the rolling window size in cross-validation");
                break;
            case FitType.Underfit:
                recommendations.Add("Model appears to be underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                recommendations.Add("- Exploring non-linear relationships in the data");
                break;
            case FitType.HighVariance:
                recommendations.Add("Model shows high variance. Consider:");
                recommendations.Add("- Increasing the size of your training data");
                recommendations.Add("- Using ensemble methods (e.g., bagging, boosting)");
                recommendations.Add("- Applying cross-validation with more folds");
                recommendations.Add("- Investigating for potential concept drift in your time series");
                break;
            case FitType.GoodFit:
                recommendations.Add("Model shows good fit. Consider:");
                recommendations.Add("- Fine-tuning hyperparameters for potential improvements");
                recommendations.Add("- Monitoring model performance over time for potential degradation");
                recommendations.Add("- Exploring more advanced time series techniques for further improvements");
                break;
            case FitType.Unstable:
                recommendations.Add("Model performance is unstable. Consider:");
                recommendations.Add("- Investigating for non-stationarity in your time series");
                recommendations.Add("- Applying appropriate transformations (e.g., differencing, log transform)");
                recommendations.Add("- Using more robust error metrics for time series (e.g., MASE, MAPE)");
                recommendations.Add("- Exploring models that can handle regime changes or structural breaks");
                break;
        }

        recommendations.Add($"Training RMSE: {evaluationData.TrainingErrorStats.RMSE:F4}");
        recommendations.Add($"Validation RMSE: {evaluationData.ValidationErrorStats.RMSE:F4}");
        recommendations.Add($"Test RMSE: {evaluationData.TestErrorStats.RMSE:F4}");
        recommendations.Add($"Training R2: {evaluationData.TrainingPredictionStats.R2:F4}");
        recommendations.Add($"Validation R2: {evaluationData.ValidationPredictionStats.R2:F4}");
        recommendations.Add($"Test R2: {evaluationData.TestPredictionStats.R2:F4}");

        return recommendations;
    }
}