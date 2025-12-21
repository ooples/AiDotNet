namespace AiDotNet.FitDetectors;

/// <summary>
/// A specialized detector that evaluates how well a model fits time series data using cross-validation techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you understand if your time series model is learning correctly from your data.
/// 
/// Time series data is information collected over time in sequence (like daily temperatures, monthly sales, 
/// or hourly website traffic). When working with time series data, we need special techniques because:
/// 
/// 1. The order of data matters (unlike regular data where order doesn't matter)
/// 2. Recent data is often more important than older data
/// 3. There might be patterns that repeat over time (like seasonal patterns)
/// 
/// This detector analyzes your model's performance and tells you if it's:
/// - Learning too much detail from your data (overfitting)
/// - Not learning enough patterns (underfitting)
/// - Performing inconsistently (high variance)
/// - Working well (good fit)
/// - Behaving unpredictably (unstable)
/// 
/// It also provides specific recommendations to improve your model based on these findings.
/// </para>
/// </remarks>
public class TimeSeriesCrossValidationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the time series cross-validation fit detector.
    /// </summary>
    private readonly TimeSeriesCrossValidationFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the TimeSeriesCrossValidationFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector object. You can customize how
    /// the detector works by providing options, or use the default settings if you're just starting out.
    /// </para>
    /// </remarks>
    public TimeSeriesCrossValidationFitDetector(TimeSeriesCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new TimeSeriesCrossValidationFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data and determines how well the model fits the time series data.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics on training, validation, and test datasets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations for improvement.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model's performance and provides a diagnosis.
    /// 
    /// It works in three steps:
    /// 1. It determines what type of fit your model has (good, overfit, underfit, etc.)
    /// 2. It calculates how confident it is in this diagnosis
    /// 3. It generates specific recommendations to help you improve your model
    /// 
    /// The result combines all this information into a single package that helps you understand
    /// your model's performance and what steps to take next.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

    /// <summary>
    /// Determines the type of fit for the model based on error metrics and thresholds.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>The detected fit type (GoodFit, Overfit, Underfit, HighVariance, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method diagnoses your model's "learning health" by comparing error measurements.
    /// 
    /// It looks at how your model performs on different datasets:
    /// - Training data: The data your model learned from
    /// - Validation data: Data used to fine-tune your model
    /// - Test data: New data your model hasn't seen before
    /// 
    /// The method uses two main metrics:
    /// - RMSE (Root Mean Square Error): Measures prediction errors (lower is better)
    /// - R² (R-squared): Measures how well your model explains the data (higher is better)
    /// 
    /// By comparing these metrics across different datasets, it can detect:
    /// - Overfitting: When your model performs much better on training data than new data
    /// - Underfitting: When your model performs poorly on all datasets
    /// - High Variance: When your model's performance varies significantly between datasets
    /// - Good Fit: When your model performs consistently well across all datasets
    /// - Unstable: When your model doesn't fit into the other categories
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingRMSE = evaluationData.TrainingSet.ErrorStats.RMSE;
        var validationRMSE = evaluationData.ValidationSet.ErrorStats.RMSE;
        var testRMSE = evaluationData.TestSet.ErrorStats.RMSE;

        var rmseRatio = NumOps.Divide(validationRMSE, trainingRMSE);
        var testTrainingRatio = NumOps.Divide(testRMSE, trainingRMSE);

        if (NumOps.GreaterThan(rmseRatio, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(trainingRMSE, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(validationRMSE, NumOps.FromDouble(_options.UnderfitThreshold)) &&
                 NumOps.LessThan(testRMSE, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(testTrainingRatio, NumOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (NumOps.GreaterThan(evaluationData.TrainingSet.PredictionStats.R2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
                 NumOps.GreaterThan(evaluationData.ValidationSet.PredictionStats.R2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
                 NumOps.GreaterThan(evaluationData.TestSet.PredictionStats.R2, NumOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates how confident the detector is in its fit type determination.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>A confidence score between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how sure the detector is about its diagnosis of your model.
    /// 
    /// Think of it like a doctor's confidence in their diagnosis. The confidence level ranges from 0 to 1:
    /// - 1.0 means "completely confident" 
    /// - 0.0 means "not confident at all"
    /// 
    /// The confidence is calculated by looking at how consistent your model's performance is between
    /// the validation and test datasets. If the performance metrics (RMSE and R²) are very similar
    /// between these datasets, the confidence will be high. If they're very different, the confidence
    /// will be low.
    /// 
    /// Low confidence might indicate that your data has unusual patterns or that your model is
    /// sensitive to small changes in the data.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var rmseStability = NumOps.Divide(
            NumOps.Abs(NumOps.Subtract(evaluationData.TestSet.ErrorStats.RMSE, evaluationData.ValidationSet.ErrorStats.RMSE)),
            evaluationData.ValidationSet.ErrorStats.RMSE
        );

        var r2Stability = NumOps.Divide(
            NumOps.Abs(NumOps.Subtract(evaluationData.TestSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2)),
            evaluationData.ValidationSet.PredictionStats.R2
        );

        var stabilityScore = NumOps.Subtract(NumOps.One, NumOps.Add(rmseStability, r2Stability));
        var lessThan = NumOps.LessThan(NumOps.One, stabilityScore) ? NumOps.One : stabilityScore;
        return NumOps.GreaterThan(NumOps.Zero, lessThan) ? NumOps.Zero : lessThan;
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The type of fit detected for the model (e.g., Overfit, Underfit, GoodFit).</param>
    /// <param name="evaluationData">Data containing the model's performance metrics across different datasets.</param>
    /// <returns>A list of string recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a personalized "improvement plan" for your model based on how it's performing.
    /// 
    /// After analyzing your model, this method provides specific advice on how to make it better. The recommendations
    /// are tailored to the specific issues detected:
    /// 
    /// - Overfitting: When your model has "memorized" the training data instead of learning general patterns.
    ///   Think of it like a student who memorizes test answers without understanding the concepts.
    ///   
    /// - Underfitting: When your model is too simple to capture the patterns in your data.
    ///   Imagine trying to draw a complex curve using only straight lines - it won't fit well.
    ///   
    /// - High Variance: When your model's performance changes dramatically with different data.
    ///   Like a weather forecast that's extremely accurate one day but completely wrong the next.
    ///   
    /// - Good Fit: When your model has learned the right patterns and performs well consistently.
    ///   This is the goal! But there are still ways to make it even better.
    ///   
    /// - Unstable: When your model's behavior doesn't fit neatly into the other categories.
    ///   Often happens with time series data that has changing patterns over time.
    ///   
    /// The method also includes key performance metrics to help you track your progress:
    /// 
    /// - RMSE (Root Mean Square Error): Measures how far your predictions are from the actual values.
    ///   Lower values are better. Think of it as the average "distance" between predictions and reality.
    ///   
    /// - R² (R-squared): Measures how well your model explains the variation in the data.
    ///   Values closer to 1 are better. Think of it as a percentage of how much your model "understands" the data.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

        recommendations.Add($"Training RMSE: {evaluationData.TrainingSet.ErrorStats.RMSE:F4}");
        recommendations.Add($"Validation RMSE: {evaluationData.ValidationSet.ErrorStats.RMSE:F4}");
        recommendations.Add($"Test RMSE: {evaluationData.TestSet.ErrorStats.RMSE:F4}");
        recommendations.Add($"Training R2: {evaluationData.TrainingSet.PredictionStats.R2:F4}");
        recommendations.Add($"Validation R2: {evaluationData.ValidationSet.PredictionStats.R2:F4}");
        recommendations.Add($"Test R2: {evaluationData.TestSet.PredictionStats.R2:F4}");

        return recommendations;
    }
}
