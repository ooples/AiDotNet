namespace AiDotNet.FitDetectors;

/// <summary>
/// A specialized detector that evaluates how well a gradient boosting model fits the data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gradient Boosting is a machine learning technique that builds multiple simple models 
/// (usually decision trees) sequentially, with each new model trying to correct the errors made by 
/// previous models. This detector helps you understand if your gradient boosting model is:
/// - Learning the data well (good fit)
/// - Not learning enough (underfit)
/// - Learning too much from the training data and not generalizing well (overfit)
/// </para>
/// </remarks>
public class GradientBoostingFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options that control how the detector evaluates model fit.
    /// </summary>
    private readonly GradientBoostingFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="GradientBoostingFitDetector{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you create this detector, you can customize how it works by providing options.
    /// If you don't provide any options, it will use reasonable default settings.
    /// </para>
    /// </remarks>
    public GradientBoostingFitDetector(GradientBoostingFitDetectorOptions? options = null)
    {
        _options = options ?? new GradientBoostingFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data to determine how well the model fits the data.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics on training and validation datasets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="evaluationData"/> is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines how your model performs on both training data (data it learned from) 
    /// and validation data (new data it hasn't seen before). It then tells you if your model is a good fit,
    /// provides a confidence score (how sure it is about its assessment), and gives you specific recommendations
    /// to improve your model.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

    /// <summary>
    /// Determines the type of fit based on the difference between training and validation errors.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A classification of the model fit quality (GoodFit, Moderate, PoorFit, or VeryPoorFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method compares how well your model performs on training data versus validation data.
    /// If the errors are similar, it's likely a good fit. If the model performs much better on training data
    /// than validation data, it may be overfitting (memorizing the training data rather than learning general patterns).
    /// </para>
    /// <para>
    /// - MSE stands for Mean Squared Error, which measures the average squared difference between predicted and actual values.
    /// - Lower MSE values indicate better performance.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainError = evaluationData.TrainingSet.ErrorStats.MSE;
        var validationError = evaluationData.ValidationSet.ErrorStats.MSE;
        var errorDifference = NumOps.Subtract(validationError, trainError);

        if (NumOps.LessThan(errorDifference, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return NumOps.LessThan(validationError, NumOps.FromDouble(_options.GoodFitThreshold))
                ? FitType.GoodFit
                : FitType.Moderate;
        }
        else
        {
            return NumOps.GreaterThan(errorDifference, NumOps.FromDouble(_options.SevereOverfitThreshold))
                ? FitType.VeryPoorFit
                : FitType.PoorFit;
        }
    }

    /// <summary>
    /// Calculates a confidence level for the fit assessment based on the relative difference between training and validation errors.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident the detector is about its assessment of your model.
    /// It looks at the difference between training and validation errors, relative to the training error.
    /// The confidence is higher when training and validation errors are close to each other.
    /// </para>
    /// <para>
    /// The exponential decay function used here means that as the difference between training and validation errors
    /// increases, the confidence decreases rapidly. This is because large differences suggest the model might be
    /// overfitting or have other issues.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainError = evaluationData.TrainingSet.ErrorStats.MSE;
        var validationError = evaluationData.ValidationSet.ErrorStats.MSE;
        var errorDifference = NumOps.Subtract(validationError, trainError);

        // Calculate confidence based on how close the validation error is to the train error
        var relativeErrorDifference = NumOps.Divide(errorDifference, trainError);

        // Use an exponential decay function to map the relative error difference to a confidence level
        var confidence = NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-5), relativeErrorDifference));

        return MathHelper.Clamp(confidence, NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The type of fit detected for the model.</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of recommendation strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on its current performance.
    /// Different recommendations are given depending on whether your model is:
    /// - Good fit: Already performing well, but might benefit from fine-tuning
    /// - Moderate fit: Performing okay, but could be improved with adjustments
    /// - Poor fit: Showing signs of overfitting (memorizing training data)
    /// - Very poor fit: Severely overfitting and needs significant changes
    /// </para>
    /// <para>
    /// Terms explained:
    /// - Learning rate: Controls how much each new tree contributes to the final prediction
    /// - Estimators: The number of trees in the gradient boosting model
    /// - min_samples_leaf: Minimum number of samples required in a leaf node
    /// - max_depth: Maximum depth of each tree
    /// - min_samples_split: Minimum number of samples required to split a node
    /// - Early stopping: Technique to stop training when performance stops improving
    /// - Data leakage: When information from validation data accidentally influences training
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

        if (NumOps.LessThan(evaluationData.TrainingSet.ErrorStats.MSE, NumOps.FromDouble(0.01)))
        {
            recommendations.Add("The training error is suspiciously low. Verify that there's no data leakage in your preprocessing pipeline.");
        }

        return recommendations;
    }

    /// <summary>
    /// Extracts key performance metrics from the evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A dictionary of performance metrics with their names as keys.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method collects important metrics that help you understand your model's performance:
    /// - MSE (Mean Squared Error): Measures prediction accuracy; lower values are better
    /// - R² (R-squared): Measures how well your model explains the variation in data; values closer to 1 are better
    /// </para>
    /// <para>
    /// Having both training and validation metrics helps you compare how well your model performs on data it has seen
    /// (training) versus data it hasn't seen (validation).
    /// </para>
    /// </remarks>
    private Dictionary<string, T> GetPerformanceMetrics(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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
