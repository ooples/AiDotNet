namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit using K-Fold Cross-Validation technique.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine if your machine learning model is a good fit for your data.
/// It uses a technique called "K-Fold Cross-Validation" which:
/// 1. Splits your data into K equal parts (or "folds")
/// 2. Trains the model K times, each time using a different fold as a validation set
/// 3. Analyzes how consistently your model performs across these different splits
/// 
/// This approach helps identify common problems like:
/// - Overfitting: When your model performs great on training data but poorly on new data (it "memorized" instead of "learned")
/// - Underfitting: When your model is too simple to capture important patterns in your data
/// - High Variance: When your model's performance changes dramatically with different data splits
/// - Instability: When your model doesn't consistently perform well across different data arrangements
/// </para>
/// </remarks>
public class KFoldCrossValidationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the K-Fold Cross-Validation fit detector.
    /// </summary>
    private readonly KFoldCrossValidationFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the K-Fold Cross-Validation fit detector.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector object. You can customize how it works by providing options,
    /// or just use the default settings if you're not sure what to change.
    /// </para>
    /// </remarks>
    public KFoldCrossValidationFitDetector(KFoldCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new KFoldCrossValidationFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data to detect the type of fit and provide recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics from training, validation, and test sets.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to check if your model is working well.
    /// It examines how your model performed on different data sets and tells you:
    /// 1. What type of fit your model has (good, overfit, underfit, etc.)
    /// 2. How confident the detector is about this assessment
    /// 3. Specific recommendations to improve your model
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
    /// Determines the type of fit based on model performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The detected fit type (Overfit, Underfit, GoodFit, HighVariance, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at how your model performed on different data sets and determines if there are any problems.
    /// 
    /// It compares the R² (R-squared) values, which measure how well your model explains the data (higher is better, with 1.0 being perfect):
    /// - If training R² is much higher than validation R², your model might be overfitting (memorizing instead of learning)
    /// - If validation R² is very low, your model might be underfitting (too simple to capture patterns)
    /// - If validation and test R² values differ significantly, your model might have high variance (inconsistent performance)
    /// - If validation R² is high and close to test R², your model likely has a good fit
    /// - Otherwise, your model might be unstable
    /// 
    /// Terminology explained:
    /// - R² (R-squared): A statistical measure between 0 and 1 that represents how well your model explains the data
    /// - Training set: Data used to train the model
    /// - Validation set: Data used to tune the model during development
    /// - Test set: Completely new data used to evaluate the final model
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var avgTrainingR2 = evaluationData.TrainingSet.PredictionStats.R2;
        var avgValidationR2 = evaluationData.ValidationSet.PredictionStats.R2;
        var testR2 = evaluationData.TestSet.PredictionStats.R2;

        var r2Difference = NumOps.Subtract(avgTrainingR2, avgValidationR2);
        var testDifference = NumOps.Subtract(avgValidationR2, testR2);

        if (NumOps.GreaterThan(r2Difference, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(avgValidationR2, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(NumOps.Abs(testDifference), NumOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (NumOps.GreaterThan(avgValidationR2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
                 NumOps.LessThan(NumOps.Abs(testDifference), NumOps.FromDouble(_options.StabilityThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A value between 0 and 1 representing the confidence level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident we are in our assessment of your model.
    /// 
    /// The confidence is based on how consistent your model performs between validation and test data:
    /// - If validation and test R² values are very close, confidence will be high (closer to 1)
    /// - If validation and test R² values differ significantly, confidence will be lower
    /// 
    /// A higher confidence value means you can trust the fit assessment more.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var avgValidationR2 = evaluationData.ValidationSet.PredictionStats.R2;
        var testR2 = evaluationData.TestSet.PredictionStats.R2;

        var r2Difference = NumOps.Abs(NumOps.Subtract(avgValidationR2, testR2));
        var maxR2 = NumOps.GreaterThan(avgValidationR2, testR2) ? avgValidationR2 : testR2;

        var confidence = NumOps.Subtract(NumOps.One, NumOps.Divide(r2Difference, maxR2));
        return NumOps.GreaterThan(confidence, NumOps.Zero) ? confidence : NumOps.Zero;
    }

    /// <summary>
    /// Generates specific recommendations based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on the detected issues.
    /// 
    /// Different recommendations are given depending on the fit type:
    /// - Good Fit: Congratulations! Your model is working well, with suggestions for deployment or fine-tuning
    /// - Overfit: Your model is "memorizing" the training data; suggestions to make it more generalized
    /// - Underfit: Your model is too simple; suggestions to make it more powerful
    /// - High Variance: Your model's performance varies too much; suggestions to make it more stable
    /// - Unstable: Your model's performance is inconsistent; suggestions to improve reliability
    /// 
    /// Terminology explained:
    /// - Regularization: A technique that prevents models from becoming too complex by adding penalties
    /// - Model complexity: How sophisticated your model is (e.g., number of parameters, depth of decision trees)
    /// - Ensemble methods: Combining multiple models to improve overall performance
    /// - Feature selection: Choosing which input variables to include in your model
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on K-Fold Cross-Validation.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more training data");
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
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust cross-validation techniques");
                break;
        }

        recommendations.Add($"Average Validation R2: {evaluationData.ValidationSet.PredictionStats.R2:F4}, Test R2: {evaluationData.TestSet.PredictionStats.R2:F4}");

        return recommendations;
    }
}
