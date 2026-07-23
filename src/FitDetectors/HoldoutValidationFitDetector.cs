namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality using holdout validation techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you determine if your machine learning model is performing well
/// by comparing how it performs on different subsets of your data:
/// - Training data: The data used to build the model
/// - Validation data: A separate set of data used to tune the model
/// - Test data: A final set of data used to evaluate the model's performance
/// 
/// By comparing performance across these sets, the detector can identify common problems like:
/// - Overfitting: When your model performs very well on training data but poorly on new data
/// - Underfitting: When your model performs poorly on all data sets
/// - High variance: When your model's performance varies significantly between different data sets
/// </para>
/// </remarks>
public class HoldoutValidationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the holdout validation fit detector.
    /// </summary>
    private readonly HoldoutValidationFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="HoldoutValidationFitDetector{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If null, default settings are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new detector with either custom settings you provide
    /// or default settings if you don't specify any. These settings control thresholds for determining
    /// if your model is overfitting, underfitting, etc.
    /// </para>
    /// </remarks>
    public HoldoutValidationFitDetector(HoldoutValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new HoldoutValidationFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model performance data to detect the quality of fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to evaluate your model. It:
    /// 1. Determines what type of fit your model has (good, overfit, underfit, etc.)
    /// 2. Calculates how confident it is in this assessment
    /// 3. Generates practical recommendations to improve your model
    /// 
    /// The result gives you actionable insights about your model's performance.
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
    /// <returns>The classified fit type of the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines how your model performs on different data sets and
    /// classifies it into one of several categories:
    /// 
    /// - Overfit: Your model has "memorized" the training data rather than learning general patterns.
    ///   It performs much better on training data than on new data.
    ///   
    /// - Underfit: Your model is too simple to capture the patterns in your data.
    ///   It performs poorly on all data sets.
    ///   
    /// - High Variance: Your model's performance varies significantly between different data sets,
    ///   suggesting it's sensitive to which data points it sees.
    ///   
    /// - Good Fit: Your model generalizes well, performing consistently across different data sets.
    ///   
    /// - Unstable: Your model doesn't fit into the other categories but has performance issues.
    /// 
    /// The method uses thresholds (set in the options) to make these determinations.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingMSE = evaluationData.TrainingSet.ErrorStats.MSE;
        var validationMSE = evaluationData.ValidationSet.ErrorStats.MSE;
        var testMSE = evaluationData.TestSet.ErrorStats.MSE;

        var trainingR2 = evaluationData.TrainingSet.PredictionStats.R2;
        var validationR2 = evaluationData.ValidationSet.PredictionStats.R2;
        var testR2 = evaluationData.TestSet.PredictionStats.R2;

        if (NumOps.GreaterThan(NumOps.Subtract(trainingR2, validationR2), NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(validationR2, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(validationMSE, testMSE)), NumOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (NumOps.GreaterThan(validationR2, NumOps.FromDouble(_options.GoodFitThreshold)) &&
                 NumOps.LessThan(NumOps.Abs(NumOps.Subtract(validationR2, testR2)), NumOps.FromDouble(_options.StabilityThreshold)))
        {
            return FitType.GoodFit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level in the fit type determination.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A value between 0 and 1 representing the confidence level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident the detector is in its assessment of your model.
    /// The confidence is based on how consistent your model's performance is between the validation
    /// and test sets.
    /// 
    /// If your model performs very similarly on both sets (measured by R² values), the confidence will be high.
    /// If there's a big difference in performance between these sets, the confidence will be lower.
    /// 
    /// The confidence value ranges from 0 to 1:
    /// - 1.0 means complete confidence (identical performance on validation and test sets)
    /// - 0.0 means no confidence (completely different performance)
    /// 
    /// R² (R-squared) is a statistical measure that represents how well your model's predictions match
    /// the actual values. It ranges from 0 to 1, where 1 means perfect predictions.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var validationR2 = evaluationData.ValidationSet.PredictionStats.R2;
        var testR2 = evaluationData.TestSet.PredictionStats.R2;

        var r2Difference = NumOps.Abs(NumOps.Subtract(validationR2, testR2));
        var maxR2 = NumOps.GreaterThan(validationR2, testR2) ? validationR2 : testR2;

        var confidence = NumOps.Subtract(NumOps.One, NumOps.Divide(r2Difference, maxR2));
        return NumOps.GreaterThan(confidence, NumOps.Zero) ? confidence : NumOps.Zero;
    }

    /// <summary>
    /// Generates practical recommendations for improving the model based on its fit type.
    /// </summary>
    /// <param name="fitType">The determined fit type of the model.</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of string recommendations for model improvement.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of practical suggestions to help you improve your model
    /// based on how it's currently performing. The recommendations are tailored to the specific issues
    /// detected in your model:
    /// 
    /// - Good Fit: If your model is performing well, you'll get suggestions for deployment or fine-tuning.
    /// 
    /// - Overfitting: If your model is "memorizing" the training data instead of learning general patterns,
    ///   you'll get suggestions like:
    ///   * Increasing regularization (adding penalties to prevent the model from becoming too complex)
    ///   * Reducing model complexity (using a simpler model with fewer parameters)
    ///   * Collecting more training data (to help the model learn more general patterns)
    /// 
    /// - Underfitting: If your model is too simple to capture the patterns in your data,
    ///   you'll get suggestions like:
    ///   * Increasing model complexity (using a more sophisticated model)
    ///   * Adding more relevant features (giving the model more useful information)
    ///   * Reducing regularization (allowing the model to become more complex)
    /// 
    /// - High Variance: If your model's performance varies significantly between different data sets,
    ///   you'll get suggestions like:
    ///   * Increasing training data (to provide more examples for learning)
    ///   * Using ensemble methods (combining multiple models to improve stability)
    ///   * Applying cross-validation (testing the model on multiple data subsets)
    /// 
    /// - Unstable: If your model has inconsistent performance,
    ///   you'll get suggestions like:
    ///   * Investigating data quality (checking for errors or inconsistencies)
    ///   * Trying different model architectures (testing alternative model designs)
    ///   * Using better feature selection (choosing more relevant input variables)
    /// 
    /// The method also includes the R² values for validation and test sets to help you
    /// understand the model's current performance. R² (R-squared) is a measure of how well
    /// your model's predictions match the actual values, with values closer to 1.0 indicating
    /// better performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on holdout validation.");
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
                recommendations.Add("- Applying cross-validation techniques");
                break;
            case FitType.Unstable:
                recommendations.Add("The model performance is unstable. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Validation R2: {evaluationData.ValidationSet.PredictionStats.R2:F4}, Test R2: {evaluationData.TestSet.PredictionStats.R2:F4}");

        return recommendations;
    }
}
