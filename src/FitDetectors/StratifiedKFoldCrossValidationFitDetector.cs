namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit using Stratified K-Fold Cross-Validation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you determine if your machine learning model is performing well
/// or if it has common problems like overfitting or underfitting.
/// 
/// Stratified K-Fold Cross-Validation is a technique that:
/// 1. Splits your data into K equal parts (folds)
/// 2. Makes sure each fold has a similar distribution of your target variable
/// 3. Trains K different models, each using K-1 folds for training and 1 fold for validation
/// 4. Averages the results to get a more reliable estimate of model performance
/// 
/// This approach helps ensure your model works well on different subsets of your data,
/// which is important for real-world applications.
/// </para>
/// </remarks>
public class StratifiedKFoldCrossValidationFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Stratified K-Fold Cross-Validation fit detector.
    /// </summary>
    private readonly StratifiedKFoldCrossValidationFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the StratifiedKFoldCrossValidationFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options for the detector. If null, default options will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new fit detector with either custom settings
    /// that you provide or default settings if you don't specify any.
    /// 
    /// Think of this as setting up a tool that will analyze your model's performance.
    /// You can either use the tool with its factory settings (default options) or
    /// customize it to your specific needs.
    /// </para>
    /// </remarks>
    public StratifiedKFoldCrossValidationFitDetector(StratifiedKFoldCrossValidationFitDetectorOptions? options = null)
    {
        _options = options ?? new StratifiedKFoldCrossValidationFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes the model's performance data and determines the type of fit.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is like a doctor examining test results to diagnose a patient.
    /// 
    /// It takes all the performance data from your model and:
    /// 1. Determines if your model has a good fit, is overfitting, underfitting, etc.
    /// 2. Calculates how confident it is in this diagnosis
    /// 3. Generates specific recommendations to improve your model
    /// 
    /// The result gives you both the diagnosis and a treatment plan for your model.
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
    /// Determines the type of fit based on the model's performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>The determined fit type (GoodFit, Overfit, Underfit, HighVariance, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance to identify common problems.
    /// 
    /// It compares how well your model performs on:
    /// - Training data (data it learned from)
    /// - Validation data (data used to tune the model)
    /// - Test data (completely new data)
    /// 
    /// Based on these comparisons, it diagnoses your model with one of these conditions:
    /// 
    /// - Overfit: Your model performs much better on training data than validation data.
    ///   This means it's memorizing the training data instead of learning general patterns.
    ///   
    /// - Underfit: Your model performs poorly even on validation data.
    ///   This means it's too simple to capture the patterns in your data.
    ///   
    /// - High Variance: Your model performs very differently on validation and test data.
    ///   This means it's sensitive to which data it sees.
    ///   
    /// - Good Fit: Your model performs well and consistently across different datasets.
    ///   This is the ideal outcome!
    ///   
    /// - Unstable: Your model doesn't fit any of the above patterns clearly.
    ///   This suggests there might be issues with your data or model setup.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var avgTrainingMetric = evaluationData.TrainingSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var avgValidationMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var testMetric = evaluationData.TestSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        var metricDifference = NumOps.Subtract(avgTrainingMetric, avgValidationMetric);
        var testDifference = NumOps.Subtract(avgValidationMetric, testMetric);

        if (NumOps.GreaterThan(metricDifference, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (NumOps.LessThan(avgValidationMetric, NumOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (NumOps.GreaterThan(NumOps.Abs(testDifference), NumOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.HighVariance;
        }
        else if (NumOps.GreaterThan(avgValidationMetric, NumOps.FromDouble(_options.GoodFitThreshold)) &&
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
    /// Calculates the confidence level in the fit type determination.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident we are in our diagnosis of the model.
    /// 
    /// It works by comparing the model's performance on validation data versus test data.
    /// If these performances are very similar, we're more confident in our diagnosis.
    /// If they're very different, we're less confident.
    /// 
    /// The confidence is expressed as a number between 0 and 1:
    /// - 1.0 means complete confidence (validation and test performances are identical)
    /// - 0.0 means no confidence (validation and test performances are completely different)
    /// 
    /// Think of it like a doctor saying "I'm 90% sure of this diagnosis" versus
    /// "I'm only 50% sure of this diagnosis."
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var avgValidationMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
        var testMetric = evaluationData.TestSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        var metricDifference = NumOps.Abs(NumOps.Subtract(avgValidationMetric, testMetric));
        var maxMetric = NumOps.GreaterThan(avgValidationMetric, testMetric) ? avgValidationMetric : testMetric;

        var confidence = NumOps.Subtract(NumOps.One, NumOps.Divide(metricDifference, maxMetric));
        return NumOps.GreaterThan(confidence, NumOps.Zero) ? confidence : NumOps.Zero;
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The type of fit detected in the model (e.g., GoodFit, Overfit, Underfit).</param>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>A list of string recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a personalized "action plan" for your model based on its diagnosis.
    /// 
    /// Think of it like getting advice from a doctor after they've diagnosed your condition:
    /// 
    /// - If your model has a "Good Fit" (healthy condition), it suggests minor improvements or deployment.
    /// 
    /// - If your model is "Overfitting" (memorizing instead of learning), it recommends ways to make
    ///   your model more general, such as:
    ///   * Regularization: Adding constraints to prevent the model from becoming too complex
    ///   * Reducing complexity: Simplifying your model so it focuses on important patterns
    ///   * More diverse data: Giving your model a wider variety of examples to learn from
    /// 
    /// - If your model is "Underfitting" (too simple to learn patterns), it suggests ways to
    ///   increase your model's learning capacity, such as:
    ///   * Increasing complexity: Making your model more sophisticated
    ///   * Adding features: Giving your model more relevant information to work with
    ///   * Reducing constraints: Removing limitations that might be holding your model back
    /// 
    /// - If your model has "High Variance" (performs differently on different data), it recommends
    ///   ways to make your model more stable, such as:
    ///   * More training data: Giving your model more examples to learn from
    ///   * Ensemble methods: Combining multiple models to get more reliable predictions
    ///   * Feature selection: Focusing on the most important information and ignoring noise
    /// 
    /// - If your model is "Unstable" (inconsistent performance), it suggests investigating
    ///   potential issues with your data or model setup.
    /// 
    /// The method also includes specific performance metrics to help you understand how your
    /// model is currently performing.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on Stratified K-Fold Cross-Validation.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more diverse training data");
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
                recommendations.Add("- Investigating data quality and class distribution");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust cross-validation techniques");
                break;
        }

        var primaryMetric = _options.PrimaryMetric;
        recommendations.Add($"Average Validation {primaryMetric}: {evaluationData.ValidationSet.PredictionStats.GetMetric(primaryMetric):F4}, Test {primaryMetric}: {evaluationData.TestSet.PredictionStats.GetMetric(primaryMetric):F4}");

        return recommendations;
    }
}
