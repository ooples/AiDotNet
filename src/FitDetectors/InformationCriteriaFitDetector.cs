namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit using information criteria metrics (AIC and BIC).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine if your machine learning model is a good fit for your data
/// by using special metrics called "information criteria" (AIC and BIC). These metrics help balance
/// how well your model performs against how complex it is. A good model should explain your data well
/// without being unnecessarily complicated.
/// 
/// Think of it like this: if you're trying to draw a line through some points, you want a line that's
/// close to most points (good performance) but isn't so wiggly that it's just memorizing the exact
/// positions of each point (too complex).
/// </para>
/// </remarks>
public class InformationCriteriaFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the information criteria fit detector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how strict or lenient the detector is when evaluating your model.
    /// They include thresholds that determine when a model is considered to be overfitting, underfitting, etc.
    /// </para>
    /// </remarks>
    private readonly InformationCriteriaFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the InformationCriteriaFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector. You can provide custom settings
    /// through the options parameter, or just let it use the default settings if you're not sure what to change.
    /// </para>
    /// </remarks>
    public InformationCriteriaFitDetector(InformationCriteriaFitDetectorOptions? options = null)
    {
        _options = options ?? new InformationCriteriaFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes model evaluation data to determine the type of fit and provide recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations for improvement.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use. It takes your model's performance data and:
    /// 1. Figures out if your model is a good fit, overfitting, underfitting, etc.
    /// 2. Calculates how confident it is in that assessment
    /// 3. Provides specific recommendations to improve your model
    /// 
    /// All of this is packaged into a single result that you can use to understand and improve your model.
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
    /// Determines the type of fit based on information criteria metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>The determined fit type (GoodFit, Overfit, Underfit, HighVariance, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at the AIC and BIC values (information criteria metrics) from your
    /// training, validation, and test data. It then compares these values to determine if your model:
    /// 
    /// - Is a good fit (balanced performance across all datasets)
    /// - Is overfitting (performs much better on training data than on new data)
    /// - Is underfitting (performs poorly on all datasets)
    /// - Has high variance (performance varies a lot between different datasets)
    /// - Is unstable (doesn't show consistent patterns in performance)
    /// 
    /// AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) are mathematical formulas
    /// that balance how well your model fits the data against how complex it is. Lower values are generally better.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingAic = evaluationData.TrainingSet.ErrorStats.AIC;
        var validationAic = evaluationData.ValidationSet.ErrorStats.AIC;
        var testAic = evaluationData.TestSet.ErrorStats.AIC;

        var trainingBic = evaluationData.TrainingSet.ErrorStats.BIC;
        var validationBic = evaluationData.ValidationSet.ErrorStats.BIC;
        var testBic = evaluationData.TestSet.ErrorStats.BIC;

        var aicDiff = Convert.ToDouble(NumOps.Subtract(NumOps.GreaterThan(validationAic, testAic) ? validationAic : testAic, trainingAic));
        var bicDiff = Convert.ToDouble(NumOps.Subtract(NumOps.GreaterThan(validationBic, testBic) ? validationBic : testBic, trainingBic));

        if (aicDiff < _options.AicThreshold && bicDiff < _options.BicThreshold)
        {
            return FitType.GoodFit;
        }
        else if (aicDiff > _options.OverfitThreshold || bicDiff > _options.OverfitThreshold)
        {
            return FitType.Overfit;
        }
        else if (aicDiff < -_options.UnderfitThreshold || bicDiff < -_options.UnderfitThreshold)
        {
            return FitType.Underfit;
        }
        else if (Math.Abs(Convert.ToDouble(NumOps.Subtract(validationAic, testAic))) > _options.HighVarianceThreshold ||
                 Math.Abs(Convert.ToDouble(NumOps.Subtract(validationBic, testBic))) > _options.HighVarianceThreshold)
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    /// <summary>
    /// Calculates the confidence level in the fit determination.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident the detector is in its assessment of your model.
    /// The confidence is based on how consistent the AIC and BIC values are across your different datasets.
    /// 
    /// If the values are very similar across training, validation, and test sets, the confidence will be high.
    /// If there are big differences, the confidence will be lower.
    /// 
    /// The result is a number between 0 and 1:
    /// - Values close to 1 mean high confidence (you can trust the assessment)
    /// - Values close to 0 mean low confidence (the assessment might not be reliable)
    /// 
    /// The calculation uses a mathematical formula that converts differences in AIC/BIC into a confidence score.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingAic = evaluationData.TrainingSet.ErrorStats.AIC;
        var validationAic = evaluationData.ValidationSet.ErrorStats.AIC;
        var testAic = evaluationData.TestSet.ErrorStats.AIC;

        var trainingBic = evaluationData.TrainingSet.ErrorStats.BIC;
        var validationBic = evaluationData.ValidationSet.ErrorStats.BIC;
        var testBic = evaluationData.TestSet.ErrorStats.BIC;

        var aicConfidence = Math.Exp(-(Convert.ToDouble(NumOps.Subtract(validationAic, trainingAic)) / 2)) *
            Math.Exp(-(Convert.ToDouble(NumOps.Subtract(testAic, trainingAic))) / 2);
        var bicConfidence = Math.Exp(-(Convert.ToDouble(NumOps.Subtract(validationBic, trainingBic)) / 2)) *
            Math.Exp(-(Convert.ToDouble(NumOps.Subtract(testBic, trainingBic))) / 2);

        var averageConfidence = (aicConfidence + bicConfidence) / 2;

        return NumOps.FromDouble(averageConfidence);
    }

    /// <summary>
    /// Generates specific recommendations based on the detected fit type of the model.
    /// </summary>
    /// <param name="fitType">The type of fit detected (GoodFit, Overfit, Underfit, HighVariance, or Unstable).</param>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A list of string recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a list of practical suggestions to help you improve your model
    /// based on what type of fit was detected. Think of it as personalized advice for your specific situation:
    /// 
    /// - If your model has a "Good Fit", it suggests you're ready to use the model or make minor improvements.
    /// - If your model is "Overfitting", it suggests ways to make your model more generalized (less memorizing, more learning).
    /// - If your model is "Underfitting", it suggests ways to make your model more powerful to capture patterns better.
    /// - If your model has "High Variance", it suggests ways to make your model more consistent across different data.
    /// - If your model is "Unstable", it suggests ways to address fundamental issues with your data or model structure.
    /// 
    /// The method also includes the threshold values used to determine fit types, which can help you understand
    /// how strict or lenient the evaluation was.
    /// 
    /// Terminology explained:
    /// - Regularization: A technique that prevents models from becoming too complex by adding penalties for complexity
    /// - Ensemble methods: Combining multiple models together to improve performance and stability
    /// - Cross-validation: A technique where you train and test your model on different subsets of data to ensure consistency
    /// - Feature selection: The process of choosing which input variables (features) to include in your model
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit based on information criteria.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Information criteria suggest potential overfitting. Consider:");
                recommendations.Add("- Increasing regularization");
                recommendations.Add("- Reducing model complexity");
                recommendations.Add("- Collecting more training data");
                break;
            case FitType.Underfit:
                recommendations.Add("Information criteria indicate underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Information criteria show high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Using ensemble methods");
                recommendations.Add("- Applying cross-validation techniques");
                break;
            case FitType.Unstable:
                recommendations.Add("Information criteria indicate unstable performance. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Experimenting with different model architectures");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"AIC threshold: {_options.AicThreshold}, BIC threshold: {_options.BicThreshold}");

        return recommendations;
    }
}
