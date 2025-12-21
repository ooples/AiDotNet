namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality using ROC (Receiver Operating Characteristic) curve analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine how well your classification model performs.
/// It uses something called an "ROC curve" which is a way to visualize how good your model is at 
/// distinguishing between positive cases (like "yes, this email is spam") and negative cases 
/// (like "no, this email is not spam").
/// 
/// The key metric used is called "AUC" (Area Under the Curve), which gives a single number between 0 and 1:
/// - AUC near 1.0: Your model is excellent at classification
/// - AUC near 0.5: Your model is no better than random guessing
/// - AUC near 0.0: Your model is consistently wrong (which means it could be fixed by inverting its predictions)
/// 
/// This detector will tell you if your model is good, moderate, poor, or very poor based on this AUC value.
/// </para>
/// </remarks>
public class ROCCurveFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the ROC curve fit detector.
    /// </summary>
    private readonly ROCCurveFitDetectorOptions _options;

    /// <summary>
    /// The Area Under the Curve (AUC) value calculated from the ROC curve.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AUC stands for "Area Under the Curve" and is a value between 0 and 1.
    /// It measures how well your model can distinguish between classes. The higher the AUC,
    /// the better your model is at predicting the correct classes.
    /// 
    /// Think of it like a school grade:
    /// - 0.9-1.0: A (Excellent)
    /// - 0.8-0.9: B (Good)
    /// - 0.7-0.8: C (Fair)
    /// - 0.6-0.7: D (Poor)
    /// - 0.5-0.6: F (Failing/Random guessing)
    /// </para>
    /// </remarks>
    private T Auc { get; set; }

    /// <summary>
    /// Creates a new instance of the ROC curve fit detector with optional custom configuration.
    /// </summary>
    /// <param name="options">Custom options for the detector, or null to use default options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new ROC curve detector.
    /// You can provide custom settings through the options parameter, or leave it as null
    /// to use the default settings.
    /// </para>
    /// </remarks>
    public ROCCurveFitDetector(ROCCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new ROCCurveFitDetectorOptions();
        Auc = NumOps.Zero;
    }

    /// <summary>
    /// Analyzes the model's performance data and determines how well the model fits the data.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics and predictions.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <exception cref="ArgumentNullException">Thrown when evaluationData is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that analyzes your model's performance.
    /// It takes your model's predictions and the actual correct answers, then:
    /// 
    /// 1. Creates an ROC curve (a special graph that shows how well your model distinguishes between classes)
    /// 2. Calculates the AUC (Area Under the Curve) - a single number that summarizes how good your model is
    /// 3. Determines if your model has a good fit, moderate fit, poor fit, or very poor fit
    /// 4. Calculates how confident it is in this assessment
    /// 5. Generates helpful recommendations for improving your model
    /// 
    /// The result contains all this information for you to use.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        var (fpr, tpr) = StatisticsHelper<T>.CalculateROCCurve(ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual),
            ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted));
        Auc = StatisticsHelper<T>.CalculateAUC(fpr, tpr);

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
                { "AUC", Convert.ToDouble(Auc) },
                { "FPR", fpr },
                { "TPR", tpr }
            }
        };
    }

    /// <summary>
    /// Determines the type of fit based on the AUC value.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The determined fit type (GoodFit, Moderate, PoorFit, or VeryPoorFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at the AUC value (which measures how good your model is)
    /// and categorizes your model into one of four groups:
    /// 
    /// - GoodFit: Your model performs very well (high AUC)
    /// - Moderate: Your model performs okay but could be improved
    /// - PoorFit: Your model doesn't perform well
    /// - VeryPoorFit: Your model performs very poorly (low AUC)
    /// 
    /// The thresholds for these categories are set in the options.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (NumOps.GreaterThanOrEquals(Auc, NumOps.FromDouble(_options.GoodFitThreshold)))
            return FitType.GoodFit;
        else if (NumOps.GreaterThanOrEquals(Auc, NumOps.FromDouble(_options.ModerateFitThreshold)))
            return FitType.Moderate;
        else if (NumOps.GreaterThanOrEquals(Auc, NumOps.FromDouble(_options.PoorFitThreshold)))
            return FitType.PoorFit;
        else
            return FitType.VeryPoorFit;
    }

    /// <summary>
    /// Calculates how confident the detector is in its assessment of the model's fit type.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sure the detector is about its assessment of your model.
    /// It uses the AUC value and multiplies it by a scaling factor to get a confidence score.
    /// 
    /// The higher the AUC, the more confident the detector is in its assessment.
    /// The confidence is a number between 0 and 1, where:
    /// - 1 means completely confident
    /// - 0 means not confident at all
    /// 
    /// This helps you understand how much you should trust the detector's assessment.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        return NumOps.Multiply(Auc, NumOps.FromDouble(_options.ConfidenceScalingFactor));
    }

    /// <summary>
    /// Generates practical recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type (GoodFit, Moderate, PoorFit, or VeryPoorFit).</param>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on
    /// how well it's currently performing.
    /// 
    /// - For good models: Suggestions for fine-tuning to make small improvements
    /// - For moderate models: Suggestions like feature engineering or trying different algorithms
    /// - For poor models: Suggestions to review your features, try different algorithms, or get more data
    /// - For very poor models: Suggestions to reassess your approach more fundamentally
    /// 
    /// It also checks if your data might be imbalanced (having many more examples of one class than another),
    /// which is a common problem in classification tasks.
    /// 
    /// These recommendations are starting points that you can try to improve your model's performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance. Consider fine-tuning for potential improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model performance is moderate. Consider feature engineering or trying different algorithms.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model performance is poor. Review feature selection, try different algorithms, or gather more data.");
                break;
            case FitType.VeryPoorFit:
                recommendations.Add("The model performance is very poor. Reassess the problem formulation, data quality, and chosen algorithm.");
                break;
        }

        if (NumOps.LessThan(Auc, NumOps.FromDouble(_options.BalancedDatasetThreshold)))
        {
            recommendations.Add("The dataset might be imbalanced. Consider using balanced accuracy, F1 score, or other metrics suitable for imbalanced data.");
        }

        return recommendations;
    }
}
