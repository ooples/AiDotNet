namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit quality using precision-recall curve metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you understand how well your classification model is performing
/// by analyzing two important metrics:
/// 
/// 1. AUC (Area Under the Curve): A number between 0 and 1 that tells you how well your model can
///    distinguish between classes. Higher is better, with 1.0 being perfect.
///    
/// 2. F1 Score: A number between 0 and 1 that balances precision (how many of your positive 
///    predictions were correct) and recall (how many actual positives your model found).
///    Higher is better, with 1.0 being perfect.
///    
/// This detector is especially useful for classification problems where you're trying to identify
/// specific categories or classes.
/// </para>
/// </remarks>
public class PrecisionRecallCurveFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    private readonly PrecisionRecallCurveFitDetectorOptions _options;

    /// <summary>
    /// Gets or sets the Area Under the Curve value for the precision-recall curve.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AUC (Area Under the Curve) is a value between 0 and 1 that measures how well
    /// your model can distinguish between classes. Think of it as a grade for your model's ability
    /// to separate positive examples from negative ones.
    /// 
    /// - AUC = 1.0: Perfect classification (your model is always correct)
    /// - AUC = 0.5: No better than random guessing (like flipping a coin)
    /// - AUC = 0.0: Completely incorrect classification (your model is always wrong)
    /// 
    /// Generally, you want this value to be as close to 1.0 as possible.
    /// </para>
    /// </remarks>
    private double Auc { get; set; }

    /// <summary>
    /// Gets or sets the F1 Score, which is the harmonic mean of precision and recall.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The F1 Score is a single number between 0 and 1 that combines two important metrics:
    /// 
    /// 1. Precision: Out of all the items your model predicted as positive, how many were actually positive?
    ///    (Like: "When my model says it's a cat, how often is it really a cat?")
    ///    
    /// 2. Recall: Out of all the actual positive items, how many did your model correctly identify?
    ///    (Like: "Out of all the cats in my images, how many did my model find?")
    ///    
    /// The F1 Score balances these two concerns. A high F1 Score (close to 1.0) means your model
    /// is both precise and thorough in its predictions.
    /// </para>
    /// </remarks>
    private double F1Score { get; set; }

    /// <summary>
    /// Initializes a new instance of the PrecisionRecallCurveFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options for the detector. If null, default options will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector. You can provide custom options
    /// to control how the detector evaluates your model, or leave it as null to use the default settings.
    /// 
    /// The options control things like what threshold values determine a "good" AUC or F1 Score.
    /// </para>
    /// </remarks>
    public PrecisionRecallCurveFitDetector(PrecisionRecallCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new PrecisionRecallCurveFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes the model's performance data and determines the quality of fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations.</returns>
    /// <exception cref="ArgumentNullException">Thrown when evaluationData is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model's performance and tells you
    /// how well it's doing. It:
    /// 
    /// 1. Calculates the AUC and F1 Score from your model's predictions
    /// 2. Determines if your model has a good fit, poor fit, or something in between
    /// 3. Calculates how confident it is in this assessment
    /// 4. Provides specific recommendations to help you improve your model
    /// 
    /// The result contains all this information in an organized way that you can use to understand
    /// and improve your model.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        (var auc, var f1Score) = StatisticsHelper<T>.CalculateAucF1Score(evaluationData);
        (Auc, F1Score) = (Convert.ToDouble(auc), Convert.ToDouble(f1Score));

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
                { "AUC", Auc },
                { "F1Score", F1Score }
            }
        };
    }

    /// <summary>
    /// Determines the type of fit based on AUC and F1 Score values.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A FitType enum value indicating the quality of the model fit.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at your model's AUC and F1 Score and decides if your model is:
    /// 
    /// - Good Fit: Both AUC and F1 Score are above the thresholds you set (or the default thresholds)
    /// - Poor Fit: Both AUC and F1 Score are below the thresholds
    /// - Moderate: Some metrics are good, others need improvement
    /// 
    /// Think of it like getting a grade on your model's performance - A (Good), C (Moderate), or F (Poor).
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (Auc > _options.AreaUnderCurveThreshold && F1Score > _options.F1ScoreThreshold)
        {
            return FitType.GoodFit;
        }
        else if (Auc < _options.AreaUnderCurveThreshold && F1Score < _options.F1ScoreThreshold)
        {
            return FitType.PoorFit;
        }
        else
        {
            return FitType.Moderate;
        }
    }

    /// <summary>
    /// Calculates the confidence level in the fit detection result.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident we are in our assessment of your model.
    /// 
    /// It combines the AUC and F1 Score using weights that you can customize in the options.
    /// By default, both metrics are weighted equally.
    /// 
    /// A higher confidence value (closer to 1) means we're more certain about our assessment of your model.
    /// This can help you decide how much to trust the recommendations provided.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Calculate confidence level as a weighted average of AUC and F1 Score
        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Auc), NumOps.FromDouble(_options.AucWeight)), NumOps.Multiply(NumOps.FromDouble(F1Score), NumOps.FromDouble(_options.F1ScoreWeight)));
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type (good fit, moderate, or poor fit).</param>
    /// <param name="evaluationData">Data containing model performance metrics across training, validation, and test sets.</param>
    /// <returns>A list of recommendations as strings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your model based on its current performance.
    /// 
    /// The recommendations are tailored to the specific fit type:
    /// 
    /// - Good Fit: Suggestions for fine-tuning an already well-performing model
    /// - Moderate: Ideas for improving specific aspects of your model that need work
    /// - Poor Fit: More extensive recommendations for addressing fundamental issues with your model
    /// 
    /// These recommendations are designed to be actionable steps you can take to improve your model's performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good performance based on the precision-recall curve analysis.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Moderate:
                recommendations.Add("The model shows moderate performance. There's room for improvement.");
                recommendations.Add("Try adjusting the classification threshold to optimize precision and recall trade-off.");
                recommendations.Add("Consider feature engineering or selection to improve model performance.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model's performance is suboptimal based on the precision-recall curve analysis.");
                recommendations.Add("Review your feature set and consider adding more relevant features.");
                recommendations.Add("Experiment with different algorithms or ensemble methods.");
                recommendations.Add("Check for class imbalance and consider using techniques like oversampling or undersampling.");
                break;
        }

        return recommendations;
    }
}
