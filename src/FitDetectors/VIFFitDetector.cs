namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit by analyzing Variance Inflation Factor (VIF) to identify multicollinearity issues.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you identify if your model has a problem called "multicollinearity" - 
/// which happens when your input features (variables) are too closely related to each other.
/// 
/// Think of multicollinearity like this: if you're trying to predict house prices using both 
/// "square footage" and "number of rooms" as inputs, these two features might be strongly related 
/// (bigger houses tend to have more rooms). This relationship can confuse your model and make it 
/// less reliable.
/// 
/// The VIF (Variance Inflation Factor) is like a warning system that measures how much each feature 
/// is related to other features. Higher VIF values mean stronger relationships:
/// - VIF = 1: No relationship with other features (ideal)
/// - VIF = 5-10: Moderate relationship (concerning)
/// - VIF > 10: Strong relationship (problematic)
/// 
/// This detector helps you identify these issues and suggests ways to fix them.
/// </para>
/// </remarks>
public class VIFFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the VIF fit detector.
    /// </summary>
    private readonly VIFFitDetectorOptions _options;

    /// <summary>
    /// Configuration options for model statistics calculations.
    /// </summary>
    private readonly ModelStatsOptions _modelStatsOptions;

    /// <summary>
    /// Initializes a new instance of the VIFFitDetector class.
    /// </summary>
    /// <param name="options">Configuration options for the VIF detector. If null, default options are used.</param>
    /// <param name="modelStatsOptions">Configuration options for model statistics. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new VIF detector. You can customize how 
    /// it works by providing options, or just use the default settings if you're not sure.
    /// </para>
    /// </remarks>
    public VIFFitDetector(VIFFitDetectorOptions? options = null, ModelStatsOptions? modelStatsOptions = null)
    {
        _options = options ?? new VIFFitDetectorOptions();
        _modelStatsOptions = modelStatsOptions ?? new ModelStatsOptions();
    }

    /// <summary>
    /// Analyzes the model's performance data to detect fit issues related to multicollinearity.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics and statistics.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines your model and tells you if it has problems.
    /// It works in three steps:
    /// 
    /// 1. It determines what type of fit your model has (good, poor, or has multicollinearity issues)
    /// 2. It calculates how confident it is about this assessment
    /// 3. It creates a list of recommendations to help you improve your model
    /// 
    /// The result combines all this information into one package that helps you understand 
    /// your model's health and how to make it better.
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
    /// Determines the type of fit based on VIF values and performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics and statistics.</param>
    /// <returns>The detected fit type (SevereMulticollinearity, ModerateMulticollinearity, GoodFit, or PoorFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines your model to determine if it has multicollinearity problems
    /// or other fit issues. It does this by:
    /// 
    /// 1. Calculating VIF values for each feature (higher values indicate multicollinearity)
    /// 2. Finding the highest VIF value
    /// 3. Comparing this value to thresholds to determine if there's a problem
    /// 4. If multicollinearity isn't the main issue, it checks if the model performs well overall
    /// 
    /// The result tells you which category your model falls into, which helps determine
    /// what kind of improvements you should make.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var vifValues = StatisticsHelper<T>.CalculateVIF(evaluationData.ModelStats.CorrelationMatrix, _modelStatsOptions);
        var maxVIF = vifValues.Max() ?? NumOps.Zero;

        if (NumOps.GreaterThan(maxVIF, NumOps.FromDouble(_options.SevereMulticollinearityThreshold)))
        {
            return FitType.SevereMulticollinearity;
        }
        else if (NumOps.GreaterThan(maxVIF, NumOps.FromDouble(_options.ModerateMulticollinearityThreshold)))
        {
            return FitType.ModerateMulticollinearity;
        }
        else
        {
            var primaryMetric = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);
            if (NumOps.GreaterThan(primaryMetric, NumOps.FromDouble(_options.GoodFitThreshold)))
            {
                return FitType.GoodFit;
            }
            else
            {
                return FitType.PoorFit;
            }
        }
    }

    /// <summary>
    /// Calculates the confidence level of the fit detection based on VIF values and performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing the model's performance metrics and statistics.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is about its assessment of your model.
    /// 
    /// It calculates confidence by:
    /// 1. Looking at both the average and maximum VIF values
    /// 2. Considering how well your model performs on validation data
    /// 3. Combining these factors into a single confidence score
    /// 
    /// A higher confidence score (closer to 1) means the detector is more certain about its assessment.
    /// A lower score (closer to 0) means there's more uncertainty, and you might want to investigate further.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var vifValues = StatisticsHelper<T>.CalculateVIF(evaluationData.ModelStats.CorrelationMatrix, _modelStatsOptions);
        var maxVIF = vifValues.Max() ?? NumOps.Zero;
        var avgVIF = NumOps.Divide(vifValues.Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(vifValues.Count));

        var vifConfidence = NumOps.Subtract(NumOps.One, NumOps.Divide(avgVIF, maxVIF));
        var metricConfidence = evaluationData.ValidationSet.PredictionStats.GetMetric(_options.PrimaryMetric);

        return NumOps.Multiply(vifConfidence, metricConfidence);
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The type of fit detected for the model.</param>
    /// <param name="evaluationData">Data containing the model's performance metrics.</param>
    /// <returns>A list of string recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a personalized "improvement plan" for your model based on its issues.
    /// 
    /// Depending on what problems were detected, it provides different advice:
    /// 
    /// - For severe multicollinearity: Suggests removing highly related features or using techniques 
    ///   like Ridge regression that can handle related features better
    ///   
    /// - For moderate multicollinearity: Offers milder suggestions to investigate and potentially 
    ///   combine related features
    ///   
    /// - For good fit: Congratulates you on a well-performing model and suggests fine-tuning
    ///   
    /// - For poor fit: Recommends ways to improve overall model performance
    /// 
    /// It also includes your model's actual performance metrics so you can track improvements.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(
        FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.SevereMulticollinearity:
                recommendations.Add("Severe multicollinearity detected. Consider:");
                recommendations.Add("- Removing highly correlated features");
                recommendations.Add("- Using regularization techniques (e.g., Ridge, Lasso)");
                recommendations.Add("- Applying dimensionality reduction methods (e.g., PCA)");
                break;
            case FitType.ModerateMulticollinearity:
                recommendations.Add("Moderate multicollinearity detected. Consider:");
                recommendations.Add("- Investigating feature correlations");
                recommendations.Add("- Combining or removing some correlated features");
                recommendations.Add("- Using regularization techniques");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit with acceptable multicollinearity levels.");
                recommendations.Add("Consider fine-tuning for even better performance.");
                break;
            case FitType.PoorFit:
                recommendations.Add("The model shows poor fit. Consider:");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Collecting more training data");
                break;
        }

        var primaryMetric = _options.PrimaryMetric;
        recommendations.Add($"Validation {primaryMetric}: {evaluationData.ValidationSet.PredictionStats.GetMetric(primaryMetric):F4}, Test {primaryMetric}: {evaluationData.TestSet.PredictionStats.GetMetric(primaryMetric):F4}");

        return recommendations;
    }
}
