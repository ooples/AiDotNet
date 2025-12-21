namespace AiDotNet.FitDetectors;

/// <summary>
/// A default implementation of a fit detector that analyzes model performance and provides recommendations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double)</typeparam>
/// <remarks>
/// This class evaluates how well a machine learning model fits the data by comparing performance
/// metrics across training, validation, and test datasets. It can detect common issues like
/// overfitting and underfitting, and provide appropriate recommendations.
/// </remarks>
public class DefaultFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the DefaultFitDetector class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new default fit detector with standard settings.
    /// </para>
    /// <para>
    /// Unlike other fit detectors that may require configuration options, the DefaultFitDetector uses 
    /// predefined thresholds and logic to assess model fit, making it a good starting point for 
    /// beginners or for quick model evaluation.
    /// </para>
    /// </remarks>
    public DefaultFitDetector()
    {
    }

    /// <summary>
    /// Analyzes model performance data and determines the type of fit, confidence level, and recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations for improvement</returns>
    /// <remarks>
    /// This method examines how well your model performs on different datasets to determine if it's
    /// learning properly. It checks for common problems like:
    /// - Overfitting: When your model performs very well on training data but poorly on new data
    /// - Underfitting: When your model fails to capture the underlying pattern in the data
    /// - High variance: When your model is too sensitive to small fluctuations in the training data
    /// - High bias: When your model is too simple to capture the complexity of the data
    /// 
    /// Based on this analysis, it provides practical recommendations to improve your model.
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType);

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
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>The detected fit type (e.g., GoodFit, Overfit, Underfit)</returns>
    /// <remarks>
    /// This method uses R² (R-squared) values to classify the model's fit:
    /// - GoodFit: High R² values across all datasets (>0.9)
    /// - Overfit: High R² on training (>0.9) but lower on validation (<0.7)
    /// - Underfit: Low R² values on both training and validation (<0.7)
    /// - HighVariance: Large difference between training and validation R² (>0.2)
    /// - HighBias: Very low R² values across all datasets (<0.5)
    /// - Unstable: Any other pattern that doesn't fit the above categories
    /// 
    /// R² is a statistical measure that represents how well the model explains the variance in the data,
    /// with values closer to 1 indicating better fit.
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        T threshold09 = NumOps.FromDouble(0.9);
        T threshold07 = NumOps.FromDouble(0.7);
        T threshold05 = NumOps.FromDouble(0.5);
        T threshold02 = NumOps.FromDouble(0.2);
        var training = evaluationData.TrainingSet.PredictionStats;
        var validation = evaluationData.ValidationSet.PredictionStats;
        var test = evaluationData.TestSet.PredictionStats;

        if (NumOps.GreaterThan(training.R2, threshold09) && NumOps.GreaterThan(validation.R2, threshold09) && NumOps.GreaterThan(test.R2, threshold09))
            return FitType.GoodFit;
        if (NumOps.GreaterThan(training.R2, threshold09) && NumOps.LessThan(validation.R2, threshold07))
            return FitType.Overfit;
        if (NumOps.LessThan(training.R2, threshold07) && NumOps.LessThan(validation.R2, threshold07))
            return FitType.Underfit;
        if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(training.R2, validation.R2)), threshold02))
            return FitType.HighVariance;
        if (NumOps.LessThan(training.R2, threshold05) && NumOps.LessThan(validation.R2, threshold05) && NumOps.LessThan(test.R2, threshold05))
            return FitType.HighBias;

        return FitType.Unstable;
    }

    /// <summary>
    /// Calculates the overall confidence level in the model's performance.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>A confidence score between 0 and 1, with higher values indicating better performance</returns>
    /// <remarks>
    /// This method calculates the average R² value across training, validation, and test datasets.
    /// The resulting value gives a simple measure of overall model quality:
    /// - Values close to 1 indicate high confidence in the model's predictions
    /// - Values close to 0 indicate poor model performance
    /// 
    /// This average helps balance the performance across different datasets, ensuring the model
    /// works well on both seen and unseen data.
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        return NumOps.Divide(NumOps.Add(NumOps.Add(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2),
            evaluationData.TestSet.PredictionStats.R2), NumOps.FromDouble(3));
    }

    /// <summary>
    /// Generates practical recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The type of fit detected in the model</param>
    /// <returns>A list of recommendations as strings</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method provides actionable advice based on the specific issues 
    /// detected in your model.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Overfitting: The model is too complex and needs to be simplified</description></item>
    /// <item><description>Underfitting: The model is too simple and needs more complexity</description></item>
    /// <item><description>High Variance: The model's performance varies too much across different data</description></item>
    /// <item><description>High Bias: The model is too simple to capture the complexity of the data</description></item>
    /// <item><description>Unstable: The model's performance is inconsistent and needs further investigation</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The recommendations are designed to be practical and actionable, focusing on common techniques 
    /// that can address each type of issue.
    /// </para>
    /// </remarks>
    private List<string> GenerateRecommendations(FitType fitType)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization");
                recommendations.Add("Try reducing model complexity");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider decreasing regularization");
                recommendations.Add("Try increasing model complexity");
                break;
            case FitType.HighVariance:
                recommendations.Add("Collect more training data");
                recommendations.Add("Try feature selection to reduce noise");
                break;
            case FitType.HighBias:
                recommendations.Add("Add more features");
                recommendations.Add("Increase model complexity");
                break;
            case FitType.Unstable:
                recommendations.Add("Check for data quality issues");
                recommendations.Add("Consider ensemble methods for more stable predictions");
                break;
        }

        return recommendations;
    }
}
