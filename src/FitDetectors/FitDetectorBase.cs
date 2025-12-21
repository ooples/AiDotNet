namespace AiDotNet.FitDetectors;

/// <summary>
/// Base class for all fit detectors that provides common functionality and defines the required interface.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This abstract class serves as a template for all fit detectors in the library. 
/// It defines the common structure and behavior that all fit detectors should have, while allowing 
/// specific implementations to customize how they detect different types of model fit.
/// </para>
/// <para>
/// A fit detector analyzes a machine learning model's performance to determine if it's underfitting 
/// (too simple), overfitting (too complex), or has a good fit (just right). This helps you understand 
/// how to improve your model.
/// </para>
/// </remarks>
public abstract class FitDetectorBase<T, TInput, TOutput> : IFitDetector<T, TInput, TOutput>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This helper object allows the fit detector to perform mathematical operations 
    /// (like addition, multiplication, etc.) on the generic type T, which could be float, double, or 
    /// another numeric type.
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Random number generator used for feature permutation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is used to randomly shuffle feature values when calculating 
    /// permutation importance. Using a fixed seed ensures reproducible results.
    /// </remarks>
    protected readonly Random Random;

    /// <summary>
    /// Initializes a new instance of the FitDetectorBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor initializes the base functionality for all fit detectors.
    /// </para>
    /// <para>
    /// It sets up the numeric operations helper that will be used for mathematical calculations 
    /// throughout the fit detection process.
    /// </para>
    /// </remarks>
    protected FitDetectorBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Random = new();
    }

    /// <summary>
    /// Analyzes model performance data and determines the type of fit, confidence level, and recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>A result object containing the fit type, confidence level, and recommendations for improvement</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that all fit detectors must implement. It takes 
    /// data about your model's performance and returns an assessment of how well the model fits the data.
    /// </para>
    /// <para>
    /// Each specific fit detector will implement this method differently based on what techniques it uses 
    /// to assess model fit.
    /// </para>
    /// </remarks>
    public abstract FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData);

    /// <summary>
    /// Determines the type of fit based on model performance metrics.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>The detected fit type (e.g., GoodFit, Overfit, Underfit)</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method must be implemented by each specific fit detector to determine 
    /// what type of fit your model has.
    /// </para>
    /// <para>
    /// Different fit detectors will use different techniques and metrics to categorize the model's fit, 
    /// such as:
    /// <list type="bullet">
    /// <item><description>Comparing performance across training, validation, and test datasets</description></item>
    /// <item><description>Analyzing feature importances</description></item>
    /// <item><description>Examining residuals or error patterns</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected abstract FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData);

    /// <summary>
    /// Calculates the confidence level in the fit type determination.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>A value indicating the confidence level of the detection</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method must be implemented by each specific fit detector to determine 
    /// how confident it is in its assessment of your model's fit.
    /// </para>
    /// <para>
    /// The confidence level is typically a value between 0 and 1, with higher values indicating greater 
    /// confidence in the fit assessment.
    /// </para>
    /// </remarks>
    protected abstract T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData);

    /// <summary>
    /// Generates recommendations for improving the model based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type</param>
    /// <param name="evaluationData">Data containing model performance metrics for training, validation, and test sets</param>
    /// <returns>A list of recommendations as strings</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model.
    /// </para>
    /// <para>
    /// The base implementation provides general recommendations for each fit type, but specific fit 
    /// detectors can override this method to provide more tailored recommendations based on their 
    /// specific analysis techniques.
    /// </para>
    /// <para>
    /// Different types of fit issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Good Fit: The model is performing well and may be ready for deployment</description></item>
    /// <item><description>High Variance: The model is too sensitive to fluctuations in the training data</description></item>
    /// <item><description>High Bias: The model is too simple to capture the complexity of the data</description></item>
    /// <item><description>Overfit: The model is too complex and memorizes training data instead of learning patterns</description></item>
    /// <item><description>Underfit: The model is too simple and fails to capture important patterns</description></item>
    /// <item><description>Unstable: The model's performance varies too much across different datasets</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected virtual List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model appears to be well-fitted. Consider deploying it and monitoring its performance on new data.");
                break;
            case FitType.HighVariance:
                recommendations.Add("The model shows high variance. Consider simplifying the model, using regularization techniques, or gathering more diverse training data.");
                break;
            case FitType.HighBias:
                recommendations.Add("The model shows high bias. Consider increasing model complexity, adding more relevant features, or using a different algorithm.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider using regularization, reducing model complexity, or gathering more training data.");
                break;
            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider increasing model complexity, reducing regularization, or adding more relevant features.");
                break;
            case FitType.Unstable:
                recommendations.Add("The model's performance is unstable across datasets. Consider using more robust feature selection, cross-validation techniques, or ensemble methods.");
                break;
        }

        return recommendations;
    }
}
