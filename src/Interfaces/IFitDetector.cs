namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for detecting how well a machine learning model fits the data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface helps determine if your model is learning properly or has problems.
/// 
/// When training machine learning models, three common problems can occur:
/// 
/// 1. Underfitting: The model is too simple and doesn't capture important patterns in the data.
///    - Like using only a house's age to predict its price, ignoring size, location, etc.
///    - Signs: Poor performance on both training and new data
/// 
/// 2. Overfitting: The model memorizes the training data instead of learning general patterns.
///    - Like memorizing specific houses and their prices instead of understanding what makes houses valuable
///    - Signs: Excellent performance on training data but poor performance on new data
/// 
/// 3. Good fit: The model captures the important patterns without memorizing noise.
///    - Like understanding that location, size, and condition affect house prices
///    - Signs: Good performance on both training data and new data
/// 
/// This interface provides methods to analyze your model's performance and detect which
/// of these situations you're dealing with, so you can make appropriate adjustments.
/// </remarks>
public interface IFitDetector<T, TInput, TOutput>
{
    /// <summary>
    /// Analyzes model evaluation data to detect whether the model is underfitting, overfitting, or has a good fit.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics on training and validation datasets.</param>
    /// <returns>A FitDetectorResult object containing the detected fit type and relevant metrics.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method examines how your model performs on different data sets to diagnose problems.
    /// 
    /// The input parameter:
    /// - evaluationData: Contains information about how well your model performs on:
    ///   - Training data (the data your model learned from)
    ///   - Validation data (new data your model hasn't seen before)
    /// 
    /// The method compares these performances to determine if your model:
    /// - Is underfitting (performs poorly on both training and validation data)
    /// - Is overfitting (performs very well on training data but poorly on validation data)
    /// - Has a good fit (performs well on both training and validation data)
    /// 
    /// The returned FitDetectorResult contains:
    /// - The detected fit type (underfitting, overfitting, or good fit)
    /// - Relevant metrics that support this conclusion
    /// - Possibly recommendations for how to improve the model
    /// 
    /// This information helps you decide what to do next:
    /// - If underfitting: Try a more complex model or add more features
    /// - If overfitting: Try regularization, get more training data, or simplify the model
    /// - If good fit: Your model is ready to use!
    /// </remarks>
    FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData);
}
