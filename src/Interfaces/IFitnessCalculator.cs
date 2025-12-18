namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for calculating how well a machine learning model performs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface helps measure how "fit" or effective your machine learning model is.
/// 
/// In machine learning, we need a way to measure how good our model is at making predictions.
/// This is similar to how we might grade a test - we need a scoring system.
/// 
/// Different types of problems need different scoring methods:
/// - For predicting house prices, we might measure how close our predictions are to actual prices
/// - For classifying emails as spam/not spam, we might count how many emails we classified correctly
/// 
/// The "fitness score" is this measurement of how well the model performs. Some important points:
/// - Sometimes higher scores are better (like accuracy: 95% is better than 90%)
/// - Sometimes lower scores are better (like error: 5% error is better than 10% error)
/// - The score helps us compare different models to choose the best one
/// 
/// This interface provides methods to calculate these fitness scores in a standardized way.
/// </remarks>
public interface IFitnessCalculator<T, TInput, TOutput>
{
    /// <summary>
    /// Calculates a fitness score based on comprehensive model evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics on training and validation datasets.</param>
    /// <returns>A fitness score representing how well the model performs.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method grades your model based on detailed performance information.
    /// 
    /// The input parameter:
    /// - evaluationData: Contains various measurements about how your model performed on different data
    ///   This typically includes information from both:
    ///   - Training data (data your model learned from)
    ///   - Validation data (new data your model hasn't seen before)
    /// 
    /// The method processes these measurements and returns a single number (the fitness score)
    /// that represents how good your model is. This makes it easy to compare different models
    /// or different versions of the same model.
    /// 
    /// For example, if you're predicting house prices, this might calculate the average 
    /// percentage error in your predictions compared to actual prices.
    /// </remarks>
    T CalculateFitnessScore(ModelEvaluationData<T, TInput, TOutput> evaluationData);

    /// <summary>
    /// Calculates a fitness score based on basic dataset statistics.
    /// </summary>
    /// <param name="dataSet">Statistical information about model performance on a dataset.</param>
    /// <returns>A fitness score representing how well the model performs.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method grades your model based on simplified performance statistics.
    /// 
    /// The input parameter:
    /// - dataSet: Contains basic statistics about how your model performed
    ///   This is a simpler version of the evaluation data used in the other method
    /// 
    /// This method is useful when you only have access to summary statistics rather than
    /// detailed evaluation data. It still produces a fitness score that measures how good
    /// your model is, just using less detailed information.
    /// 
    /// For example, if you're building a spam filter, this might calculate accuracy
    /// based on how many emails were correctly classified as spam or not spam.
    /// </remarks>
    T CalculateFitnessScore(DataSetStats<T, TInput, TOutput> dataSet);

    /// <summary>
    /// Indicates whether higher fitness scores represent better performance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property tells you whether bigger numbers mean better performance.
    /// 
    /// Different metrics work in opposite ways:
    /// - For some metrics like accuracy, higher is better (95% accuracy is better than 90%)
    /// - For other metrics like error rate, lower is better (5% error is better than 10%)
    /// 
    /// This property returns:
    /// - true: if higher scores mean better performance (like accuracy)
    /// - false: if lower scores mean better performance (like error)
    /// 
    /// Knowing this helps you correctly interpret and compare fitness scores.
    /// </remarks>
    bool IsHigherScoreBetter { get; }

    /// <summary>
    /// Compares two fitness scores and determines if the current score is better than the best score so far.
    /// </summary>
    /// <param name="currentFitness">The fitness score to evaluate.</param>
    /// <param name="bestFitness">The best fitness score found so far.</param>
    /// <returns>True if the current fitness score is better than the best fitness score; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method compares two scores and tells you if the new one is better.
    /// 
    /// When comparing models, you need to know which one performs better. This method handles
    /// the comparison logic for you, taking into account whether higher or lower scores are better.
    /// 
    /// The input parameters:
    /// - currentFitness: The new score you want to evaluate
    /// - bestFitness: The best score you've seen so far
    /// 
    /// The method returns:
    /// - true: if the current fitness is better than the best fitness
    /// - false: if the current fitness is worse than or equal to the best fitness
    /// 
    /// This is particularly useful in algorithms that try many different models and need
    /// to keep track of which one is best.
    /// </remarks>
    bool IsBetterFitness(T currentFitness, T bestFitness);
}
