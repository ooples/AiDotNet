namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for strategies that combine predictions from multiple models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A combination strategy is the method used to merge predictions 
/// from multiple models into a single final prediction. Think of it as the rule for 
/// combining different expert opinions into one decision.
/// </para>
/// <para>
/// Different strategies work better for different situations. For example:
/// - Averaging works well when all models are similarly good
/// - Weighted averaging gives more importance to better models
/// - Voting is good for classification problems
/// - Stacking can learn complex combinations
/// </para>
/// </remarks>
public interface ICombinationStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Combines multiple predictions into a single prediction.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights assigned to each model's prediction.</param>
    /// <returns>The combined prediction.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes all the different predictions and merges 
    /// them according to the strategy's rules. The weights determine how much each 
    /// prediction contributes to the final result.
    /// </remarks>
    TOutput Combine(List<TOutput> predictions, Vector<T> weights);
    
    /// <summary>
    /// Validates if the predictions can be combined using this strategy.
    /// </summary>
    /// <param name="predictions">The list of predictions to validate.</param>
    /// <returns>True if the predictions can be combined; false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if all predictions have compatible shapes and 
    /// types that can be combined. For example, you can't average predictions if they 
    /// have different dimensions.
    /// </remarks>
    bool CanCombine(List<TOutput> predictions);
    
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    /// <value>A descriptive name for the strategy.</value>
    string StrategyName { get; }
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    /// <value>True if weights need to be learned; false if they can be set directly.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Some strategies like simple averaging don't need special 
    /// training, while others like stacking need to learn how to combine predictions.
    /// </remarks>
    bool RequiresTraining { get; }
}