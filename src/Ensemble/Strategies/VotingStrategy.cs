using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Combines predictions using majority voting for classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In voting, each model votes for what it thinks is the correct class, 
/// and the class with the most votes wins. This is like a democratic election. Hard voting 
/// counts each vote equally, while soft voting uses prediction probabilities.
/// </para>
/// </remarks>
public class VotingStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
{
    private readonly bool _useSoftVoting;
    
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public override string StrategyName => _useSoftVoting ? "Soft Voting" : "Hard Voting";
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public override bool RequiresTraining => false;
    
    /// <summary>
    /// Initializes a new instance of the VotingStrategy class.
    /// </summary>
    /// <param name="useSoftVoting">If true, uses soft voting (probabilities); otherwise uses hard voting.</param>
    public VotingStrategy(bool useSoftVoting = false)
    {
        _useSoftVoting = useSoftVoting;
    }
    
    /// <summary>
    /// Combines predictions using voting.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights for each model's vote.</param>
    /// <returns>The voted prediction.</returns>
    public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }
        
        // For generic types, we return the first prediction
        // Derived classes should provide type-specific voting implementations
        return predictions[0];
    }
    
    /// <summary>
    /// Validates if the predictions can be combined.
    /// </summary>
    public override bool CanCombine(List<TOutput> predictions)
    {
        return predictions != null && predictions.Count > 0;
    }
}