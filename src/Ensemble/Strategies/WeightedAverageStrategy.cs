using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Combines predictions using a weighted average based on model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This strategy gives more importance to better-performing models. 
/// Each prediction is multiplied by its model's weight before averaging. A model with 
/// weight 2.0 contributes twice as much to the final prediction as a model with weight 1.0.
/// </para>
/// </remarks>
public class WeightedAverageStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public override string StrategyName => "Weighted Average";
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public override bool RequiresTraining => false;
    
    /// <summary>
    /// Combines predictions using weighted averaging.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights for each model.</param>
    /// <returns>The weighted average prediction.</returns>
    public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }
        
        if (weights.Length != predictions.Count)
        {
            throw new ArgumentException(
                $"Number of weights ({weights.Length}) must match number of predictions ({predictions.Count})");
        }
        
        // For generic types, we need type-specific implementations
        // This is a placeholder that returns the prediction with highest weight
        var normalizedWeights = NormalizeWeights(weights);
        
        // Find index of highest weight
        int maxIndex = 0;
        for (int i = 1; i < normalizedWeights.Length; i++)
        {
            if (NumOps.GreaterThan(normalizedWeights[i], normalizedWeights[maxIndex]))
            {
                maxIndex = i;
            }
        }
        
        return predictions[maxIndex];
    }
    
    /// <summary>
    /// Validates if the predictions can be combined.
    /// </summary>
    public override bool CanCombine(List<TOutput> predictions)
    {
        return predictions != null && predictions.Count > 0;
    }
}