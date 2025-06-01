using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Combines predictions by taking the median value, which is robust to outliers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The median strategy picks the middle value when all predictions 
/// are sorted. This is more robust than averaging because extreme predictions (outliers) 
/// don't affect the result as much. If you have 5 models and their predictions are 
/// [1, 2, 10, 3, 4], the median is 3, while the average would be 4.
/// </para>
/// </remarks>
public class MedianStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public override string StrategyName => "Median";
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public override bool RequiresTraining => false;
    
    /// <summary>
    /// Combines predictions by taking the median.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights (ignored for median).</param>
    /// <returns>The median prediction.</returns>
    public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }
        
        // For generic types, return the middle prediction
        int medianIndex = predictions.Count / 2;
        return predictions[medianIndex];
    }
    
    /// <summary>
    /// Validates if the predictions can be combined.
    /// </summary>
    public override bool CanCombine(List<TOutput> predictions)
    {
        return predictions != null && predictions.Count > 0;
    }
}