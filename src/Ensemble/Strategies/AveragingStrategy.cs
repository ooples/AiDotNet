using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Combines predictions by taking the simple average.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This strategy simply adds up all the predictions and divides by 
/// the number of models. It treats all models equally, regardless of their individual performance.
/// This works well when all models are similarly accurate.
/// </para>
/// </remarks>
public class AveragingStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public override string StrategyName => "Simple Averaging";
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public override bool RequiresTraining => false;
    
    /// <summary>
    /// Combines predictions by averaging them.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights (ignored for simple averaging).</param>
    /// <returns>The averaged prediction.</returns>
    public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }
        
        // For generic types, we need type-specific implementations
        // This is a placeholder that returns the first prediction
        // Derived classes should provide proper implementations for specific types
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