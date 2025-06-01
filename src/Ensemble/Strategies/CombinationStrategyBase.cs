using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Base class for combination strategies providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
public abstract class CombinationStrategyBase<T, TInput, TOutput> : ICombinationStrategy<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    
    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public abstract string StrategyName { get; }
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public abstract bool RequiresTraining { get; }
    
    /// <summary>
    /// Initializes a new instance of the CombinationStrategyBase class.
    /// </summary>
    protected CombinationStrategyBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }
    
    /// <summary>
    /// Combines multiple predictions into a single prediction.
    /// </summary>
    public abstract TOutput Combine(List<TOutput> predictions, Vector<T> weights);
    
    /// <summary>
    /// Validates if the predictions can be combined using this strategy.
    /// </summary>
    public abstract bool CanCombine(List<TOutput> predictions);
    
    /// <summary>
    /// Normalizes weights so they sum to 1.
    /// </summary>
    protected Vector<T> NormalizeWeights(Vector<T> weights)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < weights.Length; i++)
        {
            sum = NumOps.Add(sum, weights[i]);
        }
        
        if (NumOps.Equals(sum, NumOps.Zero))
        {
            // If all weights are zero, use uniform weights
            var uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(weights.Length));
            var uniformWeights = new T[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                uniformWeights[i] = uniformWeight;
            }
            return new Vector<T>(uniformWeights);
        }
        
        var normalized = new T[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            normalized[i] = NumOps.Divide(weights[i], sum);
        }
        
        return new Vector<T>(normalized);
    }
    
}