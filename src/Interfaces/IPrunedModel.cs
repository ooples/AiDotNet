using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have been pruned.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IPrunedModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
    where T : unmanaged
{
    /// <summary>
    /// Gets the overall sparsity level of the model.
    /// </summary>
    T SparsityLevel { get; }
    
    /// <summary>
    /// Gets whether structured pruning was used.
    /// </summary>
    bool IsStructuredPruning { get; }
    
    /// <summary>
    /// Gets the sparsity level for each layer.
    /// </summary>
    Vector<T> GetLayerSparsityLevels();
    
    /// <summary>
    /// Gets the pruning mask for a specific layer.
    /// </summary>
    /// <param name="layerIndex">The index of the layer.</param>
    /// <returns>The pruning mask for the layer as a tensor.</returns>
    Tensor<T> GetPruningMask(int layerIndex);
}