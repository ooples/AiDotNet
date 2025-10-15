namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that support pruning compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IPrunableModel<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a pruned version of this model.
    /// </summary>
    /// <param name="targetSparsity">The target sparsity level (0-1) as a value of type T.</param>
    /// <param name="structured">Whether to use structured pruning.</param>
    /// <returns>A pruned version of the model.</returns>
    TModel Prune(T targetSparsity, bool structured);
}