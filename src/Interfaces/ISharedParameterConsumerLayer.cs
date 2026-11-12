using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Optional contract for layers that reference trainable parameter tensors owned by another layer
/// (such as a tied output head reading an embedding layer's table, or a shared-substrate child block
/// referencing a root block's weights) without re-registering them as owned parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISharedParameterConsumerLayer<T>
{
    /// <summary>
    /// Returns external trainable parameter tensors referenced during this layer's forward/backward pass.
    /// </summary>
    IReadOnlyList<Tensor<T>> GetSharedTrainableParameters();
}
