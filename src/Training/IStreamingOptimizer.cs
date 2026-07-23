using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Optimizer-in-backward contract for the memory-bounded streaming training path.
/// </summary>
internal interface IStreamingOptimizer<T>
{
    void BeginStep();
    void Apply(Tensor<T> param, Tensor<T> grad);

    /// <summary>
    /// Called once after every parameter's gradient has been handed to <see cref="Apply"/> for
    /// the current step. First-order optimizers update in place during <see cref="Apply"/> and
    /// no-op here; full-gradient (second-order) optimizers like streaming L-BFGS buffer the
    /// per-parameter gradients and perform their global update here.
    /// </summary>
    void EndStep();
}

