namespace AiDotNet.Interfaces;

/// <summary>
/// Implemented by models whose loss function can be replaced by the caller after construction.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <remarks>
/// <para>
/// This is the seam <c>AiModelBuilder.ConfigureLossFunction</c> uses. Setting the loss on the
/// <i>model</i> — rather than somewhere in the facade — is deliberate: the loss must reach the
/// component that computes gradients, and the optimizer already adopts a model's
/// <see cref="IFullModel{T,TInput,TOutput}.DefaultLossFunction"/> when the model is attached
/// (<c>GradientBasedOptimizerBase.OnModelChanged</c>). Injecting here therefore reaches both the
/// model's own training path and the optimizer's, and keeps the loss reported in metrics identical
/// to the loss actually optimized.
/// </para>
/// <para>
/// <b>Not every model can implement this.</b> Where the loss is intrinsic to the architecture —
/// DeepAR's Gaussian likelihood over a (mean, scale) head, or the Temporal Fusion Transformer's
/// quantile pinball loss, which defines the shape of its output — an arbitrary
/// <see cref="ILossFunction{T}"/> is not merely suboptimal but incorrect, and the model should not
/// implement this interface. The facade reports that as an error rather than silently corrupting
/// the model.
/// </para>
/// <para>
/// <b>For Beginners:</b> The loss function is how a model measures how wrong it is. Most models let
/// you pick one. A few are built around a specific loss and cannot be given a different one — those
/// don't implement this interface, so asking for a different loss tells you so instead of quietly
/// doing nothing.
/// </para>
/// </remarks>
public interface ISupportsLossFunction<T>
{
    /// <summary>
    /// Replaces the loss function this model trains against.
    /// </summary>
    /// <param name="lossFunction">The loss to use. Must not be null.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown once the model has been trained — the loss is locked after training so that the loss reported in
    /// metrics matches the loss actually optimized; construct a new model to train against a different loss.
    /// </exception>
    void SetLossFunction(ILossFunction<T> lossFunction);
}
