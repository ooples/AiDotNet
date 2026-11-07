using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// A readout layer that can compute <c>loss(Forward(input), target)</c> as one fused, differentiable operation without
/// materialising its output on the autodiff tape - for example a language-model head fused with cross-entropy
/// (<c>IEngine.FusedLinearCrossEntropyWithLogits</c>), whose output is a <c>[rows, vocabulary]</c> logits tensor.
/// Local-objective credit rules evaluate the readout once per hidden layer, so the decomposed path keeps one such
/// tensor (and its softmax intermediates) per layer alive until the backward pass.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface IFusedReadoutLossLayer<T>
{
    /// <summary>
    /// Returns the scalar loss equal to <c>loss.ComputeTapeLoss(Forward(input), target)</c> reduced to a scalar,
    /// computed by a fused operation recorded on the current tape, or <c>null</c> when this loss / target combination
    /// is not supported (the caller then falls back to the decomposed path).
    /// </summary>
    /// <param name="input">The layer input.</param>
    /// <param name="target">The training target for the layer output.</param>
    /// <param name="loss">The loss function the caller would apply to the layer output.</param>
    /// <returns>The fused scalar loss, or null to fall back.</returns>
    Tensor<T>? TryComputeFusedLoss(Tensor<T> input, Tensor<T> target, ILossFunction<T> loss);
}
