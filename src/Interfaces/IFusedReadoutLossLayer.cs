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

/// <summary>
/// Internal pre-head readout layer contract (e.g. final RMSNorm or LayerNorm) allowing hidden-layer local objectives
/// to evaluate the shared readout on the same tape while detaching the normalization layer's own parameters.
/// </summary>
internal interface IDetachablePreHeadLayer<T>
{
    bool CanForwardWithDetachedParameters { get; }

    bool PreservesIndependentStackedBatches(int inputRank);

    Tensor<T> ForwardWithDetachedParameters(Tensor<T> input, ref object? detachedState);
}

/// <summary>
/// Internal extension of <see cref="IFusedReadoutLossLayer{T}"/> that prepares integer class-index targets once per
/// step, supports parameter-detached evaluation across intermediate hidden layers, and exposes vocabulary rows for
/// sampled readout local credit without materializing full <c>[rows, vocabulary]</c> logits.
/// </summary>
internal interface IDetachableFusedReadoutLossLayer<T> : IFusedReadoutLossLayer<T>
{
    Tensor<int>? TryPrepareClassIndexTarget(Tensor<T> target, ILossFunction<T> loss, int inputRank, int[] inputShape, out int vocabularySize);

    Tensor<T>? TryComputeFusedLossPrepared(
        Tensor<T> input,
        Tensor<int> ids,
        bool detachParameters,
        ref Tensor<T>? tracedWeight,
        ref Tensor<T>? detachedWeight,
        ref Tensor<T>? detachedBias);

    bool TryGetVocabularyRowsForSampling(out Tensor<T> vocabRows, out Tensor<T>? bias, out bool vocabRowsAreInputMajor);
}
