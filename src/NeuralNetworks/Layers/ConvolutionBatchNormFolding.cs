using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Folds a batch normalization, and a ReLU after it, into the preceding convolution in inference
/// (see <see cref="InferenceFusionSettings.FoldBatchNormIntoConvolution"/>).
/// </summary>
/// <remarks>
/// The engine caches the folded weights and bias by the convolution's kernel and rebuilds them when any source tensor's
/// identity or <see cref="Tensor{T}.Version"/> changes, which every in-place write advances (an optimizer step, a
/// <c>SetParameters</c>, a deserialization). A clone is a new layer, so it folds its own weights.
/// </remarks>
internal static class ConvolutionBatchNormFolding<T>
{
    /// <summary>
    /// <c>act(bn(conv(input)))</c> as one convolution with the folded weights, <paramref name="relu"/> selecting a ReLU
    /// after the batch norm. False, with nothing computed, when the pair cannot fold here: folding is off, either layer
    /// is training, a gradient tape is recording, a graph is being captured, a forward observer is attached, the
    /// convolution is grouped or has its own activation, a layer has not resolved its shape yet (its first forward runs
    /// unfused and allocates it), or the input is not rank 4.
    /// </summary>
    internal static bool TryForward(ConvolutionalLayer<T> conv, BatchNormalizationLayer<T> norm, Tensor<T> input,
        bool relu, out Tensor<T> output)
    {
        output = input;
        if (!InferenceFusionSettings.FoldBatchNormIntoConvolution) return false;
        if (input.Rank != 4 || conv.InTrainingMode || norm.InTrainingMode) return false;
        if (AiDotNet.Autodiff.GradientRecording.IsRecording<T>() || AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
            return false;
        if (LayerBase<T>.IsInferringShapes) return false;
        if (!conv.CanFoldFollowingBatchNorm || input.Shape[1] != conv.InputDepth) return false;
        if (!norm.TryGetFoldingStatistics(conv.OutputDepth, out var gamma, out var beta, out var mean, out var variance,
                out double epsilon))
            return false;
        // Last, so the observer learns only of a fold that would otherwise have run: a traced forward
        // that suppressed one differs from an untraced forward by rounding (NeuralNetworkBase re-runs it).
        if (LayerForwardObserver<T>.Current is { } observer)
        {
            observer.NoteSuppressedFolding();
            return false;
        }

        // The engine folds and caches (AiDotNet.Tensors' FusedConv2DBatchNorm, ooples/AiDotNet.Tensors#1432): rebuilt when a
        // source tensor's identity or version changes, and on a GPU folded on the device when the parameters live only
        // there (after on-device training), instead of downloading all six to fold them on the host.
        output = AiDotNetEngine.Current.FusedConv2DBatchNorm(input, conv.KernelsForFolding, conv.BiasesForFolding,
            gamma, beta, mean, variance, epsilon, conv.Stride, conv.Stride, conv.Padding, conv.Padding, 1, 1,
            relu ? FusedActivationType.ReLU : FusedActivationType.None);
        return true;
    }

    /// <summary>
    /// For a sequential layer walk: when <paramref name="layers"/>[<paramref name="index"/>] is a convolution followed
    /// by a batch normalization (and optionally a ReLU <see cref="ActivationLayer{T}"/>), runs them folded and reports
    /// how many layers it consumed (2 or 3). False, with nothing computed, otherwise.
    /// </summary>
    internal static bool TryForwardSequence(IReadOnlyList<ILayer<T>> layers, int index, Tensor<T> input,
        out Tensor<T> output, out int consumed)
    {
        output = input;
        consumed = 0;
        if (index + 1 >= layers.Count
            || layers[index] is not ConvolutionalLayer<T> conv
            || layers[index + 1] is not BatchNormalizationLayer<T> norm)
            return false;
        bool relu = index + 2 < layers.Count
            && layers[index + 2] is ActivationLayer<T> { VectorActivation: null, ScalarActivation: ReLUActivation<T> } activation
            && !activation.InTrainingMode;
        if (!TryForward(conv, norm, input, relu, out output)) return false;
        consumed = relu ? 3 : 2;
        return true;
    }
}
