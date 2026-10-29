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
        => TryForward(conv, norm, input, relu ? FusedActivationType.ReLU : FusedActivationType.None, out output);

    /// <summary>
    /// <c>act(bn(conv(input)))</c> as one convolution with the folded weights, for any activation the engine fuses into
    /// a convolution (a ReLU6 for MobileNetV2's expansion stage). False, with nothing computed, under the same
    /// conditions as the ReLU overload.
    /// </summary>
    internal static bool TryForward(ConvolutionalLayer<T> conv, BatchNormalizationLayer<T> norm, Tensor<T> input,
        FusedActivationType activation, out Tensor<T> output)
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
            activation);
        return true;
    }

    /// <summary>
    /// <c>act(bn(input))</c> in one pass over the activation (AiDotNet.Tensors' BatchNormInference with an activation),
    /// where a batch norm cannot fold into the convolution before it: a depthwise convolution, say. False, with nothing
    /// computed, when folding is off, the layer is training, a tape is recording, a graph is being captured, shapes are
    /// being inferred, an observer is attached, or the input is not rank 4.
    /// </summary>
    internal static bool TryBatchNormActivation(BatchNormalizationLayer<T> norm, Tensor<T> input,
        FusedActivationType activation, out Tensor<T> output)
    {
        output = input;
        if (!InferenceFusionSettings.FoldBatchNormIntoConvolution) return false;
        // BatchNormInference reads the activation as one contiguous span (as BatchNormalizationLayer's own fused path
        // requires); a strided view takes the layer's unfused path instead.
        if (input.Rank != 4 || !input.IsContiguous || norm.InTrainingMode) return false;
        if (AiDotNet.Autodiff.GradientRecording.IsRecording<T>() || AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
            return false;
        if (LayerBase<T>.IsInferringShapes) return false;
        if (!norm.TryGetFoldingStatistics(input.Shape[1], out var gamma, out var beta, out var mean, out var variance,
                out double epsilon))
            return false;
        if (LayerForwardObserver<T>.Current is { } observer)
        {
            observer.NoteSuppressedFolding();
            return false;
        }

        output = AiDotNetEngine.Current.BatchNormInference(input, gamma, beta, mean, variance, epsilon, activation);
        return true;
    }

    /// <summary>
    /// <c>act(bn(depthwise(input) + bias))</c> as a bias-free depthwise convolution and one batch-norm pass with the bias
    /// folded into its mean (mean - bias): the bias add and the activation are no separate passes. False, with nothing
    /// computed, under the conditions of <see cref="TryBatchNormActivation"/> or when the convolution is not a depthwise
    /// one without an activation of its own (<see cref="ConvolutionalLayer{T}.CanFoldBiasIntoFollowingBatchNorm"/>).
    /// </summary>
    internal static bool TryDepthwiseBatchNormActivation(ConvolutionalLayer<T> conv, BatchNormalizationLayer<T> norm,
        Tensor<T> input, FusedActivationType activation, out Tensor<T> output)
    {
        output = input;
        if (!InferenceFusionSettings.FoldBatchNormIntoConvolution) return false;
        if (input.Rank != 4 || conv.InTrainingMode || norm.InTrainingMode) return false;
        if (AiDotNet.Autodiff.GradientRecording.IsRecording<T>() || AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
            return false;
        if (LayerBase<T>.IsInferringShapes) return false;
        if (!conv.CanFoldBiasIntoFollowingBatchNorm || input.Shape[1] != conv.InputDepth) return false;
        if (!norm.TryGetFoldingStatistics(conv.OutputDepth, out var gamma, out var beta, out var mean, out var variance,
                out double epsilon))
            return false;
        if (LayerForwardObserver<T>.Current is { } observer)
        {
            observer.NoteSuppressedFolding();
            return false;
        }

        var engine = AiDotNetEngine.Current;
        var depthwise = engine.DepthwiseConv2D(input, conv.KernelsForFolding, new[] { conv.Stride, conv.Stride },
            new[] { conv.Padding, conv.Padding });
        var shiftedMean = conv.BiasesForFolding is { } bias ? engine.TensorSubtract(mean, bias) : mean;
        output = engine.BatchNormInference(depthwise, gamma, beta, shiftedMean, variance, epsilon, activation);
        return true;
    }
    /// <summary>
    /// The engine-fusable form of <paramref name="activation"/>: None for no activation, ReLU and ReLU6 for exactly those
    /// activation types (a subclass may compute something else, so it does not qualify). False otherwise.
    /// </summary>
    internal static bool TryGetFusedActivation(IActivationFunction<T>? activation, out FusedActivationType fused)
    {
        fused = FusedActivationType.None;
        if (activation is null) return true;
        if (activation.GetType() == typeof(ReLU6Activation<T>)) { fused = FusedActivationType.ReLU6; return true; }
        if (activation.GetType() == typeof(ReLUActivation<T>)) { fused = FusedActivationType.ReLU; return true; }
        return false;
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
