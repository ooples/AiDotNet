using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
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
/// The folded weights and bias are cached per convolution and rebuilt when any source tensor changes: each is held by
/// reference and checked by <see cref="Tensor{T}.Version"/>, which every in-place write advances (an optimizer step, a
/// <c>SetParameters</c>, a deserialization). A clone is a new layer, so it folds its own weights.
/// </remarks>
internal static class ConvolutionBatchNormFolding<T>
{
    private sealed class Folded
    {
        internal BatchNormalizationLayer<T>? Norm;
        internal Tensor<T>? Kernels, Biases, Gamma, Beta, Mean, Variance;
        internal int KernelsVersion, BiasesVersion, GammaVersion, BetaVersion, MeanVersion, VarianceVersion;
        internal double Epsilon;
        internal Tensor<T>? FoldedKernels, FoldedBiases;
    }

    private static readonly ConditionalWeakTable<ConvolutionalLayer<T>, Folded> s_folded = new();

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

        var kernels = conv.KernelsForFolding;
        var biases = conv.BiasesForFolding;
        var (foldedKernels, foldedBiases) = GetFolded(conv, norm, kernels, biases, gamma, beta, mean, variance, epsilon);
        output = AiDotNetEngine.Current.FusedConv2D(input, foldedKernels, foldedBiases, conv.Stride, conv.Stride, conv.Padding,
            conv.Padding, 1, 1, relu ? FusedActivationType.ReLU : FusedActivationType.None);
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

    private static (Tensor<T> Kernels, Tensor<T> Biases) GetFolded(ConvolutionalLayer<T> conv,
        BatchNormalizationLayer<T> norm, Tensor<T> kernels, Tensor<T>? biases, Tensor<T> gamma, Tensor<T> beta,
        Tensor<T> mean, Tensor<T> variance, double epsilon)
    {
        var entry = s_folded.GetValue(conv, static _ => new Folded());
        lock (entry)
        {
            if (entry.FoldedKernels is { } cachedKernels && entry.FoldedBiases is { } cachedBiases
                && ReferenceEquals(entry.Norm, norm)
                && BitConverter.DoubleToInt64Bits(entry.Epsilon) == BitConverter.DoubleToInt64Bits(epsilon)
                && Same(entry.Kernels, entry.KernelsVersion, kernels) && SameOrNull(entry.Biases, entry.BiasesVersion, biases)
                && Same(entry.Gamma, entry.GammaVersion, gamma) && Same(entry.Beta, entry.BetaVersion, beta)
                && Same(entry.Mean, entry.MeanVersion, mean) && Same(entry.Variance, entry.VarianceVersion, variance))
                return (cachedKernels, cachedBiases);

            var ops = MathHelper.GetNumericOperations<T>();
            int channels = conv.OutputDepth;
            int perChannel = kernels.Length / channels;
            var w = kernels.ToArray();
            var b = biases?.ToArray();
            var g = gamma.ToArray();
            var bt = beta.ToArray();
            var m = mean.ToArray();
            var v = variance.ToArray();
            var foldedW = new T[w.Length];
            var foldedB = new T[channels];
            // In double, then rounded once to T: the folded weights are as close to the exact products as T allows.
            for (int c = 0; c < channels; c++)
            {
                double scale = ops.ToDouble(g[c]) / System.Math.Sqrt(ops.ToDouble(v[c]) + epsilon);
                for (int i = c * perChannel, end = i + perChannel; i < end; i++)
                    foldedW[i] = ops.FromDouble(ops.ToDouble(w[i]) * scale);
                double bias = b is null ? 0.0 : ops.ToDouble(b[c]);
                foldedB[c] = ops.FromDouble((bias - ops.ToDouble(m[c])) * scale + ops.ToDouble(bt[c]));
            }

            entry.Norm = norm;
            entry.Epsilon = epsilon;
            entry.Kernels = kernels; entry.KernelsVersion = kernels.Version;
            entry.Biases = biases; entry.BiasesVersion = biases?.Version ?? 0;
            entry.Gamma = gamma; entry.GammaVersion = gamma.Version;
            entry.Beta = beta; entry.BetaVersion = beta.Version;
            entry.Mean = mean; entry.MeanVersion = mean.Version;
            entry.Variance = variance; entry.VarianceVersion = variance.Version;
            entry.FoldedKernels = new Tensor<T>(foldedW, kernels.Shape.ToArray());
            entry.FoldedBiases = new Tensor<T>(foldedB, new[] { channels });
            return (entry.FoldedKernels, entry.FoldedBiases);
        }
    }

    private static bool Same(Tensor<T>? cached, int version, Tensor<T> current)
        => ReferenceEquals(cached, current) && current.Version == version;

    private static bool SameOrNull(Tensor<T>? cached, int version, Tensor<T>? current)
        => current is null ? cached is null : Same(cached, version, current);
}
