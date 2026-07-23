using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single NVIDIA Citrinet residual mega-block over <c>[B, C, T]</c> feature data
/// (Majumdar et al., 2021, "Citrinet: Closing the Gap between Non-Autoregressive and
/// Autoregressive End-to-End Models for ASR", https://arxiv.org/abs/2104.01721).
/// </summary>
/// <remarks>
/// <para>Implements the paper's block exactly:</para>
/// <code>
/// residual = BN(PointwiseConv1x1(x, stride))                 # projected skip path
/// h = x
/// for r in 0..R-1:                                           # R time-channel separable sub-blocks
///     h = DepthwiseConv1d(h, K, stride if r==0 else 1)       #   temporal (per-channel) conv
///     h = PointwiseConv1x1(h)                                #   channel-mixing conv
///     h = BN(h)
///     if r &lt; R-1: h = dropout(ReLU(h))                       #   activation between sub-blocks
/// h = SqueezeExcitation(h)                                   # channel attention (time-pooled)
/// h = ReLU(h + residual)                                     # residual add + activation
/// h = dropout(h)
/// </code>
/// <para>
/// The 1-D time-channel separable convolution (depthwise-temporal + pointwise-channel), the
/// squeeze-and-excitation channel attention, and the residual connection are Citrinet's three
/// defining features — a plain 1-D conv stack (no separability, no SE, no residual) is NOT
/// Citrinet. The residual path carries the signal through the deep block stack so gradients flow;
/// channel width <c>C</c> is constant across the block (the residual add requires <c>C_out == C_in</c>).
/// </para>
/// <para>
/// Composed from tape-aware inner layers (<see cref="DepthwiseConv1DLayer{T}"/>,
/// <see cref="Conv1DLayer{T}"/>, <see cref="BatchNormalizationLayer{T}"/>,
/// <see cref="SqueezeAndExcitationLayer{T}"/>) and <see cref="IEngine"/> ops, so the gradient tape
/// autodiffs the block — no hand-written backward. The block is fully reconstructable from
/// <c>(channels, kernelSize, numSubBlocks, seReductionRatio, dropoutRate, stride)</c> for
/// Clone/Deserialize.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 16", TestConstructorArgs = "8, 3, 2")]
public partial class CitrinetBlockLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _numSubBlocks;
    private readonly int _seReductionRatio;
    private readonly double _dropoutRate;
    private readonly int _stride;

    // Residual (skip) path: 1x1 pointwise conv (+ optional stride) followed by BN.
    private readonly Conv1DLayer<T> _residualConv;
    private readonly BatchNormalizationLayer<T> _residualBn;

    // Main path: R time-channel separable sub-blocks.
    private readonly DepthwiseConv1DLayer<T>[] _depthwise;
    private readonly Conv1DLayer<T>[] _pointwise;
    private readonly BatchNormalizationLayer<T>[] _bn;

    // Channel attention (applied in channels-last layout via transpose).
    private readonly SqueezeAndExcitationLayer<T> _se;

    // Dropout between sub-blocks and after the residual add (null when dropoutRate == 0).
    private readonly DropoutLayer<T>? _dropout;

    /// <summary>Constructs a Citrinet residual mega-block.</summary>
    /// <param name="channels">Constant channel width <c>C</c> (input and output).</param>
    /// <param name="kernelSize">Depthwise temporal kernel width <c>K</c> (Citrinet grows this per block, e.g. 5..39).</param>
    /// <param name="numSubBlocks">Number of time-channel separable sub-blocks <c>R</c> (Citrinet uses 5).</param>
    /// <param name="seReductionRatio">Squeeze-and-excitation reduction ratio. Defaults to 8.</param>
    /// <param name="dropoutRate">Dropout rate applied between sub-blocks and after the residual add. Defaults to 0.</param>
    /// <param name="stride">Temporal stride for this block's subsampling (Citrinet uses 2 at group boundaries). Defaults to 1.</param>
    /// <param name="seed">Base seed for deterministic, order-independent weight init.</param>
    public CitrinetBlockLayer(
        int channels,
        int kernelSize,
        int numSubBlocks = 5,
        int seReductionRatio = 8,
        double dropoutRate = 0.0,
        int stride = 1,
        int seed = 1009)
        : base(new[] { channels, -1 }, new[] { channels, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (numSubBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numSubBlocks));
        if (seReductionRatio <= 0) throw new ArgumentOutOfRangeException(nameof(seReductionRatio));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        _channels = channels;
        _kernelSize = kernelSize;
        _numSubBlocks = numSubBlocks;
        _seReductionRatio = seReductionRatio;
        _dropoutRate = dropoutRate;
        _stride = stride;

        // Deterministic per-position seeds so weight init is order-independent — inner layers are
        // hidden from LayerHelper's per-layer RandomSeed wiring, and an unseeded shared-RNG init
        // would make training depend on run/test order and flake MoreData_ShouldNotDegrade.
        int s = unchecked(channels * 131 + kernelSize * 17 + numSubBlocks * 7 + stride * 3 + seed);
        IInitializationStrategy<T> Init(int i) => new HeInitializationStrategy<T>(RandomHelper.CreateSeededRandom(unchecked(s + i)));

        // Residual projection: 1x1 pointwise conv (downsamples T when stride > 1) + BN.
        _residualConv = new Conv1DLayer<T>(channels, channels, kernelSize: 1, dilation: 1, stride: stride,
            padding: 0, activation: new IdentityActivation<T>(), initializationStrategy: Init(1));
        _residualBn = new BatchNormalizationLayer<T>(channels);

        _depthwise = new DepthwiseConv1DLayer<T>[numSubBlocks];
        _pointwise = new Conv1DLayer<T>[numSubBlocks];
        _bn = new BatchNormalizationLayer<T>[numSubBlocks];
        for (int r = 0; r < numSubBlocks; r++)
        {
            int subStride = r == 0 ? stride : 1;
            _depthwise[r] = new DepthwiseConv1DLayer<T>(channels, kernelSize, multiplier: 1, stride: subStride,
                padding: null, activation: new IdentityActivation<T>(), initializationStrategy: Init(10 + r * 3));
            _pointwise[r] = new Conv1DLayer<T>(channels, channels, kernelSize: 1, dilation: 1, stride: 1,
                padding: 0, activation: new IdentityActivation<T>(), initializationStrategy: Init(11 + r * 3));
            _bn[r] = new BatchNormalizationLayer<T>(channels);
        }

        // Paper SE: ReLU on the reduced FC, sigmoid on the expand FC. Passing explicit scalar
        // activations also disambiguates the scalar/vector-activation constructor overloads.
        _se = new SqueezeAndExcitationLayer<T>(channels, seReductionRatio,
            (IActivationFunction<T>?)new ReLUActivation<T>(), (IActivationFunction<T>?)new SigmoidActivation<T>());
        _dropout = dropoutRate > 0 ? new DropoutLayer<T>(dropoutRate) : null;
    }

    private IEnumerable<ILayer<T>> TrainableSubLayers()
    {
        yield return _residualConv;
        yield return _residualBn;
        for (int r = 0; r < _numSubBlocks; r++)
        {
            yield return _depthwise[r];
            yield return _pointwise[r];
            yield return _bn[r];
        }
        yield return _se;
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var l in TrainableSubLayers()) total += l.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException($"CitrinetBlockLayer requires rank-3 [B, C, T] input; got rank {input.Shape.Length}.", nameof(input));

        // Projected residual (skip) path.
        var residual = _residualBn.Forward(_residualConv.Forward(input));

        // Main path: R time-channel separable sub-blocks.
        var h = input;
        for (int r = 0; r < _numSubBlocks; r++)
        {
            h = _depthwise[r].Forward(h);   // depthwise-temporal
            h = _pointwise[r].Forward(h);   // pointwise-channel (1x1)
            h = _bn[r].Forward(h);
            if (r < _numSubBlocks - 1)
            {
                h = Engine.ReLU(h);
                if (_dropout is not null) h = _dropout.Forward(h);
            }
        }

        // Squeeze-and-excitation channel attention. The SE layer squeezes over the sequence axis of
        // a channels-LAST tensor, so transpose [B, C, T] -> [B, T, C], recalibrate, transpose back.
        var hLast = Engine.TensorPermute(h, new[] { 0, 2, 1 });
        hLast = _se.Forward(hLast);
        h = Engine.TensorPermute(hLast, new[] { 0, 2, 1 });

        // Residual add + activation (Engine.TensorAdd records the skip on the tape).
        h = Engine.ReLU(Engine.TensorAdd(h, residual));
        if (_dropout is not null) h = _dropout.Forward(h);
        return h;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var l in TrainableSubLayers()) l.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        Vector<T> all = Vector<T>.Empty();
        foreach (var l in TrainableSubLayers())
            all = Vector<T>.Concatenate(all, l.GetParameters());
        return all;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Validate the total length BEFORE slicing so a short vector throws the informative
        // ArgumentException below instead of a generic ArgumentOutOfRangeException from Slice()
        // mid-loop (mirrors DepthwiseConv1DLayer.SetParameters + SetExtraParameters).
        long expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters for CitrinetBlockLayer, but got {parameters.Length}.");
        }
        int offset = 0;
        foreach (var l in TrainableSubLayers())
        {
            int len = (int)l.ParameterCount;
            var slice = new Vector<T>(parameters.AsSpan().Slice(offset, len).ToArray());
            l.SetParameters(slice);
            offset += len;
        }
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var l in TrainableSubLayers()) l.SetTrainingMode(isTraining);
        _dropout?.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var l in TrainableSubLayers()) l.ResetState();
        _dropout?.ResetState();
    }

    // BN layers own non-trainable running mean/variance state that GetParameters/SetParameters
    // (weights only) does NOT carry. They must round-trip through ILayerSerializationExtras or a
    // deserialized clone runs inference with default (0,1) stats and diverges from the trained model
    // (the #1221 Clone_AfterTraining class). Fixed order, used by both Get and Set.
    private IEnumerable<BatchNormalizationLayer<T>> BnLayers()
    {
        yield return _residualBn;
        for (int r = 0; r < _numSubBlocks; r++) yield return _bn[r];
    }

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            foreach (var bn in BnLayers())
                if (bn is ILayerSerializationExtras<T> e) count += e.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var parts = new List<T>();
        foreach (var bn in BnLayers())
            if (bn is ILayerSerializationExtras<T> e) parts.AddRange(e.GetExtraParameters().ToArray());
        return new Vector<T>(parts.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        int offset = 0;
        foreach (var bn in BnLayers())
        {
            if (bn is not ILayerSerializationExtras<T> e) continue;
            int count = e.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for CitrinetBlockLayer BN: need {offset + count} but got {extraParameters.Length}.");
            e.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
    }

    /// <summary>Serialization metadata — the block is fully reconstructable from these.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["NumSubBlocks"] = _numSubBlocks.ToString();
        metadata["SeReductionRatio"] = _seReductionRatio.ToString();
        metadata["DropoutRate"] = _dropoutRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Stride"] = _stride.ToString();
        return metadata;
    }
}
