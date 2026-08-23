using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A depthwise 1-D convolution over <c>[B, C, T]</c> feature/waveform data: each input channel is
/// convolved with its own temporal filter (no cross-channel mixing), producing
/// <c>C * multiplier</c> output channels.
/// </summary>
/// <remarks>
/// <para>
/// This is the <b>time</b> half of the 1-D time-channel separable convolution used by NVIDIA
/// QuartzNet / Citrinet (Majumdar et al., 2021). A full TCSConv is this depthwise-temporal layer
/// followed by a pointwise (1×1) <see cref="Conv1DLayer{T}"/> that mixes channels — factorizing a
/// dense 1-D convolution into far fewer parameters while preserving the receptive field.
/// </para>
/// <para>
/// Built on the tape-aware <see cref="IEngine.DepthwiseConv1D{T}"/> engine primitive, so the
/// gradient flows through the recorded op with no hand-written backward. The kernel width is
/// <c>C</c> (channels are known at construction), so weights are allocated eagerly — avoiding the
/// lazy-init/Clone-via-SetParameters disagreement that afflicts input-inferred layers.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(NormalizesInput = false, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "4, 3")]
// Rank 3 ONLY, enforced rather than assumed: ForwardTraced throws
// "requires rank-3 [B, C, T] input" for anything else, and the axis names are its own.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DepthwiseConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// From this layer's own arithmetic in <see cref="ForwardTraced"/>:
    /// <c>tOut = (tIn + 2*_padding - _kernelSize) / _stride + 1</c> - the sliding-window formula at
    /// dilation 1, which this layer does not parameterize.
    /// </para>
    /// <para>
    /// The channel relation is <c>Fixed</c> rather than <c>Scaled</c> by the multiplier, and the
    /// difference is not cosmetic. A depthwise convolution's channel count is fixed AT CONSTRUCTION -
    /// ForwardTraced rejects any other input width outright ("holds one filter per input channel, so
    /// the channel count is fixed at construction"), so the output width is <c>_channels *
    /// _multiplier</c> for every input this layer accepts, never a function of the one it was given.
    /// <c>Scaled(Channels, _multiplier)</c> would be numerically identical on accepted input and wrong
    /// about why, implying the layer would follow a wider input it would in fact refuse.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3 || _channels <= 0 || _multiplier <= 0
            || _kernelSize <= 0 || _stride <= 0 || _padding < 0)
        {
            return null;
        }

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_channels * _multiplier)),
            new OutputAxisContract(
                TensorAxis.Time, AxisRelation.Window(TensorAxis.Time, _kernelSize, _stride, _padding)),
        };
    }

    private readonly int _channels;
    private readonly int _multiplier;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    private Tensor<T> _kernel;   // [channels, multiplier, kernelSize]
    private Tensor<T> _bias;     // [channels * multiplier]

    /// <summary>Constructs a depthwise 1-D convolution.</summary>
    /// <param name="channels">Number of input channels (<c>C</c>).</param>
    /// <param name="kernelSize">Temporal kernel width (<c>K</c>).</param>
    /// <param name="multiplier">Depthwise channel multiplier (output channels = <c>C * multiplier</c>). Defaults to 1.</param>
    /// <param name="stride">Temporal stride (Citrinet uses 2 for its subsampling blocks). Defaults to 1.</param>
    /// <param name="padding">Zero padding along time. Defaults to <c>(kernelSize-1)/2</c> for "same" output length when stride=1.</param>
    /// <param name="activation">Optional scalar activation. Defaults to identity (activation is usually applied after the pointwise stage + BN).</param>
    /// <param name="initializationStrategy">Optional weight initialization (defaults to He).</param>
    public DepthwiseConv1DLayer(
        [LayerState] int channels,
        [LayerState] int kernelSize,
        [LayerState] int multiplier = 1,
        [LayerState] int stride = 1,
        int? padding = null,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { channels, -1 }, new[] { channels * multiplier, -1 },
               activation ?? new IdentityActivation<T>())
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (multiplier <= 0) throw new ArgumentOutOfRangeException(nameof(multiplier));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _channels = channels;
        _multiplier = multiplier;
        _kernelSize = kernelSize;
        _stride = stride;
        // "same" padding when stride=1 so the depthwise stage preserves T (the pointwise stage and
        // the residual add in a Citrinet block rely on a length-preserving default).
        _padding = padding ?? ((kernelSize - 1) / 2);

        int outChannels = channels * multiplier;
        _kernel = AllocateLazyWeight([channels, multiplier, kernelSize]);
        _bias = AllocateLazyWeight([outChannels]);
        // Depthwise fan-in is just the kernel width (one filter per channel, no cross-channel sum).
        InitializeLayerWeights(_kernel, kernelSize, outChannels);
        InitializeLayerBiases(_bias);
        RegisterTrainableParameter(_kernel, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);

        // Resolve against a placeholder T large enough for the kernel to fit; the real T binds on
        // first Forward and does not change the parameter count (conv is translation-invariant in T).
        int minTime = _kernelSize;
        int outTime = (minTime + 2 * _padding - _kernelSize) / _stride + 1;
        ResolveShapes(new[] { channels, minTime }, new[] { outChannels, outTime });
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException($"DepthwiseConv1DLayer requires rank-3 [B, C, T] input; got rank {input.Shape.Length}.", nameof(input));

        // The channel count is a HARD constraint for a depthwise convolution, not a preference: the
        // kernel is [channels, multiplier, kernelSize], one filter per input channel, so there is no
        // filter for a channel this layer was not built for. Without this check the mismatch reached
        // Engine.DepthwiseConv1D and surfaced as IndexOutOfRangeException from inside the kernel — an
        // error that names no constraint, points at no caller mistake, and is indistinguishable from an
        // engine defect. Unlike the sequence length above, which really is free, this one cannot be
        // rebound: the parameters are sized by it.
        if (input.Shape[1] != _channels)
        {
            throw new ArgumentException(
                $"DepthwiseConv1DLayer was built for {_channels} channel(s) and received "
                + $"{input.Shape[1]}. A depthwise convolution holds one filter per input channel, so the "
                + "channel count is fixed at construction; only the batch and sequence length are free.",
                nameof(input));
        }

        // Bind the real sequence length so GetOutputShape() reports the actual T_out (the ctor only
        // resolved against a placeholder T since the real length is unknown until the first forward —
        // mirrors Conv1DLayer.OnFirstForward). The parameter count is unaffected (conv is
        // translation-invariant in T).
        int tIn = input.Shape[2];
        int tOut = (tIn + 2 * _padding - _kernelSize) / _stride + 1;
        ResolveShapes(new[] { _channels, tIn }, new[] { _channels * _multiplier, tOut });

        // Tape-aware depthwise temporal conv: [B, C, T] -> [B, C*mult, T_out].
        var conv = Engine.DepthwiseConv1D(input, _kernel, _stride, _padding);

        // Broadcast bias [C*mult] -> [1, C*mult, 1] and add, then activate.
        var biasReshaped = Engine.Reshape(_bias, new[] { 1, _channels * _multiplier, 1 });
        var withBias = Engine.TensorAdd(conv, biasReshaped);
        return ApplyActivation(withBias);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
    }

    /// <summary>Serialization metadata — the layer is fully reconstructable from these.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString();
        metadata["Multiplier"] = _multiplier.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["Stride"] = _stride.ToString();
        metadata["Padding"] = _padding.ToString();
        return metadata;
    }
}
