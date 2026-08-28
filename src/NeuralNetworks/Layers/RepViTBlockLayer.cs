using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the token mixer and channel mixer from a RepViT block.
/// </summary>
/// <remarks>
/// The training graph follows the authors' reference implementation: a re-parameterizable
/// depthwise token mixer, optional squeeze-and-excitation, then a residual two-layer pointwise
/// channel mixer with GELU. Re-parameterization is an inference optimization and does not change
/// the training graph represented here.
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerCategory(LayerCategory.Residual)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4,
    Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "4, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class RepViTBlockLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _stride;
    private readonly bool _useSE;

    private readonly ConvolutionalLayer<T> _tokenDepthwise;
    private readonly BatchNormalizationLayer<T> _tokenDepthwiseBn;
    private readonly ConvolutionalLayer<T>? _tokenReparamPointwise;
    private readonly BatchNormalizationLayer<T>? _tokenReparamBn;
    private readonly ConvolutionalLayer<T>? _tokenProjection;
    private readonly BatchNormalizationLayer<T>? _tokenProjectionBn;
    private readonly SqueezeAndExcitationLayer<T>? _se;
    private readonly ConvolutionalLayer<T> _channelExpand;
    private readonly BatchNormalizationLayer<T> _channelExpandBn;
    private readonly ActivationLayer<T> _gelu;
    private readonly ConvolutionalLayer<T> _channelProject;
    private readonly BatchNormalizationLayer<T> _channelProjectBn;

    /// <summary>Creates one RepViT block for known input and output channel widths.</summary>
    public RepViTBlockLayer(
        [LayerState] int inChannels,
        [LayerState] int outChannels,
        [LayerState] int stride = 1,
        [LayerState] bool useSE = false)
        : base([inChannels, -1, -1], [outChannels, -1, -1])
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (stride is not (1 or 2))
            throw new ArgumentOutOfRangeException(nameof(stride), stride, "RepViT stride must be 1 or 2.");
        if (stride == 1 && inChannels != outChannels)
            throw new ArgumentException("A stride-1 RepViT block requires matching input and output channels.");

        _inChannels = inChannels;
        _outChannels = outChannels;
        _stride = stride;
        _useSE = useSE;

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _tokenDepthwise = new ConvolutionalLayer<T>(
            inChannels, 3, stride, 1, identity, groups: inChannels);
        _tokenDepthwiseBn = CreateChannelsFirstBatchNorm(inChannels);

        if (stride == 1)
        {
            // RepVGGDW training form: Conv3x3-BN + Conv1x1 + identity, followed by BN.
            _tokenReparamPointwise = new ConvolutionalLayer<T>(
                inChannels, 1, 1, 0, identity, groups: inChannels);
            _tokenReparamBn = CreateChannelsFirstBatchNorm(inChannels);
        }
        else
        {
            // Downsampling token mixer projects channels only after its depthwise spatial step.
            _tokenProjection = new ConvolutionalLayer<T>(outChannels, 1, 1, 0, identity);
            _tokenProjectionBn = CreateChannelsFirstBatchNorm(outChannels);
        }

        _se = useSE
            ? new SqueezeAndExcitationLayer<T>(
                inChannels, GetSEReductionRatio(inChannels), firstActivation: (IActivationFunction<T>?)null)
            : null;

        _channelExpand = new ConvolutionalLayer<T>(outChannels * 2, 1, 1, 0, identity);
        _channelExpandBn = CreateChannelsFirstBatchNorm(outChannels * 2);
        _gelu = new ActivationLayer<T>((IActivationFunction<T>)new GELUActivation<T>());
        _channelProject = new ConvolutionalLayer<T>(outChannels, 1, 1, 0, identity);
        _channelProjectBn = CreateChannelsFirstBatchNorm(outChannels);

        // The reference initializes the final channel-mixer BN scale to zero, making this
        // residual branch an exact identity at construction while preserving every other BN state.
        _channelProjectBn.ZeroInitGamma();

        foreach (var layer in EnumerateAllLayers()) RegisterSubLayer(layer);
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels));
        var height = new OutputAxisContract(
            TensorAxis.Height,
            AxisRelation.Window(TensorAxis.Height, kernel: 3, stride: _stride, padding: 1));
        var width = new OutputAxisContract(
            TensorAxis.Width,
            AxisRelation.Window(TensorAxis.Width, kernel: 3, stride: _stride, padding: 1));

        return inputRank switch
        {
            3 => [channels, height, width],
            4 =>
            [
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels,
                height,
                width
            ],
            _ => null
        };
    }

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        int channelAxis;
        int heightAxis;
        int widthAxis;
        if (input.Rank == 3)
        {
            channelAxis = 0;
            heightAxis = 1;
            widthAxis = 2;
        }
        else if (input.Rank == 4)
        {
            channelAxis = 1;
            heightAxis = 2;
            widthAxis = 3;
        }
        else
        {
            throw new ArgumentException(
                $"RepViT block requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {input.Rank}.",
                nameof(input));
        }

        if (input.Shape[channelAxis] != _inChannels)
        {
            throw new ArgumentException(
                $"RepViT block expected {_inChannels} input channels, got {input.Shape[channelAxis]}.",
                nameof(input));
        }

        int inputHeight = input.Shape[heightAxis];
        int inputWidth = input.Shape[widthAxis];
        int outputHeight = (inputHeight - 1) / _stride + 1;
        int outputWidth = (inputWidth - 1) / _stride + 1;
        ResolveShapes(
            [_inChannels, inputHeight, inputWidth],
            [_outChannels, outputHeight, outputWidth]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        var x = unbatched
            ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]])
            : input;
        if (x.Rank != 4 || x.Shape[1] != _inChannels)
            throw new ArgumentException($"RepViT block expects [B,{_inChannels},H,W].", nameof(input));

        var token = _tokenDepthwiseBn.Forward(_tokenDepthwise.Forward(x));
        if (_stride == 1)
        {
            var pointwise = _tokenReparamPointwise
                ?? throw new InvalidOperationException("Stride-1 token mixer was not initialized.");
            var outputBn = _tokenReparamBn
                ?? throw new InvalidOperationException("Stride-1 token mixer BN was not initialized.");
            token = outputBn.Forward(Engine.TensorAdd(Engine.TensorAdd(token, pointwise.Forward(x)), x));
            token = ApplySqueezeExcitation(token);
        }
        else
        {
            var projection = _tokenProjection
                ?? throw new InvalidOperationException("Downsampling token projection was not initialized.");
            var projectionBn = _tokenProjectionBn
                ?? throw new InvalidOperationException("Downsampling token projection BN was not initialized.");
            token = ApplySqueezeExcitation(token);
            token = projectionBn.Forward(projection.Forward(token));
        }

        var channel = _channelExpandBn.Forward(_channelExpand.Forward(token));
        channel = _gelu.Forward(channel);
        channel = _channelProjectBn.Forward(_channelProject.Forward(channel));
        var output = Engine.TensorAdd(token, channel);

        if (!IsShapeResolved)
            ResolveShapes(
                [_inChannels, x.Shape[2], x.Shape[3]],
                [_outChannels, output.Shape[2], output.Shape[3]]);

        return unbatched
            ? Engine.Reshape(output, [output.Shape[1], output.Shape[2], output.Shape[3]])
            : output;
    }

    private Tensor<T> ApplySqueezeExcitation(Tensor<T> input)
    {
        if (_se is null) return input;

        var channelsLast = Engine.TensorPermute(input, [0, 2, 3, 1]);
        return Engine.TensorPermute(_se.Forward(channelsLast), [0, 3, 1, 2]);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients() => Concatenate(EnumerateParameterLayers(), gradients: true);

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in EnumerateParameterLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in EnumerateParameterLayers()) layer.ClearGradients();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in EnumerateAllLayers()) layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InChannels"] = _inChannels.ToString();
        metadata["OutChannels"] = _outChannels.ToString();
        metadata["Stride"] = _stride.ToString();
        metadata["UseSE"] = _useSE.ToString();
        return metadata;
    }

    private static BatchNormalizationLayer<T> CreateChannelsFirstBatchNorm(int channels)
    {
        return new BatchNormalizationLayer<T>(channels)
        {
            Layout = BatchNormDataLayout.ChannelsFirst
        };
    }

    private static int GetSEReductionRatio(int channels)
    {
        // timm's SqueezeExcite rounds rd_ratio=0.25 to a multiple of eight. RepViT-M0.9's
        // 48-channel stage therefore uses 16 reduced channels (ratio 3), not 12 (ratio 4).
        int reducedChannels = Math.Max(8, (int)(channels * 0.25 + 4) / 8 * 8);
        if (reducedChannels < 0.9 * channels * 0.25) reducedChannels += 8;
        if (channels % reducedChannels != 0)
            throw new ArgumentException("RepViT SE width must divide the block channel width.", nameof(channels));
        return channels / reducedChannels;
    }

    private IEnumerable<LayerBase<T>> EnumerateParameterLayers()
    {
        yield return _tokenDepthwise;
        yield return _tokenDepthwiseBn;
        if (_tokenReparamPointwise is not null) yield return _tokenReparamPointwise;
        if (_tokenReparamBn is not null) yield return _tokenReparamBn;
        if (_tokenProjection is not null) yield return _tokenProjection;
        if (_tokenProjectionBn is not null) yield return _tokenProjectionBn;
        if (_se is not null) yield return _se;
        yield return _channelExpand;
        yield return _channelExpandBn;
        yield return _channelProject;
        yield return _channelProjectBn;
    }

    private IEnumerable<LayerBase<T>> EnumerateAllLayers()
    {
        foreach (var layer in EnumerateParameterLayers()) yield return layer;
        yield return _gelu;
    }

    private static Vector<T> Concatenate(IEnumerable<LayerBase<T>> layers, bool gradients)
    {
        var values = new List<T>();
        foreach (var layer in layers)
        {
            var vector = gradients ? layer.GetParameterGradients() : layer.GetParameters();
            for (int i = 0; i < vector.Length; i++) values.Add(vector[i]);
        }
        return new Vector<T>(values.ToArray());
    }
}
