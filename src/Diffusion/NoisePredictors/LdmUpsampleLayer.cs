using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// LDM's U-Net <c>Upsample</c> (<c>ldm/modules/diffusionmodules/openaimodel.py</c>, <c>use_conv=True</c>): nearest
/// interpolation by 2 (<c>F.interpolate(x, scale_factor=2, mode="nearest")</c>) followed by a 3x3 convolution with
/// padding 1 (<c>conv</c>).
/// </summary>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
internal sealed partial class LdmUpsampleLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly ConvolutionalLayer<T> _conv;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>The 3x3 convolution after the interpolation (<c>conv</c>).</summary>
    internal ConvolutionalLayer<T> Convolution => _conv;

    /// <param name="channels">The channel count, kept by the layer.</param>
    public LdmUpsampleLayer(int channels)
        : base([channels, -1, -1], [channels, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _conv = ConvolutionalLayer<T>.WithInputDepth(
            inputDepth: channels,
            outputDepth: channels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>(),
            initializationStrategy: InitializationStrategies<T>.Lazy);
        RegisterSubLayer(_conv);
    }

    /// <inheritdoc />
    /// <remarks>Channels and batch pass through; both spatial axes double.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4) return null;
        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 2)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 2)),
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4 || input.Shape[1] != _channels)
            throw new ArgumentException($"LdmUpsampleLayer reads [B, {_channels}, H, W].", nameof(input));
        var output = _conv.Forward(Engine.Upsample(input, 2, 2));
        ResolveShapes([_channels, input.Shape[2], input.Shape[3]], [_channels, input.Shape[2] * 2, input.Shape[3] * 2]);
        return output;
    }

    /// <summary>Loads <c>{prefix}conv.weight [C, C, 3, 3]</c> and <c>{prefix}conv.bias</c>.</summary>
    internal IReadOnlyList<string> LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
        => LdmCheckpoint.Conv(source, _conv, prefix + "conv", _channels, _channels, 3);

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate) => _conv.UpdateParameters(learningRate);

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _conv.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override void ResetState() => _conv.ResetState();

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
