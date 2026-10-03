using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A residual stack of <c>Conv1D → LayerNorm → ReLU → Dropout</c> layers with a zero-initialized 1×1 output
/// projection: <c>y = x + P(f(x))</c>, which starts as the identity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Glow-TTS's encoder pre-net (Kim et al. 2020, §3.3: "the pre-net ... three convolutional layers ... with a residual
/// connection"; reference implementation <c>modules.ConvReluNorm</c>: kernel 5, dropout 0.5, layer normalization over
/// channels with ε = 1e-4, and a projection whose weight and bias start at zero).
/// </para>
/// <para>Input and output are <c>[time, channels]</c> or <c>[batch, time, channels]</c>.</para>
/// <para><b>For Beginners:</b> A few convolutions that learn a correction to their input, starting from "no change".</para>
/// </remarks>
[LayerCategory(LayerCategory.Residual)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 6, 8", TestConstructorArgs = "8, 5, 3, 0.0")]
[ElementWiseShape(Note = "A residual correction; the shape is carried through.")]
[AutoParameters]
public partial class ResidualConvReluNormLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _layers;
    private readonly double _dropoutRate;

    private readonly List<Conv1DLayer<T>> _convs = new();
    private readonly List<LayerNormalizationLayer<T>> _norms = new();
    private readonly DropoutLayer<T>? _dropout;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _projectionWeight;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _projectionBias;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the stack.</summary>
    /// <param name="channels">Channels of the input, hidden layers and output.</param>
    /// <param name="kernelSize">Odd convolution kernel (5).</param>
    /// <param name="layers">Convolutions (3).</param>
    /// <param name="dropoutRate">Dropout after each ReLU (0.5).</param>
    public ResidualConvReluNormLayer(
        [LayerState] int channels,
        [LayerState] int kernelSize,
        [LayerState] int layers,
        [LayerState] double dropoutRate)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (kernelSize <= 0 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd.");
        if (layers <= 0) throw new ArgumentOutOfRangeException(nameof(layers));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _channels = channels;
        _kernelSize = kernelSize;
        _layers = layers;
        _dropoutRate = dropoutRate;
        for (int i = 0; i < layers; i++)
        {
            var conv = new Conv1DLayer<T>(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize);
            var norm = new LayerNormalizationLayer<T>(channels, 1e-4);
            _convs.Add(conv);
            _norms.Add(norm);
            RegisterSubLayer(conv);
            RegisterSubLayer(norm);
        }
        if (dropoutRate > 0)
        {
            _dropout = new DropoutLayer<T>(dropoutRate);
            RegisterSubLayer(_dropout);
        }
        _projectionWeight = new Tensor<T>(new[] { channels, channels });
        _projectionBias = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_projectionWeight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_projectionBias, PersistentTensorRole.Biases);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _channels)
            throw new ArgumentException($"Expected [time, {_channels}] or [batch, time, {_channels}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], _channels }) : input;
        int batch = x.Shape[0], time = x.Shape[1];
        var h = x;
        for (int i = 0; i < _layers; i++)
        {
            var channelsFirst = Engine.TensorPermute(h, new[] { 0, 2, 1 }).Contiguous();
            h = Engine.TensorPermute(_convs[i].Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
            h = Engine.ReLU(_norms[i].Forward(h));
            if (_dropout is not null) h = _dropout.Forward(h);
        }
        var rows = Engine.Reshape(h, new[] { batch * time, _channels });
        var projected = Engine.TensorAdd(Engine.TensorMatMul(rows, Engine.TensorTranspose(_projectionWeight)),
            Engine.TensorTile(Engine.Reshape(_projectionBias, new[] { 1, _channels }), new[] { batch * time, 1 }));
        var output = Engine.TensorAdd(x, Engine.Reshape(projected, new[] { batch, time, _channels }));
        return unbatched ? Engine.Reshape(output, input._shape) : output;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Layers"] = _layers.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        return metadata;
    }
}
