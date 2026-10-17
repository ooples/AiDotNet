using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A stack of 1-D convolutions, each followed by batch normalization, an activation and dropout: Tacotron 2's encoder
/// convolutions and post-net.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tacotron 2 (Shen et al. 2018, §2.2–2.3): the encoder's "3 convolutional layers each containing 512 filters with shape
/// 5 × 1, followed by batch normalization and ReLU activations", and the post-net's "5 convolutional layers ... 512
/// filters with shape 5 × 1 with batch normalization, followed by tanh activations on all but the final layer"; dropout
/// 0.5 regularizes convolutional layers. Transformer TTS (Li et al. 2019, §3.3, §3.7) reuses both. Convolutions use
/// "same" padding (odd kernels), so the time axis keeps its length; batch normalization is per channel over batch and
/// time.
/// </para>
/// <para>Input <c>[time, channels]</c> or <c>[batch, time, channels]</c>; output the same with the last layer's channels.</para>
/// <para><b>For Beginners:</b> A few convolutions in a row, each looking at five neighbouring frames, with normalization
/// keeping the values in a healthy range.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ChangesShape = true, TestInputShape = "1, 6, 8",
    TestConstructorArgs = "8, new[] { 8, 4 }, 5, true, true, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ConvBatchNormStackLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputChannels;
    private readonly int[] _channels;
    private readonly int _kernelSize;
    private readonly bool _useTanh;
    private readonly bool _linearLast;
    private readonly double _dropoutRate;

    private readonly List<Conv1DLayer<T>> _convs = new();
    private readonly List<BatchNormalizationLayer<T>> _norms = new();
    private readonly List<DropoutLayer<T>> _dropouts = new();

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the stack.</summary>
    /// <param name="inputChannels">Channels of the input.</param>
    /// <param name="channels">Output channels of each convolution.</param>
    /// <param name="kernelSize">Odd kernel of every convolution (5 in Tacotron 2).</param>
    /// <param name="useTanh">tanh activations (post-net) rather than ReLU (encoder).</param>
    /// <param name="linearLast">Leave the last layer without an activation (post-net).</param>
    /// <param name="dropoutRate">Dropout after every layer (0.5 in Tacotron 2).</param>
    public ConvBatchNormStackLayer(
        [LayerState] int inputChannels,
        [LayerState] int[] channels,
        [LayerState] int kernelSize,
        [LayerState] bool useTanh,
        [LayerState] bool linearLast,
        [LayerState] double dropoutRate)
        : base(new[] { inputChannels }, new[] { channels is { Length: > 0 } ? channels[^1] : 1 })
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (channels is null || channels.Length == 0 || channels.Any(c => c <= 0))
            throw new ArgumentException("Expected at least one positive channel count.", nameof(channels));
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd so 'same' padding keeps the length.");
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _inputChannels = inputChannels;
        _channels = (int[])channels.Clone();
        _kernelSize = kernelSize;
        _useTanh = useTanh;
        _linearLast = linearLast;
        _dropoutRate = dropoutRate;

        int previous = inputChannels;
        foreach (int c in _channels)
        {
            var conv = new Conv1DLayer<T>(inputChannels: previous, outputChannels: c, kernelSize: kernelSize);
            var norm = new BatchNormalizationLayer<T>(c, epsilon: 1e-5, momentum: 0.9);
            _convs.Add(conv);
            _norms.Add(norm);
            RegisterSubLayer(conv);
            RegisterSubLayer(norm);
            if (dropoutRate > 0)
            {
                var dropout = new DropoutLayer<T>(dropoutRate);
                _dropouts.Add(dropout);
                RegisterSubLayer(dropout);
            }
            previous = c;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_channels[^1]));
        var time = new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time));
        return inputRank switch
        {
            2 => new[] { time, features },
            3 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), time, features },
            _ => null,
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _inputChannels)
            throw new ArgumentException(
                $"Expected [time, {_inputChannels}] or [batch, time, {_inputChannels}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        int batch = x.Shape[0], time = x.Shape[1];
        for (int i = 0; i < _convs.Count; i++)
        {
            var channelsFirst = Engine.TensorPermute(x, new[] { 0, 2, 1 }).Contiguous();
            var y = Engine.TensorPermute(_convs[i].Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
            int c = _channels[i];
            y = Engine.Reshape(_norms[i].Forward(Engine.Reshape(y, new[] { batch * time, c })), new[] { batch, time, c });
            if (!(_linearLast && i == _convs.Count - 1))
                y = _useTanh ? Engine.Tanh(y) : Engine.ReLU(y);
            if (_dropouts.Count > 0) y = _dropouts[i].Forward(y);
            x = y;
        }
        return unbatched ? Engine.Reshape(x, new[] { time, _channels[^1] }) : x;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments so deserialization can rebuild the sublayers before loading weights.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputChannels"] = _inputChannels.ToString(inv);
        metadata["Channels"] = string.Join(",", _channels);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["UseTanh"] = _useTanh.ToString(inv);
        metadata["LinearLast"] = _linearLast.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        return metadata;
    }
}
