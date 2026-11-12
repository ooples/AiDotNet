using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// SpeedySpeech's residual block: <c>n</c> repetitions of a dilated 1D convolution, ReLU and temporal batch
/// normalization, with a residual connection around them: <c>y = x + f(x)</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SpeedySpeech (Vainer &amp; Dušek 2020, §3.2): "progressively dilated residual convolutional blocks, each of which
/// contains a 1D convolution, ReLU activation and temporal batch normalization. A residual connection is applied." The
/// kernel size and the exact composition are not in the paper; they follow the authors' implementation
/// (github.com/janvainer/speedyspeech, <c>layers.ResidualBlock</c>): each convolution is unpadded, its output is
/// zero-padded back to the input length (<c>⌊d(k−1)/2⌋</c> frames in front, the rest behind, so even kernels work), then
/// ReLU, then batch normalization of each channel over the batch and time axes. The paper's "26 encoder blocks with
/// dilations 1, 1, 2, 2, 4, 4" are 13 of these blocks with two convolutions each.
/// </para>
/// <para>Input and output are <c>[time, channels]</c> or <c>[batch, time, channels]</c>.</para>
/// <para><b>For Beginners:</b> Each convolution looks at a window of neighbouring frames, spaced further apart as the
/// dilation grows, so a deep stack can see a long stretch of the sequence without attention.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 6, 8", TestConstructorArgs = "8, 4, 2, 2")]
[ElementWiseShape(Note = "Residual block: the zero-padded convolutions keep the time axis, and channels are fixed.")]
[AutoParameters]
public partial class DilatedResidualConvBlock<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _dilation;
    private readonly int _convolutions;

    [SubLayerInput("1, _channels, 1")]
    private readonly List<Conv1DLayer<T>> _convs = new();
    [SubLayerInput("_channels")]
    private readonly List<BatchNormalizationLayer<T>> _norms = new();

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates a residual block.</summary>
    /// <param name="channels">Channels of the input, every convolution and the output (128 in SpeedySpeech).</param>
    /// <param name="kernelSize">Kernel of every convolution (4 in SpeedySpeech's encoder and decoder).</param>
    /// <param name="dilation">Dilation of every convolution.</param>
    /// <param name="convolutions">Convolution–ReLU–norm repetitions inside the residual connection.</param>
    public DilatedResidualConvBlock(
        [LayerState] int channels,
        [LayerState] int kernelSize,
        [LayerState] int dilation,
        [LayerState] int convolutions = 2)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (convolutions <= 0) throw new ArgumentOutOfRangeException(nameof(convolutions));
        _channels = channels;
        _kernelSize = kernelSize;
        _dilation = dilation;
        _convolutions = convolutions;
        for (int i = 0; i < convolutions; i++)
        {
            var conv = new Conv1DLayer<T>(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                dilation: dilation, padding: 0);
            var norm = new BatchNormalizationLayer<T>(channels, epsilon: 1e-5, momentum: 0.9); // torch BatchNorm1d defaults (momentum 0.1 = keep 0.9)
            _convs.Add(conv);
            _norms.Add(norm);
            RegisterSubLayer(conv);
            RegisterSubLayer(norm);
        }
    }

    /// <summary>Kernel of the convolutions.</summary>
    public int KernelSize => _kernelSize;

    /// <summary>Dilation of the convolutions.</summary>
    public int Dilation => _dilation;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _channels)
            throw new ArgumentException(
                $"Expected [time, {_channels}] or [batch, time, {_channels}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], _channels }) : input;
        int batch = x.Shape[0], time = x.Shape[1];

        int padTotal = _dilation * (_kernelSize - 1);
        int padFront = padTotal / 2, padBack = padTotal - padFront;
        var y = x;
        for (int i = 0; i < _convolutions; i++)
        {
            int valid = time - padTotal;
            Tensor<T> convolved;
            if (valid > 0)
            {
                var channelsFirst = Engine.TensorPermute(y, new[] { 0, 2, 1 }).Contiguous();
                convolved = Engine.TensorPermute(_convs[i].Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
                var parts = new List<Tensor<T>>(3);
                if (padFront > 0) parts.Add(new Tensor<T>(new[] { batch, padFront, _channels }));
                parts.Add(convolved);
                if (padBack > 0) parts.Add(new Tensor<T>(new[] { batch, padBack, _channels }));
                convolved = parts.Count == 1 ? convolved : Engine.TensorConcatenate(parts.ToArray(), 1);
            }
            else
            {
                // Shorter than the receptive field: the unpadded convolution has no output, so the reference implementation
                // pads nothing but zeros back to the input length.
                convolved = new Tensor<T>(new[] { batch, time, _channels });
            }
            var activated = Engine.ReLU(convolved);
            var rows = Engine.Reshape(activated, new[] { batch * time, _channels });
            y = Engine.Reshape(_norms[i].Forward(rows), new[] { batch, time, _channels });
        }

        var output = Engine.TensorAdd(x, y);
        return unbatched ? Engine.Reshape(output, input._shape) : output;
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
        metadata["Channels"] = _channels.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Dilation"] = _dilation.ToString(inv);
        metadata["Convolutions"] = _convolutions.ToString(inv);
        return metadata;
    }
}
