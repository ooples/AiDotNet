using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// AdaSpeech's acoustic condition encoder: two blocks of <c>Conv1D → ReLU → LayerNorm → Dropout</c>, then either a
/// mean pooling over time (one vector per utterance) or a linear projection per position (one vector per phoneme).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AdaSpeech (Chen et al. 2021, §2.1, Fig. 2, §3) uses three networks of this shape:
/// </para>
/// <list type="bullet">
/// <item>the utterance-level acoustic encoder (Fig. 2b): convolutions of kernel 5 and stride 3 with 256 filters on
/// the reference mel spectrogram, then mean pooling to a single vector;</item>
/// <item>the phoneme-level acoustic encoder (Fig. 2c): convolutions of kernel 3 and stride 1 with 256 filters on the
/// phoneme-level mel (frames averaged per phoneme), then a linear layer to dimension 4;</item>
/// <item>the phoneme-level acoustic predictor (Fig. 2d): the same structure as the phoneme-level encoder, reading the
/// phoneme encoder's hidden sequence.</item>
/// </list>
/// <para>Input is <c>[time, channels]</c> or <c>[batch, time, channels]</c>. Output is <c>[filters]</c> /
/// <c>[batch, filters]</c> with pooling, or <c>[time', outputSize]</c> / <c>[batch, time', outputSize]</c> without,
/// where <c>time'</c> is the convolved length (equal to <c>time</c> at stride 1). The convolutions pad by
/// <c>(kernel − 1) / 2</c>.</para>
/// <para><b>For Beginners:</b> This network listens to a stretch of speech and summarizes it — either as one vector
/// for the whole utterance (its recording conditions and speaking style) or as a small vector per phoneme (its
/// accent, pitch and emphasis).</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ChangesShape = true, TestInputShape = "1, 6, 8",
    TestConstructorArgs = "8, 16, 3, 1, 4, false, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
// The pooled (utterance-level) output, [batch, features], is not declared as a layout: at rank 2 it would collide with
// the unpooled [time, features]. OutputAxesFor answers both from the layer's own configuration.
[AutoParameters]
public partial class AcousticConditionEncoderLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>Pooling removes the time axis. Without it, time is unchanged at stride 1 (the convolutions pad by
    /// <c>(kernel - 1) / 2</c>); at a larger stride it is two strided windows in sequence, which one relation cannot
    /// state.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3)) return null;
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(OutputSize));
        var batch = new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch));
        if (_meanPool)
            return inputRank == 3 ? new[] { batch, features } : new[] { features };
        var time = new OutputAxisContract(TensorAxis.Time, _stride == 1
            ? AxisRelation.Same(TensorAxis.Time)
            : AxisRelation.Unknown("two strided convolutions in sequence"));
        return inputRank == 3 ? new[] { batch, time, features } : new[] { time, features };
    }

    private readonly int _inputChannels;
    private readonly int _filterSize;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _outputSize;
    private readonly bool _meanPool;
    private readonly double _dropoutRate;

    [SubLayerInput("1, _inputChannels, 1")]
    private readonly Conv1DLayer<T> _conv1;
    [SubLayerInput("_filterSize")]
    private readonly LayerNormalizationLayer<T> _norm1;
    [SubLayerInput("1, _filterSize, 1")]
    private readonly Conv1DLayer<T> _conv2;
    [SubLayerInput("_filterSize")]
    private readonly LayerNormalizationLayer<T> _norm2;
    [SubLayerInput("_filterSize")]
    private readonly DenseLayer<T>? _projection;
    private readonly DropoutLayer<T>? _dropout1;
    private readonly DropoutLayer<T>? _dropout2;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates an acoustic condition encoder.</summary>
    /// <param name="inputChannels">Channels of the input sequence (mel bins, or the phoneme hidden size).</param>
    /// <param name="filterSize">Filters of both convolutions (256 in AdaSpeech).</param>
    /// <param name="kernelSize">Kernel of both convolutions (5 utterance-level, 3 phoneme-level).</param>
    /// <param name="stride">Stride of both convolutions (3 utterance-level, 1 phoneme-level).</param>
    /// <param name="outputSize">Width of the final linear projection (4 phoneme-level); 0 for none.</param>
    /// <param name="meanPool">Mean-pool over time to one vector (utterance-level).</param>
    /// <param name="dropoutRate">Dropout after each normalization.</param>
    public AcousticConditionEncoderLayer(
        [LayerState] int inputChannels,
        [LayerState] int filterSize,
        [LayerState] int kernelSize,
        [LayerState] int stride,
        [LayerState] int outputSize,
        [LayerState] bool meanPool,
        [LayerState] double dropoutRate)
        : base(new[] { inputChannels }, new[] { outputSize > 0 ? outputSize : filterSize })
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (filterSize <= 0) throw new ArgumentOutOfRangeException(nameof(filterSize));
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd so the padding is symmetric.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (outputSize < 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));

        _inputChannels = inputChannels;
        _filterSize = filterSize;
        _kernelSize = kernelSize;
        _stride = stride;
        _outputSize = outputSize;
        _meanPool = meanPool;
        _dropoutRate = dropoutRate;

        int padding = (kernelSize - 1) / 2;
        _conv1 = new Conv1DLayer<T>(inputChannels: inputChannels, outputChannels: filterSize, kernelSize: kernelSize,
            stride: stride, padding: padding, activation: new ReLUActivation<T>());
        _norm1 = new LayerNormalizationLayer<T>(filterSize);
        _conv2 = new Conv1DLayer<T>(inputChannels: filterSize, outputChannels: filterSize, kernelSize: kernelSize,
            stride: stride, padding: padding, activation: new ReLUActivation<T>());
        _norm2 = new LayerNormalizationLayer<T>(filterSize);
        if (outputSize > 0)
            _projection = new DenseLayer<T>(outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
        if (dropoutRate > 0)
        {
            _dropout1 = new DropoutLayer<T>(dropoutRate);
            _dropout2 = new DropoutLayer<T>(dropoutRate);
        }

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_norm2);
        if (_projection is not null) RegisterSubLayer(_projection);
        if (_dropout1 is not null) RegisterSubLayer(_dropout1);
        if (_dropout2 is not null) RegisterSubLayer(_dropout2);
    }

    /// <summary>Width of each output vector.</summary>
    public int OutputSize => _outputSize > 0 ? _outputSize : _filterSize;

    /// <summary>Whether the encoder pools the sequence to one vector.</summary>
    public bool MeanPool => _meanPool;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _inputChannels)
            throw new ArgumentException(
                $"Expected [time, {_inputChannels}] or [batch, time, {_inputChannels}], got [{string.Join(", ", input.Shape)}].",
                nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;

        x = Block(_conv1, _norm1, _dropout1, x);
        x = Block(_conv2, _norm2, _dropout2, x);

        if (_meanPool)
            x = Engine.ReduceMean(x, new[] { 1 }, keepDims: false);
        if (_projection is not null)
            x = _projection.Forward(x);

        if (!unbatched)
            return x;
        return _meanPool
            ? Engine.Reshape(x, new[] { x.Shape[1] })
            : Engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] });
    }

    // [B, T, C] -> Conv1D on [B, C, T] -> back to [B, T', F] -> LayerNorm over F -> Dropout.
    private Tensor<T> Block(Conv1DLayer<T> conv, LayerNormalizationLayer<T> norm, DropoutLayer<T>? dropout, Tensor<T> x)
    {
        var channelsFirst = Engine.TensorPermute(x, new[] { 0, 2, 1 }).Contiguous();
        var convolved = Engine.TensorPermute(conv.Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
        var normalized = norm.Forward(convolved);
        return dropout is null ? normalized : dropout.Forward(normalized);
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
        metadata["FilterSize"] = _filterSize.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Stride"] = _stride.ToString(inv);
        metadata["OutputSize"] = _outputSize.ToString(inv);
        metadata["MeanPool"] = _meanPool.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        return metadata;
    }
}
