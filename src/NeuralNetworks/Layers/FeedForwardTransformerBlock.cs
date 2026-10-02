using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The feed-forward Transformer (FFT) block of FastSpeech: multi-head self-attention followed by a two-layer 1D
/// convolutional network, each with a residual connection, dropout and post-layer normalization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FastSpeech (Ren et al. 2019, §3.1) replaces the Transformer's position-wise dense FFN with a 2-layer 1D
/// convolution and ReLU, "motivated by the observation that the adjacent hidden states are more closely related
/// in the character/phoneme and mel-spectrogram sequence"; residual connections, layer normalization and dropout
/// follow the self-attention and the convolution respectively. FastSpeech 2 (Ren et al. 2021, App. A) uses the
/// same block with hidden 256, 2 heads and kernel sizes 9 and 1 (256→1024→256):
/// </para>
/// <list type="number">
/// <item>y = LayerNorm(x + Dropout(SelfAttention(x)))</item>
/// <item>z = LayerNorm(y + Dropout(Conv1D_k2(ReLU(Conv1D_k1(y)))))</item>
/// </list>
/// <para>Input and output are <c>[batch, time, hidden]</c>; the convolutions run on the transposed
/// <c>[batch, hidden, time]</c> view with "same" padding, so the time axis keeps its length.</para>
/// <para><b>For Beginners:</b> Attention lets every phoneme (or mel frame) look at every other one; the small
/// convolution then mixes each position with its immediate neighbours, which suits speech, where neighbouring
/// sounds blend into each other.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, Cost = ComputeCost.High, TestInputShape = "1, 6, 16", TestConstructorArgs = "16, 2, 32, 9, 1, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class FeedForwardTransformerBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _filterSize;
    private readonly int _firstKernelSize;
    private readonly int _secondKernelSize;
    private readonly double _dropoutRate;

    [SubLayerInput("1, _hiddenSize")]
    private readonly MultiHeadAttentionLayer<T> _attention;
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _attentionNorm;
    [SubLayerInput("1, _hiddenSize, 1")]
    private readonly Conv1DLayer<T> _conv1;
    [SubLayerInput("1, _filterSize, 1")]
    private readonly Conv1DLayer<T> _conv2;
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _convNorm;
    private readonly DropoutLayer<T>? _attentionDropout;
    private readonly DropoutLayer<T>? _convDropout;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates an FFT block.
    /// </summary>
    /// <param name="hiddenSize">Model (input/output) width.</param>
    /// <param name="numHeads">Self-attention heads; must divide <paramref name="hiddenSize"/>.</param>
    /// <param name="filterSize">Output channels of the first convolution (1024 in FastSpeech 2).</param>
    /// <param name="firstKernelSize">Kernel of the first convolution (9 in FastSpeech 2).</param>
    /// <param name="secondKernelSize">Kernel of the second convolution (1 in FastSpeech 2).</param>
    /// <param name="dropoutRate">Dropout after each sublayer, before the residual add (0.1 in FastSpeech 2).</param>
    public FeedForwardTransformerBlock(
        [LayerState] int hiddenSize,
        [LayerState] int numHeads,
        [LayerState] int filterSize,
        [LayerState] int firstKernelSize = 9,
        [LayerState] int secondKernelSize = 1,
        [LayerState] double dropoutRate = 0.0)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (filterSize <= 0) throw new ArgumentOutOfRangeException(nameof(filterSize));
        if (firstKernelSize <= 0 || firstKernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(firstKernelSize), "Kernel sizes must be odd so 'same' padding keeps the length.");
        if (secondKernelSize <= 0 || secondKernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(secondKernelSize), "Kernel sizes must be odd so 'same' padding keeps the length.");
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException($"Hidden size ({hiddenSize}) must be divisible by the number of heads ({numHeads}).", nameof(hiddenSize));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _filterSize = filterSize;
        _firstKernelSize = firstKernelSize;
        _secondKernelSize = secondKernelSize;
        _dropoutRate = dropoutRate;

        _attention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: new IdentityActivation<T>());
        _attentionNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _conv1 = new Conv1DLayer<T>(inputChannels: hiddenSize, outputChannels: filterSize, kernelSize: firstKernelSize,
            activation: new ReLUActivation<T>());
        _conv2 = new Conv1DLayer<T>(inputChannels: filterSize, outputChannels: hiddenSize, kernelSize: secondKernelSize);
        _convNorm = new LayerNormalizationLayer<T>(hiddenSize);
        if (dropoutRate > 0)
        {
            _attentionDropout = new DropoutLayer<T>(dropoutRate);
            _convDropout = new DropoutLayer<T>(dropoutRate);
        }

        RegisterSubLayer(_attention);
        RegisterSubLayer(_attentionNorm);
        RegisterSubLayer(_conv1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_convNorm);
        if (_attentionDropout is not null) RegisterSubLayer(_attentionDropout);
        if (_convDropout is not null) RegisterSubLayer(_convDropout);
    }

    /// <summary>Model width.</summary>
    public int HiddenSize => _hiddenSize;
    /// <summary>Attention heads.</summary>
    public int NumHeads => _numHeads;
    /// <summary>Output channels of the first convolution.</summary>
    public int FilterSize => _filterSize;
    /// <summary>Kernel of the first convolution.</summary>
    public int FirstKernelSize => _firstKernelSize;
    /// <summary>Kernel of the second convolution.</summary>
    public int SecondKernelSize => _secondKernelSize;
    /// <summary>Dropout probability.</summary>
    public double DropoutRate => _dropoutRate;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;

        var attended = _attention.Forward(x);
        if (_attentionDropout is not null) attended = _attentionDropout.Forward(attended);
        var y = _attentionNorm.Forward(Engine.TensorAdd(x, attended));

        // [B, T, H] -> [B, H, T] for the convolutions, then back.
        var channelsFirst = Engine.TensorPermute(y, new[] { 0, 2, 1 }).Contiguous();
        var hidden = _conv1.Forward(channelsFirst);
        var convOut = Engine.TensorPermute(_conv2.Forward(hidden), new[] { 0, 2, 1 }).Contiguous();
        if (_convDropout is not null) convOut = _convDropout.Forward(convOut);
        var z = _convNorm.Forward(Engine.TensorAdd(y, convOut));

        return unbatched ? Engine.Reshape(z, input._shape) : z;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>
    /// Persists the constructor arguments so deserialization can rebuild the sublayers before loading weights.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["NumHeads"] = _numHeads.ToString(inv);
        metadata["FilterSize"] = _filterSize.ToString(inv);
        metadata["FirstKernelSize"] = _firstKernelSize.ToString(inv);
        metadata["SecondKernelSize"] = _secondKernelSize.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString(inv);
        return metadata;
    }
}
