using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The decoder layer of the original Transformer (Vaswani et al. 2017, §3.1): masked multi-head self-attention,
/// multi-head attention over the encoder output, and a position-wise feed-forward network, each followed by dropout,
/// a residual connection and layer normalization (post-normalization).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <list type="number">
/// <item><c>a = LayerNorm(x + Dropout(MaskedSelfAttention(x)))</c></item>
/// <item><c>b = LayerNorm(a + Dropout(Attention(a, memory)))</c></item>
/// <item><c>z = LayerNorm(b + Dropout(W₂ ReLU(W₁ b + b₁) + b₂))</c></item>
/// </list>
/// <para>Self-attention is causal: position <c>i</c> attends only to positions <c>≤ i</c> ("masking ... ensures that
/// the predictions for position i can depend only on the known outputs at positions less than i"). Inputs: the
/// target sequence <c>[time, hidden]</c> or <c>[batch, time, hidden]</c> and the encoder memory of the same rank.
/// Transformer TTS (Li et al. 2019, §3.6) uses this decoder.</para>
/// <para><see cref="TransformerDecoderBlock{T}"/> is the pre-normalization variant; <see cref="TransformerDecoderLayer{T}"/>
/// applies its activation to the attention outputs and has no dropout.</para>
/// <para><b>For Beginners:</b> Each output position looks back at what has been produced so far, then at the encoded
/// input, then refines itself — never peeking at positions that have not been produced yet.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "1, 3, 8",
    TestConstructorArgs = "8, 2, 16, 0.0")]
[ElementWiseShape(Note = "The target sequence's shape is carried through; the memory only feeds the cross-attention.")]
[AutoParameters]
public partial class PostNormTransformerDecoderBlock<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly double _dropoutRate;

    [SubLayerInput("1, _hiddenSize")]
    private readonly MultiHeadAttentionLayer<T> _selfAttention;
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _selfNorm;
    [SubLayerInput("1, _hiddenSize")]
    private readonly MultiHeadAttentionLayer<T> _crossAttention;
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _crossNorm;
    [SubLayerInput("1, _hiddenSize")]
    private readonly DenseLayer<T> _ffnUp;
    [SubLayerInput("1, _ffnDim")]
    private readonly DenseLayer<T> _ffnDown;
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _ffnNorm;
    private readonly DropoutLayer<T>? _selfDropout;
    private readonly DropoutLayer<T>? _crossDropout;
    private readonly DropoutLayer<T>? _ffnDropout;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool RequiresMultipleInputs => true;

    /// <summary>Creates a decoder layer.</summary>
    /// <param name="hiddenSize">Model width d_model.</param>
    /// <param name="numHeads">Attention heads; must divide <paramref name="hiddenSize"/>.</param>
    /// <param name="ffnDim">Inner width d_ff of the feed-forward network.</param>
    /// <param name="dropoutRate">Dropout on each sublayer output (0.1 in the base Transformer).</param>
    public PostNormTransformerDecoderBlock(
        [LayerState] int hiddenSize,
        [LayerState] int numHeads,
        [LayerState] int ffnDim,
        [LayerState] double dropoutRate = 0.1)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0 || hiddenSize % numHeads != 0)
            throw new ArgumentException($"The number of heads ({numHeads}) must divide the hidden size ({hiddenSize}).", nameof(numHeads));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _dropoutRate = dropoutRate;

        IActivationFunction<T> identity = new IdentityActivation<T>();
        _selfAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: identity)
        {
            UseCausalMask = true,
        };
        _selfNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _crossAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: identity);
        _crossNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _ffnUp = new DenseLayer<T>(ffnDim, new ReLUActivation<T>() as IActivationFunction<T>);
        _ffnDown = new DenseLayer<T>(hiddenSize, identity);
        _ffnNorm = new LayerNormalizationLayer<T>(hiddenSize);
        if (dropoutRate > 0)
        {
            _selfDropout = new DropoutLayer<T>(dropoutRate);
            _crossDropout = new DropoutLayer<T>(dropoutRate);
            _ffnDropout = new DropoutLayer<T>(dropoutRate);
        }

        RegisterSubLayer(_selfAttention);
        RegisterSubLayer(_selfNorm);
        RegisterSubLayer(_crossAttention);
        RegisterSubLayer(_crossNorm);
        RegisterSubLayer(_ffnUp);
        RegisterSubLayer(_ffnDown);
        RegisterSubLayer(_ffnNorm);
        if (_selfDropout is not null) RegisterSubLayer(_selfDropout);
        if (_crossDropout is not null) RegisterSubLayer(_crossDropout);
        if (_ffnDropout is not null) RegisterSubLayer(_ffnDropout);
    }

    /// <summary>Model width.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Runs the layer on the target sequence with the encoder memory.</summary>
    public Tensor<T> Forward(Tensor<T> target, Tensor<T> memory) => Forward(new[] { target, memory });

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => throw new InvalidOperationException("The decoder layer attends to an encoder memory; call Forward(target, memory).");

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length != 2)
            throw new ArgumentException("Expected the target sequence and the encoder memory.", nameof(inputs));
        var target = inputs[0];
        var memory = inputs[1];
        bool unbatched = target.Rank == 2;
        var x = unbatched ? Engine.Reshape(target, new[] { 1, target.Shape[0], target.Shape[1] }) : target;
        var m = memory.Rank == 2 ? Engine.Reshape(memory, new[] { 1, memory.Shape[0], memory.Shape[1] }) : memory;

        var self = _selfAttention.Forward(x);
        if (_selfDropout is not null) self = _selfDropout.Forward(self);
        var a = _selfNorm.Forward(Engine.TensorAdd(x, self));

        var cross = _crossAttention.Forward(a, m);
        if (_crossDropout is not null) cross = _crossDropout.Forward(cross);
        var b = _crossNorm.Forward(Engine.TensorAdd(a, cross));

        var ffn = _ffnDown.Forward(_ffnUp.Forward(b));
        if (_ffnDropout is not null) ffn = _ffnDropout.Forward(ffn);
        var z = _ffnNorm.Forward(Engine.TensorAdd(b, ffn));
        return unbatched ? Engine.Reshape(z, target._shape) : z;
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
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["NumHeads"] = _numHeads.ToString(inv);
        metadata["FfnDim"] = _ffnDim.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", inv);
        return metadata;
    }
}
