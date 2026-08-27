using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Owns the two execution branches of an autoregressive encoder-decoder architecture.
/// </summary>
/// <remarks>
/// Keeping the encoder, decoder lookup, cross-attention stack, and output head inside one
/// composite prevents a sequential model runner from feeding continuous encoder features into
/// the decoder's integer lookup table. Registered children retain the standard automatic
/// parameter, checkpoint, optimizer, training-mode, and gradient-tape behavior from
/// <see cref="LayerBase{T}"/>.
/// </remarks>
public abstract class AutoregressiveEncoderDecoderLayer<T> : LayerBase<T>
{
    private readonly List<ILayer<T>> _encoderLayers;
    private readonly EmbeddingLayer<T> _decoderEmbedding;
    private readonly List<ILayer<T>> _decoderLayers;
    private readonly ILayer<T> _outputLayer;
    protected readonly int _encoderVocabularySize;
    protected readonly int _decoderVocabularySize;
    protected readonly int _maximumDecoderLength;

    // Generated construction factories reopen the concrete derived layer, so they cannot read the
    // base's private ownership fields directly. Exact-type protected views let the generator bind
    // IEnumerable/ILayer constructor arguments without exposing mutable collections publicly.
    protected IEnumerable<ILayer<T>> EncoderLayers => _encoderLayers;
    protected EmbeddingLayer<T> DecoderEmbedding => _decoderEmbedding;
    protected IEnumerable<ILayer<T>> DecoderLayers => _decoderLayers;
    protected ILayer<T> OutputLayer => _outputLayer;

    /// <summary>Creates an encoder-decoder composite from its independently executed branches.</summary>
    protected AutoregressiveEncoderDecoderLayer(
        IEnumerable<ILayer<T>> encoderLayers,
        EmbeddingLayer<T> decoderEmbedding,
        IEnumerable<ILayer<T>> decoderLayers,
        ILayer<T> outputLayer,
        int encoderVocabularySize,
        int decoderVocabularySize,
        int maximumDecoderLength)
        : base([-1], [decoderVocabularySize])
    {
        _encoderLayers = encoderLayers?.ToList()
            ?? throw new ArgumentNullException(nameof(encoderLayers));
        _decoderEmbedding = decoderEmbedding
            ?? throw new ArgumentNullException(nameof(decoderEmbedding));
        _decoderLayers = decoderLayers?.ToList()
            ?? throw new ArgumentNullException(nameof(decoderLayers));
        _outputLayer = outputLayer ?? throw new ArgumentNullException(nameof(outputLayer));

        if (_encoderLayers.Count == 0)
            throw new ArgumentException("At least one encoder layer is required.", nameof(encoderLayers));
        if (encoderVocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(encoderVocabularySize));
        if (decoderVocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(decoderVocabularySize));
        if (maximumDecoderLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumDecoderLength));

        _encoderVocabularySize = encoderVocabularySize;
        _decoderVocabularySize = decoderVocabularySize;
        _maximumDecoderLength = maximumDecoderLength;

        foreach (var layer in _encoderLayers) RegisterSubLayer(layer);
        RegisterSubLayer(_decoderEmbedding);
        foreach (var layer in _decoderLayers) RegisterSubLayer(layer);
        RegisterSubLayer(_outputLayer);
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank);

    /// <summary>Runs only the encoder branch.</summary>
    public Tensor<T> Encode(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var current = NormalizeEncoderInput(input);
        foreach (var layer in _encoderLayers)
            current = layer.Forward(current);

        return current;
    }

    /// <summary>Runs the decoder branch using explicit decoder token IDs.</summary>
    public Tensor<T> Decode(Tensor<T> encoderMemory, Tensor<T> decoderIds)
    {
        if (encoderMemory is null) throw new ArgumentNullException(nameof(encoderMemory));
        if (decoderIds is null) throw new ArgumentNullException(nameof(decoderIds));

        var current = _decoderEmbedding.Forward(decoderIds);
        foreach (var layer in _decoderLayers)
        {
            current = layer is TransformerDecoderLayer<T> decoder
                ? decoder.Forward(current, encoderMemory)
                : layer.Forward(current);
        }

        return _outputLayer.Forward(current);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var memory = Encode(input);
        return Decode(memory, CreateInitialDecoderIds(input));
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length == 0)
            throw new ArgumentException("Encoder input is required.", nameof(inputs));
        if (inputs.Length > 2)
            throw new ArgumentException("Expected encoder input and optional decoder IDs.", nameof(inputs));

        var memory = Encode(inputs[0]);
        return Decode(memory, inputs.Length == 2 ? inputs[1] : CreateInitialDecoderIds(inputs[0]));
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedPorts(IReadOnlyDictionary<string, Tensor<T>> inputs)
    {
        if (!inputs.TryGetValue(InputPorts[0].Name, out var encoderInput))
            throw new ArgumentException($"Missing required input port '{InputPorts[0].Name}'.", nameof(inputs));

        var memory = Encode(encoderInput);
        return inputs.TryGetValue("decoder_ids", out var decoderIds)
            ? Decode(memory, decoderIds)
            : Decode(memory, CreateInitialDecoderIds(encoderInput));
    }

    /// <summary>Normalizes an external encoder input into the representation its branch consumes.</summary>
    protected virtual Tensor<T> NormalizeEncoderInput(Tensor<T> input) => input;

    /// <summary>Creates the zero-valued decoder prompt used by the single-input model path.</summary>
    protected virtual Tensor<T> CreateInitialDecoderIds(Tensor<T> input)
    {
        int batchSize = input.Rank is 2 or 4 ? input.Shape[0] : 0;
        int sourceLength = input.Rank is 1 or 2 ? input.Shape[input.Rank - 1] : 1;
        int decoderLength = Math.Max(1, Math.Min(sourceLength, _maximumDecoderLength));
        return batchSize > 0
            ? new Tensor<T>([batchSize, decoderLength])
            : new Tensor<T>([decoderLength]);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in _encoderLayers) layer.ResetState();
        _decoderEmbedding.ResetState();
        foreach (var layer in _decoderLayers) layer.ResetState();
        _outputLayer.ResetState();
    }
}

/// <summary>An autoregressive encoder-decoder whose encoder consumes token IDs.</summary>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true)]
[TensorPort("encoder_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.EncoderInput, MaxExclusiveMember = "_encoderVocabularySize")]
[TensorPort("decoder_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.DecoderIds, Required = false, MaxExclusiveMember = "_decoderVocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Output)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class TokenConditionedDecoderLayer<T> : AutoregressiveEncoderDecoderLayer<T>, IShapeContract
{
    /// <summary>Creates a token-conditioned autoregressive decoder.</summary>
    public TokenConditionedDecoderLayer(
        IEnumerable<ILayer<T>> encoderLayers,
        EmbeddingLayer<T> decoderEmbedding,
        IEnumerable<ILayer<T>> decoderLayers,
        ILayer<T> outputLayer,
        int encoderVocabularySize,
        int decoderVocabularySize,
        int maximumDecoderLength)
        : base(encoderLayers, decoderEmbedding, decoderLayers, outputLayer,
            encoderVocabularySize, decoderVocabularySize, maximumDecoderLength)
    {
    }

    /// <inheritdoc />
    protected override Tensor<T> NormalizeEncoderInput(Tensor<T> input)
    {
        if (input.Rank <= 2)
            return input;

        int batchSize = input.Shape[0];
        return input.Reshape([batchSize, input.Length / batchSize]);
    }

    /// <inheritdoc />
    protected override Tensor<T> CreateInitialDecoderIds(Tensor<T> input)
    {
        int batchSize = input.Rank >= 2 ? input.Shape[0] : 0;
        int sourceLength = batchSize > 0 ? input.Length / batchSize : input.Length;
        int decoderLength = Math.Max(1, Math.Min(sourceLength, _maximumDecoderLength));
        return batchSize > 0
            ? new Tensor<T>([batchSize, decoderLength])
            : new Tensor<T>([decoderLength]);
    }

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 =>
        [
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_decoderVocabularySize)),
        ],
        2 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_decoderVocabularySize)),
        ],
        _ => null,
    };
}

/// <summary>An autoregressive encoder-decoder whose encoder consumes continuous features.</summary>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true)]
[TensorPort("encoder_input", TensorPortDirection.Input, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.EncoderInput)]
[TensorPort("decoder_ids", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.DecoderIds, Required = false, MaxExclusiveMember = "_decoderVocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Output)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class VisionEncoderDecoderLayer<T> : AutoregressiveEncoderDecoderLayer<T>, IShapeContract
{
    /// <summary>Creates a vision-conditioned autoregressive decoder.</summary>
    public VisionEncoderDecoderLayer(
        IEnumerable<ILayer<T>> encoderLayers,
        EmbeddingLayer<T> decoderEmbedding,
        IEnumerable<ILayer<T>> decoderLayers,
        ILayer<T> outputLayer,
        int decoderVocabularySize,
        int maximumDecoderLength)
        : base(encoderLayers, decoderEmbedding, decoderLayers, outputLayer,
            encoderVocabularySize: 1,
            decoderVocabularySize: decoderVocabularySize,
            maximumDecoderLength: maximumDecoderLength)
    {
    }

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        3 =>
        [
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_decoderVocabularySize)),
        ],
        4 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_decoderVocabularySize)),
        ],
        _ => null,
    };
}
