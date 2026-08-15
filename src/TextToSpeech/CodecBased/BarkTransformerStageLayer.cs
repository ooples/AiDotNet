using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Shared GPT transformer used by Bark's semantic, coarse, and fine stages.
/// </summary>
/// <remarks>
/// The layer owns every trainable sublayer and registers those children once. Parameter counting,
/// save/load, cloning, gradients, and device movement therefore flow through the normal
/// <see cref="LayerBase{T}"/> lifecycle instead of Bark maintaining parallel hand-written lists.
/// </remarks>
[AutoParameters]
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, HasTrainingMode = true, Cost = ComputeCost.High)]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "InputVocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Output)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
internal partial class BarkTransformerStageLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly BarkStageOptions _options;
    private readonly int _inputStreams;
    private readonly int _outputHeads;
    private readonly EmbeddingLayer<T>[] _tokenEmbeddings;
    private readonly EmbeddingLayer<T> _positionEmbeddings;
    private readonly TransformerEncoderBlock<T>[] _blocks;
    private readonly LayerNormalizationLayer<T> _finalNormalization;
    private readonly DenseLayer<T>[] _languageModelHeads;
    private KVCache<T>? _cache;
    private bool _cachedAttentionInstalled;

    internal BarkTransformerStageLayer(
        BarkStageOptions options,
        int inputStreams = 1,
        int outputHeads = 1)
        : this(
            options?.InputVocabularySize ?? 0,
            options?.OutputVocabularySize ?? 0,
            options?.HiddenSize ?? 0,
            options?.NumberOfLayers ?? 0,
            options?.NumberOfHeads ?? 0,
            options?.BlockSize ?? 0,
            options?.FeedForwardSize ?? 0,
            options?.Dropout ?? 0.0,
            options?.IsCausal ?? false,
            inputStreams,
            outputHeads)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Reconstructs a Bark transformer stage from its complete saved architecture state.
    /// </summary>
    /// <remarks>
    /// Every architecture-bearing argument is marked for the generated layer-state factory. This
    /// keeps clone/checkpoint restore generic: adding this composite required no switch branch in
    /// the deserializer and cannot drift from the constructor used for a forward pass.
    /// </remarks>
    public BarkTransformerStageLayer(
        [LayerState] int inputVocabularySize,
        [LayerState] int outputVocabularySize,
        [LayerState] int hiddenSize,
        [LayerState] int numberOfLayers,
        [LayerState] int numberOfHeads,
        [LayerState] int blockSize,
        [LayerState] int feedForwardSize,
        [LayerState] double dropout,
        [LayerState] bool isCausal,
        [LayerState] int inputStreams = 1,
        [LayerState] int outputHeads = 1)
        : base([LayerShape.Dynamic], [LayerShape.Dynamic, outputVocabularySize])
    {
        var options = new BarkStageOptions
        {
            InputVocabularySize = inputVocabularySize,
            OutputVocabularySize = outputVocabularySize,
            HiddenSize = hiddenSize,
            NumberOfLayers = numberOfLayers,
            NumberOfHeads = numberOfHeads,
            BlockSize = blockSize,
            FeedForwardSize = feedForwardSize,
            Dropout = dropout,
            IsCausal = isCausal,
        };
        options.Validate(nameof(options));
        if (inputStreams <= 0) throw new ArgumentOutOfRangeException(nameof(inputStreams));
        if (outputHeads <= 0) throw new ArgumentOutOfRangeException(nameof(outputHeads));

        _options = options.Copy();
        _inputStreams = inputStreams;
        _outputHeads = outputHeads;
        _tokenEmbeddings = new EmbeddingLayer<T>[inputStreams];
        for (int stream = 0; stream < inputStreams; stream++)
        {
            _tokenEmbeddings[stream] = new EmbeddingLayer<T>(
                _options.InputVocabularySize,
                _options.HiddenSize);
            RegisterSubLayer(_tokenEmbeddings[stream]);
        }

        _positionEmbeddings = new EmbeddingLayer<T>(_options.BlockSize, _options.HiddenSize);
        RegisterSubLayer(_positionEmbeddings);

        _blocks = new TransformerEncoderBlock<T>[_options.NumberOfLayers];
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            var block = new TransformerEncoderBlock<T>(
                _options.HiddenSize,
                _options.NumberOfHeads,
                _options.FeedForwardSize > 0 ? _options.FeedForwardSize : checked(_options.HiddenSize * 4),
                _options.Dropout,
                new GELUActivation<T>());
            if (block.AttentionLayer is MultiHeadAttentionLayer<T> attention)
                attention.UseCausalMask = _options.IsCausal;
            _blocks[layer] = block;
            RegisterSubLayer(block);
        }

        _finalNormalization = new LayerNormalizationLayer<T>(_options.HiddenSize);
        RegisterSubLayer(_finalNormalization);

        _languageModelHeads = new DenseLayer<T>[outputHeads];
        for (int head = 0; head < outputHeads; head++)
        {
            _languageModelHeads[head] = new DenseLayer<T>(
                _options.OutputVocabularySize,
                (IActivationFunction<T>)new IdentityActivation<T>());
            RegisterSubLayer(_languageModelHeads[head]);
        }
    }

    private int InputVocabularySize => _options.InputVocabularySize;
    private int OutputVocabularySize => _options.OutputVocabularySize;
    private int HiddenSize => _options.HiddenSize;
    private int NumberOfLayers => _options.NumberOfLayers;
    private int NumberOfHeads => _options.NumberOfHeads;
    private int BlockSize => _options.BlockSize;
    private int FeedForwardSize => _options.FeedForwardSize;
    private double Dropout => _options.Dropout;
    private int InputStreams => _inputStreams;
    private int OutputHeads => _outputHeads;

    internal BarkStageOptions Options => _options.Copy();

    internal bool IsCausal => _options.IsCausal;

    internal bool IsCacheActive => _cache is not null && _blocks.All(
        block => block.AttentionLayer is CachedMultiHeadAttention<T> { InferenceMode: true });

    internal int CachedTokenCount => _cache?.CurrentLength ?? 0;

    public override bool SupportsTraining => true;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 =>
        [
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_options.OutputVocabularySize)),
        ],
        2 =>
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_options.OutputVocabularySize)),
        ],
        _ => null,
    };

    internal long EstimatedParameterCount
    {
        get
        {
            long hidden = _options.HiddenSize;
            long perBlock = 4L * hidden * hidden + 2L * hidden
                + 2L * hidden * (_options.FeedForwardSize > 0
                    ? _options.FeedForwardSize
                    : checked(_options.HiddenSize * 4))
                + (_options.FeedForwardSize > 0
                    ? _options.FeedForwardSize
                    : checked(_options.HiddenSize * 4))
                + hidden;
            return checked(
                (long)_inputStreams * _options.InputVocabularySize * hidden
                + (long)_options.BlockSize * hidden
                + (long)_options.NumberOfLayers * perBlock
                + 2L * hidden
                + (long)_outputHeads * (hidden * _options.OutputVocabularySize + _options.OutputVocabularySize));
        }
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => ForwardSingleStream(input, useCache: false, positionOffset: 0, outputHead: 0);

    internal Tensor<T> ForwardUncached(Tensor<T> tokenIds, int outputHead = 0)
        => ForwardSingleStream(tokenIds, useCache: false, positionOffset: 0, outputHead);

    internal Tensor<T> BeginCachedSequence(Tensor<T> promptTokenIds, int outputHead = 0)
    {
        if (!_options.IsCausal)
            throw new InvalidOperationException("KV caching is only valid for Bark's causal semantic and coarse stages.");

        EnsureCachedAttentionInstalled();
        _cache!.Clear();
        SetCachedInferenceMode(true);
        return ForwardSingleStream(promptTokenIds, useCache: true, positionOffset: 0, outputHead);
    }

    internal Tensor<T> AppendCachedToken(int tokenId, int outputHead = 0)
    {
        if (_cache is null || !_cachedAttentionInstalled)
            throw new InvalidOperationException("BeginCachedSequence must be called before appending a token.");
        if (_cache.CurrentLength >= _options.BlockSize)
            throw new InvalidOperationException(
                $"The Bark {_options.BlockSize}-token KV cache is full. Start a new sliding-window prefill before appending.");

        var token = new Tensor<T>([1, 1]);
        token[0, 0] = NumOps.FromDouble(tokenId);
        SetCachedInferenceMode(true);
        return ForwardSingleStream(token, useCache: true, positionOffset: _cache.CurrentLength, outputHead);
    }

    internal Tensor<T> ForwardCodebooks(int[,] codebooks, int outputHead)
    {
        if (codebooks is null) throw new ArgumentNullException(nameof(codebooks));
        if (codebooks.GetLength(0) != _inputStreams)
            throw new ArgumentException(
                $"This Bark stage expects {_inputStreams} codebook streams, but received {codebooks.GetLength(0)}.",
                nameof(codebooks));
        ValidateHead(outputHead);

        int frames = codebooks.GetLength(1);
        ValidateSequenceLength(frames);
        var hidden = new Tensor<T>([1, frames, _options.HiddenSize]);
        // Bark fine head h predicts codebook h + n_codes_given and may only observe embeddings
        // through that target codebook. Later codebooks are padding, not model inputs.
        int targetCodebook = outputHead + (_inputStreams - _outputHeads);
        int participatingStreams = Math.Min(_inputStreams, targetCodebook + 1);
        for (int stream = 0; stream < participatingStreams; stream++)
        {
            var tokens = new Tensor<T>([1, frames]);
            for (int frame = 0; frame < frames; frame++)
                tokens[0, frame] = NumOps.FromDouble(codebooks[stream, frame]);
            hidden = Engine.TensorAdd(hidden, _tokenEmbeddings[stream].Forward(tokens));
        }

        hidden = AddPositions(hidden, frames, 0);
        return ProjectHidden(RunBlocks(hidden, useCache: false), 1, frames, outputHead);
    }

    internal Vector<T> LastTokenLogits(Tensor<T> logits)
    {
        bool rankThree = logits.Shape.Length == 3;
        bool rankTwo = logits.Shape.Length == 2;
        if ((!rankThree && !rankTwo)
            || rankThree && (logits.Shape[0] != 1 || logits.Shape[2] != _options.OutputVocabularySize)
            || rankTwo && logits.Shape[1] != _options.OutputVocabularySize)
        {
            throw new ArgumentException(
                $"Expected Bark logits [time, {_options.OutputVocabularySize}] or [1, time, {_options.OutputVocabularySize}], got [{string.Join(", ", logits.Shape)}].",
                nameof(logits));
        }

        int last = rankThree ? logits.Shape[1] - 1 : logits.Shape[0] - 1;
        var values = new T[_options.OutputVocabularySize];
        for (int token = 0; token < values.Length; token++)
            values[token] = rankThree ? logits[0, last, token] : logits[last, token];
        return new Vector<T>(values);
    }

    internal void ResetCache()
    {
        _cache?.Clear();
        SetCachedInferenceMode(false);
    }

    public override void ResetState()
    {
        for (int stream = 0; stream < _tokenEmbeddings.Length; stream++)
            _tokenEmbeddings[stream].ResetState();
        _positionEmbeddings.ResetState();
        for (int layer = 0; layer < _blocks.Length; layer++)
            _blocks[layer].ResetState();
        _finalNormalization.ResetState();
        for (int head = 0; head < _languageModelHeads.Length; head++)
            _languageModelHeads[head].ResetState();
        ResetCache();
    }

    private Tensor<T> ForwardSingleStream(
        Tensor<T> tokenIds,
        bool useCache,
        int positionOffset,
        int outputHead)
    {
        if (_inputStreams != 1)
            throw new InvalidOperationException("Use ForwardCodebooks for a multi-codebook Bark stage.");
        ValidateHead(outputHead);

        bool wasRankOne = tokenIds.Shape.Length == 1;
        Tensor<T> normalized;
        if (wasRankOne)
            normalized = Engine.Reshape(tokenIds, [1, tokenIds.Shape[0]]);
        else if (tokenIds.Shape.Length == 2)
            normalized = tokenIds;
        else
            throw new ArgumentException("Bark token input must have shape [time] or [batch, time].", nameof(tokenIds));

        int batch = normalized.Shape[0];
        int sequence = normalized.Shape[1];
        if (batch != 1 && useCache)
            throw new ArgumentException("The contiguous Bark KV cache currently supports one sequence per generation session.", nameof(tokenIds));
        ValidateSequenceLength(sequence, positionOffset);

        if (_cachedAttentionInstalled)
            SetCachedInferenceMode(useCache);

        var hidden = _tokenEmbeddings[0].Forward(normalized);
        hidden = AddPositions(hidden, sequence, positionOffset);
        hidden = RunBlocks(hidden, useCache);
        var logits = ProjectHidden(hidden, batch, sequence, outputHead);
        return wasRankOne
            ? Engine.Reshape(logits, [sequence, _options.OutputVocabularySize])
            : logits;
    }

    private Tensor<T> AddPositions(Tensor<T> hidden, int sequence, int positionOffset)
    {
        var positions = new Tensor<T>([hidden.Shape[0], sequence]);
        for (int batch = 0; batch < hidden.Shape[0]; batch++)
        {
            for (int position = 0; position < sequence; position++)
                positions[batch, position] = NumOps.FromDouble(positionOffset + position);
        }
        return Engine.TensorAdd(hidden, _positionEmbeddings.Forward(positions));
    }

    private Tensor<T> RunBlocks(Tensor<T> hidden, bool useCache)
    {
        for (int layer = 0; layer < _blocks.Length; layer++)
            hidden = _blocks[layer].Forward(hidden);
        return _finalNormalization.Forward(hidden);
    }

    private Tensor<T> ProjectHidden(Tensor<T> hidden, int batch, int sequence, int outputHead)
    {
        var flat = Engine.Reshape(hidden, [checked(batch * sequence), _options.HiddenSize]);
        var logits = _languageModelHeads[outputHead].Forward(flat);
        return Engine.Reshape(logits, [batch, sequence, _options.OutputVocabularySize]);
    }

    private void EnsureCachedAttentionInstalled()
    {
        if (_cachedAttentionInstalled) return;

        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            if (_blocks[layer].AttentionLayer is not MultiHeadAttentionLayer<T> source)
                throw new InvalidOperationException("Bark can only install its KV cache over standard multi-head attention.");

            var replacement = new CachedMultiHeadAttention<T>(
                _options.BlockSize,
                _options.HiddenSize,
                _options.NumberOfHeads,
                useFlashAttention: true,
                layerIndex: layer,
                useCausalMask: true,
                activationFunction: new IdentityActivation<T>());
            replacement.SetParameters(source.GetParameters());
            _blocks[layer].ReplaceAttention(replacement);
        }

        _cache = new KVCache<T>(new KVCacheConfig
        {
            NumLayers = _blocks.Length,
            NumHeads = _options.NumberOfHeads,
            HeadDimension = _options.HiddenSize / _options.NumberOfHeads,
            MaxSequenceLength = _options.BlockSize,
            MaxBatchSize = 1,
            PreAllocate = false,
        });
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            var attention = (CachedMultiHeadAttention<T>)_blocks[layer].AttentionLayer;
            attention.Cache = _cache;
            attention.LayerIndex = layer;
        }
        _cachedAttentionInstalled = true;
    }

    private void SetCachedInferenceMode(bool enabled)
    {
        if (!_cachedAttentionInstalled) return;
        for (int layer = 0; layer < _blocks.Length; layer++)
            ((CachedMultiHeadAttention<T>)_blocks[layer].AttentionLayer).InferenceMode = enabled;
    }

    private void ValidateSequenceLength(int sequence, int positionOffset = 0)
    {
        if (sequence <= 0)
            throw new ArgumentException("Bark transformer sequences cannot be empty.");
        if (positionOffset < 0 || sequence + positionOffset > _options.BlockSize)
            throw new ArgumentException(
                $"Bark sequence positions [{positionOffset}, {positionOffset + sequence}) exceed this stage's block size {_options.BlockSize}.");
    }

    private void ValidateHead(int outputHead)
    {
        if (outputHead < 0 || outputHead >= _outputHeads)
            throw new ArgumentOutOfRangeException(nameof(outputHead));
    }
}
