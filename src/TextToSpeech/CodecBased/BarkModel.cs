using System.Diagnostics;
using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Generation;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Identifies a stage in Bark's semantic-to-waveform generation pipeline.</summary>
public enum BarkGenerationStage
{
    /// <summary>Text to semantic audio tokens.</summary>
    Semantic,
    /// <summary>Semantic tokens to the first two EnCodec codebooks.</summary>
    Coarse,
    /// <summary>Bidirectional refinement of the remaining EnCodec codebooks.</summary>
    Fine,
    /// <summary>EnCodec token-to-waveform decoding.</summary>
    Codec,
}

/// <summary>Stage-specific sampling and generation controls.</summary>
public sealed class BarkGenerationOptions
{
    /// <summary>Semantic sampling controls.</summary>
    public SamplingOptions SemanticSampling { get; init; } = new() { Temperature = 0.7, TopK = 50 };

    /// <summary>Coarse sampling controls.</summary>
    public SamplingOptions CoarseSampling { get; init; } = new() { Temperature = 0.7, TopK = 50 };

    /// <summary>Fine-acoustic sampling controls. Bark's released pipeline uses temperature 0.5.</summary>
    public SamplingOptions FineSampling { get; init; } = new() { Temperature = 0.5, TopK = 50 };

    /// <summary>Maximum semantic tokens; null uses <see cref="BarkOptions.MaxSemanticNewTokens"/>.</summary>
    public int? MaxSemanticTokens { get; init; }

    /// <summary>Stop semantic generation when its special padding/EOS token is selected.</summary>
    public bool AllowEarlyStop { get; init; } = true;

    /// <summary>Probability threshold for Bark's semantic EOS token. Zero disables the threshold.</summary>
    public double SemanticEarlyStopProbability { get; init; } = 0.2;

    internal void Validate()
    {
        if (MaxSemanticTokens is < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxSemanticTokens));
        if (SemanticEarlyStopProbability < 0.0 || SemanticEarlyStopProbability > 1.0)
            throw new ArgumentOutOfRangeException(nameof(SemanticEarlyStopProbability));
    }
}

/// <summary>Optional semantic and acoustic history used to preserve a Bark voice/prompt.</summary>
public sealed class BarkHistoryPrompt
{
    /// <summary>Prior semantic tokens.</summary>
    public IReadOnlyList<int> SemanticTokens { get; init; } = Array.Empty<int>();

    /// <summary>Prior EnCodec tokens shaped [codebook, frame].</summary>
    public int[,]? CodecTokens { get; init; }
}

/// <summary>Structured Bark output, including every intermediate representation.</summary>
public sealed class BarkGenerationResult<T>
{
    internal BarkGenerationResult(
        IReadOnlyList<int> semanticTokens,
        int[,] coarseTokens,
        int[,] fineTokens,
        Tensor<T> audio,
        IReadOnlyDictionary<BarkGenerationStage, TimeSpan> stageDurations)
    {
        SemanticTokens = semanticTokens;
        CoarseTokens = coarseTokens;
        FineTokens = fineTokens;
        Audio = audio;
        StageDurations = stageDurations;
    }

    /// <summary>Generated semantic audio tokens.</summary>
    public IReadOnlyList<int> SemanticTokens { get; }

    /// <summary>The two generated coarse EnCodec codebooks.</summary>
    public int[,] CoarseTokens { get; }

    /// <summary>All refined EnCodec codebooks.</summary>
    public int[,] FineTokens { get; }

    /// <summary>Decoded 24 kHz waveform.</summary>
    public Tensor<T> Audio { get; }

    /// <summary>Wall-clock duration of each pipeline stage.</summary>
    public IReadOnlyDictionary<BarkGenerationStage, TimeSpan> StageDurations { get; }
}

/// <summary>
/// Low-level, checkpoint-faithful Bark foundation model.
/// </summary>
/// <typeparam name="T">Numeric type used by the neural network.</typeparam>
/// <remarks>
/// <para>
/// Bark has four distinct stages: causal semantic GPT, causal coarse GPT, bidirectional fine GPT,
/// and EnCodec. This class is the one neural implementation used by the high-level
/// <see cref="Bark{T}"/> façade. The stage layers live in the standard model layer graph, so base
/// lifecycle and generator automation own parameters, gradients, cloning, serialization, and
/// shape discovery.
/// </para>
/// <para><b>For Beginners:</b> Use <see cref="Bark{T}"/> when starting from text. Use this class
/// when you already have tokenizer IDs, want intermediate audio tokens, or are training one stage.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bark: Text-Prompted Generative Audio Model", "https://github.com/suno-ai/bark")]
public partial class BarkModel<T> : TtsModelBase<T>
{
    private enum TrainingStage
    {
        Semantic,
        Coarse,
        Fine,
    }

    private readonly BarkOptions _options;
    // Writable aliases let the generated lifecycle rebind these named views when the canonical
    // Layers graph is replaced during restore/clone. They are assigned only by construction or
    // that generated rebind hook; model authors write no parameter plumbing.
    private BarkTransformerStageLayer<T> _semantic;
    private BarkTransformerStageLayer<T> _coarse;
    private BarkTransformerStageLayer<T> _fine;
    private readonly BarkCodecLayer<T> _codec;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private TrainingStage _trainingStage;
    private int _fineTrainingHead;
    private bool _disposed;

    /// <summary>Creates a full Bark model with the paper/checkpoint defaults.</summary>
    public BarkModel(
        BarkOptions? options = null,
        IAudioCodec<T>? codec = null,
        int? seed = null)
        : this(CreateArchitecture(options ?? new BarkOptions(), seed), options, codec)
    {
    }

    /// <summary>Creates Bark using an explicit framework architecture descriptor.</summary>
    public BarkModel(
        NeuralNetworkArchitecture<T> architecture,
        BarkOptions? options = null,
        IAudioCodec<T>? codec = null)
        : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
            new CrossEntropyWithLogitsLoss<T>(classAxis: -1))
    {
        _options = new BarkOptions(options ?? new BarkOptions());
        _options.Validate();
        Options = _options;
        SampleRate = _options.SampleRate;
        HiddenDim = _options.Semantic.HiddenSize;
        MelChannels = 0;
        HopSize = Math.Max(1, _options.SampleRate / _options.CodecFrameRate);

        _semantic = new BarkTransformerStageLayer<T>(_options.Semantic);
        _coarse = new BarkTransformerStageLayer<T>(_options.Coarse);
        _fine = new BarkTransformerStageLayer<T>(
            _options.Fine,
            inputStreams: _options.NumCodebooks,
            outputHeads: _options.NumCodebooks - _options.NumberOfFineCodebooksGiven);
        _codec = new BarkCodecLayer<T>(codec ?? CreateDefaultCodec(_options));
        InitializeLayers();
        StreamingTraining = StreamingTrainingMode.Auto;
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>The validated Bark configuration.</summary>
    public BarkOptions BarkConfiguration => new(_options);

    /// <summary>Number of EnCodec residual-vector-quantization codebooks.</summary>
    public int NumberOfCodebooks => _options.NumCodebooks;

    /// <summary>EnCodec codebook vocabulary size.</summary>
    public int CodebookSize => _options.CodebookSize;

    /// <summary>EnCodec frame rate.</summary>
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <summary>
    /// Exact structural parameter estimate without materializing any lazy transformer weights.
    /// </summary>
    public long EstimatedBarkTransformerParameterCount => checked(
        _semantic.EstimatedParameterCount + _coarse.EstimatedParameterCount + _fine.EstimatedParameterCount);

    /// <summary>Whether the semantic stage currently has an active KV-cache.</summary>
    public bool IsSemanticCacheActive => _semantic.IsCacheActive;

    /// <summary>Whether the coarse stage currently has an active KV-cache.</summary>
    public bool IsCoarseCacheActive => _coarse.IsCacheActive;

    /// <summary>Raised as a generation request enters each pipeline stage.</summary>
    public event Action<BarkGenerationStage>? StageStarted;

    /// <inheritdoc />
    protected override int OutputFeatureWidth => _options.Semantic.OutputVocabularySize;

    /// <summary>Computes semantic next-token logits without starting generation.</summary>
    public Tensor<T> PredictSemanticLogits(IReadOnlyList<int> tokenIds)
        => _semantic.ForwardUncached(ToTokenTensor(tokenIds));

    /// <summary>Prefills the semantic KV-cache and returns next-token logits.</summary>
    public Vector<T> BeginSemanticSequence(IReadOnlyList<int> tokenIds)
        => _semantic.LastTokenLogits(_semantic.BeginCachedSequence(ToTokenTensor(tokenIds)));

    /// <summary>Appends one semantic token to the KV-cache and returns next-token logits.</summary>
    public Vector<T> AppendSemanticToken(int tokenId)
        => _semantic.LastTokenLogits(_semantic.AppendCachedToken(tokenId));

    /// <summary>Clears semantic and coarse generation caches.</summary>
    public void ResetGenerationCaches()
    {
        _semantic.ResetCache();
        _coarse.ResetCache();
    }

    /// <summary>Generates semantic audio tokens from tokenizer IDs.</summary>
    public IReadOnlyList<int> GenerateSemanticTokens(
        IReadOnlyList<int> textTokenIds,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (textTokenIds is null) throw new ArgumentNullException(nameof(textTokenIds));
        generationOptions ??= new BarkGenerationOptions();
        generationOptions.Validate();
        cancellationToken.ThrowIfCancellationRequested();

        var context = BuildSemanticPrompt(textTokenIds, history);
        int maxNewTokens = generationOptions.MaxSemanticTokens ?? _options.MaxSemanticNewTokens;
        var generated = new List<int>(maxNewTokens);
        Vector<T>? nextLogits = null;

        for (int step = 0; step < maxNewTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (step == 0 || !_options.UseKeyValueCache || nextLogits is null)
            {
                var visible = Tail(context, _options.Semantic.BlockSize);
                var logits = _options.UseKeyValueCache
                    ? _semantic.BeginCachedSequence(ToTokenTensor(visible))
                    : _semantic.ForwardUncached(ToTokenTensor(visible));
                nextLogits = _semantic.LastTokenLogits(logits);
            }

            var masked = RestrictLogits(
                nextLogits!,
                0,
                _options.SemanticVocabularySize,
                generationOptions.AllowEarlyStop ? _options.SemanticPadTokenId : null);
            int next = Sample(masked, generationOptions.SemanticSampling);
            bool reachedEosProbability = generationOptions.AllowEarlyStop
                && generationOptions.SemanticEarlyStopProbability > 0.0
                && TokenSampler<T>.ProbabilityOf(
                    masked,
                    generationOptions.SemanticSampling,
                    _options.SemanticPadTokenId) >= generationOptions.SemanticEarlyStopProbability;
            if (generationOptions.AllowEarlyStop
                && (next == _options.SemanticPadTokenId || reachedEosProbability))
                break;
            if (next >= _options.SemanticVocabularySize)
                continue;

            generated.Add(next);
            context.Add(next);
            if (_options.UseKeyValueCache && step + 1 < maxNewTokens)
            {
                if (_semantic.CachedTokenCount < _options.Semantic.BlockSize)
                    nextLogits = _semantic.LastTokenLogits(_semantic.AppendCachedToken(next));
                else
                    nextLogits = null;
            }
        }

        return generated;
    }

    /// <summary>Generates Bark's two coarse EnCodec codebooks from semantic tokens.</summary>
    public int[,] GenerateCoarseTokens(
        IReadOnlyList<int> semanticTokens,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (semanticTokens is null) throw new ArgumentNullException(nameof(semanticTokens));
        if (semanticTokens.Count == 0)
            throw new ArgumentException("Bark coarse generation requires at least one semantic token.", nameof(semanticTokens));
        generationOptions ??= new BarkGenerationOptions();
        generationOptions.Validate();

        double semanticToCoarseRatio = _options.CoarseRateHz / _options.SemanticRateHz
            * _options.NumberOfCoarseCodebooks;
        int frames = Math.Max(1, (int)Math.Floor(
            semanticTokens.Count * semanticToCoarseRatio / _options.NumberOfCoarseCodebooks));
        int flattenedCount = checked(frames * _options.NumberOfCoarseCodebooks);
        var flatOutput = new int[flattenedCount];
        var (semanticContext, coarseContext, baseSemanticIndex) = BuildCoarseContexts(
            semanticTokens,
            history,
            semanticToCoarseRatio);
        List<int>? windowContext = null;
        Vector<T>? nextLogits = null;

        for (int step = 0; step < flattenedCount; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            bool startsWindow = step % _options.CoarseSlidingWindowLength == 0;
            if (startsWindow)
            {
                windowContext = BuildCoarseWindow(
                    semanticContext,
                    coarseContext,
                    baseSemanticIndex,
                    step,
                    semanticToCoarseRatio);
                var logits = _options.UseKeyValueCache
                    ? _coarse.BeginCachedSequence(ToTokenTensor(windowContext))
                    : _coarse.ForwardUncached(ToTokenTensor(windowContext));
                nextLogits = _coarse.LastTokenLogits(logits);
            }
            else if (!_options.UseKeyValueCache)
            {
                nextLogits = _coarse.LastTokenLogits(
                    _coarse.ForwardUncached(ToTokenTensor(windowContext!)));
            }

            int codebook = step % _options.NumberOfCoarseCodebooks;
            int tokenStart = checked(_options.SemanticVocabularySize + codebook * _options.CodebookSize);
            var masked = RestrictLogits(nextLogits!, tokenStart, tokenStart + _options.CodebookSize);
            int sampled = Sample(masked, generationOptions.CoarseSampling);
            flatOutput[step] = sampled - tokenStart;
            coarseContext.Add(sampled);
            windowContext!.Add(sampled);

            bool endsWindow = (step + 1) % _options.CoarseSlidingWindowLength == 0;
            if (_options.UseKeyValueCache && step + 1 < flattenedCount && !endsWindow)
            {
                nextLogits = _coarse.LastTokenLogits(_coarse.AppendCachedToken(sampled));
            }
        }

        var result = new int[_options.NumberOfCoarseCodebooks, frames];
        for (int frame = 0; frame < frames; frame++)
        {
            for (int codebook = 0; codebook < _options.NumberOfCoarseCodebooks; codebook++)
                result[codebook, frame] = flatOutput[frame * _options.NumberOfCoarseCodebooks + codebook];
        }
        return result;
    }

    /// <summary>Refines coarse tokens into all eight Bark/EnCodec codebooks.</summary>
    public int[,] GenerateFineTokens(
        int[,] coarseTokens,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (coarseTokens is null) throw new ArgumentNullException(nameof(coarseTokens));
        if (coarseTokens.GetLength(0) != _options.NumberOfCoarseCodebooks)
            throw new ArgumentException(
                $"Bark expects {_options.NumberOfCoarseCodebooks} coarse codebooks.",
                nameof(coarseTokens));

        generationOptions ??= new BarkGenerationOptions();
        generationOptions.Validate();
        int frames = coarseTokens.GetLength(1);
        if (frames <= 0)
            throw new ArgumentException("Bark fine generation requires at least one coarse frame.", nameof(coarseTokens));

        int historyFrames = 0;
        int[,]? fineHistory = history?.CodecTokens;
        if (fineHistory is not null)
        {
            if (fineHistory.GetLength(0) != _options.NumCodebooks)
                throw new ArgumentException(
                    $"Bark fine history must contain {_options.NumCodebooks} codebooks.",
                    nameof(history));
            historyFrames = Math.Min(fineHistory.GetLength(1), _options.FineWindowStride);
        }

        int window = Math.Min(_options.FineWindowLength, _options.Fine.BlockSize);
        int unpaddedFrames = checked(historyFrames + frames);
        int padFrames = Math.Max(0, window - unpaddedFrames);
        int workingFrames = checked(unpaddedFrames + padFrames);
        var working = new int[_options.NumCodebooks, workingFrames];
        for (int codebook = 0; codebook < _options.NumCodebooks; codebook++)
        {
            for (int frame = 0; frame < historyFrames; frame++)
            {
                int sourceFrame = fineHistory!.GetLength(1) - historyFrames + frame;
                working[codebook, frame] = ValidateCodebookToken(
                    fineHistory[codebook, sourceFrame],
                    nameof(history));
            }
            for (int frame = 0; frame < frames; frame++)
            {
                working[codebook, historyFrames + frame] = codebook < _options.NumberOfCoarseCodebooks
                    ? ValidateCodebookToken(coarseTokens[codebook, frame], nameof(coarseTokens))
                    : _options.CodebookSize;
            }
            for (int frame = unpaddedFrames; frame < workingFrames; frame++)
                working[codebook, frame] = _options.CodebookSize;
        }

        int loops = Math.Max(
            0,
            (int)Math.Ceiling((frames - (window - historyFrames)) / (double)_options.FineWindowStride)) + 1;
        for (int loop = 0; loop < loops; loop++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = Math.Min(loop * _options.FineWindowStride, workingFrames - window);
            int fillStart = Math.Min(
                historyFrames + loop * _options.FineWindowStride,
                workingFrames - _options.FineWindowStride);
            int relativeFillStart = fillStart - start;
            var slice = new int[_options.NumCodebooks, window];
            for (int codebook = 0; codebook < _options.NumCodebooks; codebook++)
                for (int frame = 0; frame < window; frame++)
                    slice[codebook, frame] = working[codebook, start + frame];

            // The checkpoint contains heads 1..7 (n_codes_given=1), while standard Bark generation
            // supplies and preserves two coarse codebooks and therefore invokes heads 2..7.
            for (int codebook = _options.NumberOfCoarseCodebooks;
                 codebook < _options.NumCodebooks;
                 codebook++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var logits = _fine.ForwardCodebooks(
                    slice,
                    codebook - _options.NumberOfFineCodebooksGiven);
                for (int frame = relativeFillStart; frame < window; frame++)
                {
                    var frameLogits = ExtractPositionLogits(logits, frame, _options.CodebookSize);
                    int token = Sample(frameLogits, generationOptions.FineSampling);
                    slice[codebook, frame] = token;
                }
            }

            for (int codebook = _options.NumberOfCoarseCodebooks;
                 codebook < _options.NumCodebooks;
                 codebook++)
            {
                for (int frame = relativeFillStart; frame < window; frame++)
                    working[codebook, start + frame] = slice[codebook, frame];
            }
        }

        var fine = new int[_options.NumCodebooks, frames];
        for (int codebook = 0; codebook < _options.NumCodebooks; codebook++)
            for (int frame = 0; frame < frames; frame++)
                fine[codebook, frame] = working[codebook, historyFrames + frame];
        return fine;
    }

    /// <summary>Decodes Bark/EnCodec tokens to a waveform.</summary>
    public Tensor<T> DecodeAudioTokens(int[,] tokens)
    {
        ThrowIfDisposed();
        ValidateCodecTokens(tokens);
        return _codec.Decode(tokens);
    }

    /// <summary>Encodes a waveform into Bark/EnCodec tokens.</summary>
    public int[,] EncodeAudio(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        return _codec.Encode(audio);
    }

    /// <summary>Runs all four Bark stages from already-tokenized text.</summary>
    public BarkGenerationResult<T> Generate(
        IReadOnlyList<int> textTokenIds,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        var timings = new Dictionary<BarkGenerationStage, TimeSpan>();
        var stopwatch = Stopwatch.StartNew();

        StageStarted?.Invoke(BarkGenerationStage.Semantic);
        var semantic = GenerateSemanticTokens(textTokenIds, generationOptions, history, cancellationToken);
        timings[BarkGenerationStage.Semantic] = stopwatch.Elapsed;

        StageStarted?.Invoke(BarkGenerationStage.Coarse);
        stopwatch.Restart();
        var coarse = GenerateCoarseTokens(semantic, generationOptions, history, cancellationToken);
        timings[BarkGenerationStage.Coarse] = stopwatch.Elapsed;

        StageStarted?.Invoke(BarkGenerationStage.Fine);
        stopwatch.Restart();
        var fine = GenerateFineTokens(coarse, generationOptions, history, cancellationToken);
        timings[BarkGenerationStage.Fine] = stopwatch.Elapsed;

        StageStarted?.Invoke(BarkGenerationStage.Codec);
        stopwatch.Restart();
        var audio = DecodeAudioTokens(fine);
        timings[BarkGenerationStage.Codec] = stopwatch.Elapsed;

        return new BarkGenerationResult<T>(semantic, coarse, fine, audio, timings);
    }

    /// <summary>Asynchronous four-stage generation with cooperative cancellation.</summary>
    public async Task<BarkGenerationResult<T>> GenerateAsync(
        IReadOnlyList<int> textTokenIds,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        cancellationToken.ThrowIfCancellationRequested();
        return Generate(textTokenIds, generationOptions, history, cancellationToken);
    }

    /// <summary>Trains the semantic causal language model with categorical next-token targets.</summary>
    public void TrainSemantic(Tensor<T> inputTokenIds, Tensor<T> expectedTokenDistribution)
        => TrainStage(TrainingStage.Semantic, 0, inputTokenIds, expectedTokenDistribution);

    /// <summary>Trains the coarse causal language model with categorical next-token targets.</summary>
    public void TrainCoarse(Tensor<T> inputTokenIds, Tensor<T> expectedTokenDistribution)
        => TrainStage(TrainingStage.Coarse, 0, inputTokenIds, expectedTokenDistribution);

    /// <summary>Trains one fine codebook head with categorical token targets.</summary>
    public void TrainFine(
        Tensor<T> inputCodebooks,
        int targetCodebook,
        Tensor<T> expectedTokenDistribution)
    {
        if (targetCodebook < _options.NumberOfFineCodebooksGiven || targetCodebook >= _options.NumCodebooks)
            throw new ArgumentOutOfRangeException(nameof(targetCodebook));
        TrainStage(
            TrainingStage.Fine,
            targetCodebook - _options.NumberOfFineCodebooksGiven,
            inputCodebooks,
            expectedTokenDistribution);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
        => TrainSemantic(input, expected);

    protected override void InitializeLayers()
    {
        if (Layers.Count != 0) return;
        Layers.Add(_semantic);
        Layers.Add(_coarse);
        Layers.Add(_fine);
        // The injected codec remains a generated extra-layer component rather than a serialized
        // sequential layer. Its concrete implementation can be external, while the base lifecycle
        // still discovers all of its trainable child layers through GetExtraTrainableLayers().
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        return _trainingStage switch
        {
            TrainingStage.Coarse => _coarse.ForwardUncached(input),
            TrainingStage.Fine => _fine.ForwardCodebooks(ToCodebookArray(input), _fineTrainingHead),
            _ => _semantic.ForwardUncached(input),
        };
    }

    /// <inheritdoc />
    /// <remarks>
    /// Bark's three transformers are parallel training objectives, not a sequential layer chain:
    /// semantic logits are never token IDs for the coarse embedding, and coarse logits are never
    /// fine-codebook inputs. The base still owns the tape, optimizer, gradients, parameters, and
    /// lifecycle; this override supplies only Bark's unique stage routing.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return PredictCore(input);
    }

    protected override Tensor<T> PreprocessText(string text)
        => throw new NotSupportedException(
            "BarkModel is the tokenizer-level API. Use Bark<T>.Synthesize(text), or call Generate with tokenizer IDs.");

    protected override Tensor<T> PostprocessAudio(Tensor<T> modelOutput) => modelOutput;

    protected override long EstimateStructuralParameterCount() => EstimatedBarkTransformerParameterCount;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Bark",
            Version = "1.0",
            Description = "Suno Bark semantic/coarse/fine GPT pipeline with EnCodec waveform decoding",
            FeatureCount = _options.Semantic.HiddenSize,
            Complexity = EstimatedBarkTransformerParameterCount,
        };
        metadata.SetProperty("architecture", "semantic-gpt/coarse-gpt/fine-transformer/encodec");
        metadata.SetProperty("semantic_causal", true);
        metadata.SetProperty("coarse_causal", true);
        metadata.SetProperty("fine_bidirectional", true);
        metadata.SetProperty("kv_cache", _options.UseKeyValueCache);
        metadata.SetProperty("sample_rate", _options.SampleRate);
        metadata.SetProperty("codebooks", _options.NumCodebooks);
        metadata.SetProperty("codebook_size", _options.CodebookSize);
        metadata.SetProperty("structural_parameter_count", EstimatedBarkTransformerParameterCount);
        metadata.AdditionalInfo["Architecture"] = "semantic-gpt/coarse-gpt/fine-transformer/encodec";
        metadata.AdditionalInfo["SemanticCausal"] = true;
        metadata.AdditionalInfo["CoarseCausal"] = true;
        metadata.AdditionalInfo["FineBidirectional"] = true;
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate;
        metadata.AdditionalInfo["Codebooks"] = _options.NumCodebooks;
        metadata.AdditionalInfo["CodebookSize"] = _options.CodebookSize;
        return metadata;
    }





    /// <summary>
    /// Recreates the injected codec dependency for cloning while leaving Bark parameter transfer to
    /// the framework's single generated layer manifest.
    /// </summary>
    protected IAudioCodec<T> CreateCodecForNewInstance()
    {
        if (_codec.Codec is IFullModel<T, Tensor<T>, Tensor<T>> model
            && model.Clone() is IAudioCodec<T> clonedCodec)
        {
            return clonedCodec;
        }

        // Stateless/custom codecs do not expose a clone contract. Sharing that external service is
        // safe; all Bark-owned trainable state still lives in the generated layer graph.
        return _codec.Codec;
    }

    private void TrainStage(
        TrainingStage stage,
        int fineHead,
        Tensor<T> input,
        Tensor<T> expected)
    {
        ThrowIfDisposed();
        _trainingStage = stage;
        _fineTrainingHead = fineHead;
        SetTrainingMode(true);
        try
        {
            _optimizer ??= new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    WeightDecay = _options.WeightDecay,
                });
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
            _trainingStage = TrainingStage.Semantic;
            _fineTrainingHead = 0;
        }
    }

    private List<int> BuildSemanticPrompt(IReadOnlyList<int> textTokenIds, BarkHistoryPrompt? history)
    {
        var prompt = new List<int>(_options.MaxTextLength + _options.SemanticHistoryLength + 1);
        int textCount = Math.Min(textTokenIds.Count, _options.MaxTextLength);
        for (int index = 0; index < textCount; index++)
        {
            int encoded = checked(textTokenIds[index] + _options.TextEncodingOffset);
            if (encoded < 0 || encoded >= _options.Semantic.InputVocabularySize)
                throw new ArgumentOutOfRangeException(
                    nameof(textTokenIds),
                    $"Text token {textTokenIds[index]} plus offset {_options.TextEncodingOffset} is outside the semantic vocabulary.");
            prompt.Add(encoded);
        }
        while (prompt.Count < _options.MaxTextLength)
            prompt.Add(_options.TextPadTokenId);

        var semanticHistory = history?.SemanticTokens ?? Array.Empty<int>();
        int historyStart = Math.Max(0, semanticHistory.Count - _options.SemanticHistoryLength);
        for (int index = historyStart; index < semanticHistory.Count; index++)
            prompt.Add(semanticHistory[index]);
        while (prompt.Count < _options.MaxTextLength + _options.SemanticHistoryLength)
            prompt.Add(_options.SemanticPadTokenId);
        prompt.Add(_options.SemanticInferenceTokenId);
        return Tail(prompt, _options.Semantic.BlockSize);
    }

    private (List<int> Semantic, List<int> Coarse, int BaseSemanticIndex) BuildCoarseContexts(
        IReadOnlyList<int> semanticTokens,
        BarkHistoryPrompt? history,
        double semanticToCoarseRatio)
    {
        var semanticHistory = history?.SemanticTokens ?? Array.Empty<int>();
        var flattenedHistory = new List<int>();
        if (history?.CodecTokens is { } codecHistory)
        {
            if (codecHistory.GetLength(0) < _options.NumberOfCoarseCodebooks)
                throw new ArgumentException(
                    $"Bark coarse history must contain at least {_options.NumberOfCoarseCodebooks} codebooks.",
                    nameof(history));
            for (int frame = 0; frame < codecHistory.GetLength(1); frame++)
                for (int codebook = 0; codebook < _options.NumberOfCoarseCodebooks; codebook++)
                    flattenedHistory.Add(checked(
                        _options.SemanticVocabularySize
                        + codebook * _options.CodebookSize
                        + ValidateCodebookToken(codecHistory[codebook, frame], nameof(history))));
        }

        int maxSemanticHistory = Math.Max(1, (int)Math.Floor(
            _options.CoarseHistoryLength / semanticToCoarseRatio));
        int evenSemanticHistory = semanticHistory.Count - semanticHistory.Count % 2;
        int alignedSemanticCount = flattenedHistory.Count == 0
            ? 0
            : Math.Min(
                Math.Min(maxSemanticHistory, evenSemanticHistory),
                (int)Math.Floor(flattenedHistory.Count / semanticToCoarseRatio));
        int alignedCoarseCount = Math.Min(
            flattenedHistory.Count,
            (int)Math.Round(alignedSemanticCount * semanticToCoarseRatio));
        if (alignedCoarseCount >= _options.NumberOfCoarseCodebooks)
            alignedCoarseCount -= _options.NumberOfCoarseCodebooks;

        var semantic = new List<int>(alignedSemanticCount + semanticTokens.Count);
        for (int index = semanticHistory.Count - alignedSemanticCount; index < semanticHistory.Count; index++)
            semantic.Add(ValidateSemanticToken(semanticHistory[index], nameof(history)));
        int baseSemanticIndex = semantic.Count;
        for (int index = 0; index < semanticTokens.Count; index++)
            semantic.Add(ValidateSemanticToken(semanticTokens[index], nameof(semanticTokens)));

        var coarse = new List<int>(alignedCoarseCount);
        for (int index = flattenedHistory.Count - alignedCoarseCount; index < flattenedHistory.Count; index++)
            coarse.Add(flattenedHistory[index]);
        return (semantic, coarse, baseSemanticIndex);
    }

    private List<int> BuildCoarseWindow(
        IReadOnlyList<int> semanticContext,
        IReadOnlyList<int> coarseContext,
        int baseSemanticIndex,
        int generatedCoarseTokens,
        double semanticToCoarseRatio)
    {
        int maxSemanticHistory = Math.Max(1, (int)Math.Floor(
            _options.CoarseHistoryLength / semanticToCoarseRatio));
        int semanticIndex = baseSemanticIndex + (int)Math.Round(generatedCoarseTokens / semanticToCoarseRatio);
        int semanticStart = Math.Max(0, semanticIndex - maxSemanticHistory);
        var prompt = new List<int>(_options.Coarse.BlockSize);
        for (int index = semanticStart;
             index < semanticContext.Count && prompt.Count < _options.CoarseSemanticContextLength;
             index++)
        {
            prompt.Add(semanticContext[index]);
        }
        while (prompt.Count < _options.CoarseSemanticContextLength)
            prompt.Add(_options.CoarseSemanticPadTokenId);
        prompt.Add(_options.CoarseInferenceTokenId);
        int coarseStart = Math.Max(0, coarseContext.Count - _options.CoarseHistoryLength);
        for (int index = coarseStart; index < coarseContext.Count; index++) prompt.Add(coarseContext[index]);
        return prompt;
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(BarkOptions options, int? seed)
    {
        options.Validate();
        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.Semantic.BlockSize,
            outputSize: options.Semantic.OutputVocabularySize);
        architecture.RandomSeed = seed;
        return architecture;
    }

    private static IAudioCodec<T> CreateDefaultCodec(BarkOptions options)
    {
        bool tiny = options.Semantic.HiddenSize <= 32;
        var codecOptions = new EnCodecOptions
        {
            SampleRate = options.SampleRate,
            NumQuantizers = options.NumCodebooks,
            CodebookSize = options.CodebookSize,
            EncoderChannels = tiny ? [4, 8] : [32, 64, 128, 256, 512],
            DownsampleRatios = tiny ? [4, 2] : [8, 5, 4, 2],
            EncoderDim = tiny ? 8 : 128,
            CodebookDim = tiny ? 8 : 128,
        };
        var architecture = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: codecOptions.EncoderDim);
        return new EnCodec<T>(architecture, codecOptions);
    }

    private Tensor<T> ToTokenTensor(IReadOnlyList<int> tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count == 0) throw new ArgumentException("Bark token sequences cannot be empty.", nameof(tokens));
        var tensor = new Tensor<T>([1, tokens.Count]);
        for (int index = 0; index < tokens.Count; index++)
            tensor[0, index] = NumOps.FromDouble(tokens[index]);
        return tensor;
    }

    private int[,] ToCodebookArray(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[0] != _options.NumCodebooks)
            throw new ArgumentException(
                $"Fine training input must have shape [{_options.NumCodebooks}, frame].",
                nameof(input));
        var result = new int[input.Shape[0], input.Shape[1]];
        for (int codebook = 0; codebook < input.Shape[0]; codebook++)
            for (int frame = 0; frame < input.Shape[1]; frame++)
                result[codebook, frame] = Convert.ToInt32(NumOps.ToDouble(input[codebook, frame]));
        return result;
    }

    private Vector<T> RestrictLogits(Vector<T> logits, int startInclusive, int endExclusive, int? additionalToken = null)
    {
        if (startInclusive < 0 || endExclusive > logits.Length || startInclusive >= endExclusive)
            throw new ArgumentOutOfRangeException(nameof(startInclusive));
        T suppressed = NumOps.FromDouble(double.NegativeInfinity);
        var result = new T[logits.Length];
        for (int index = 0; index < logits.Length; index++)
            result[index] = index >= startInclusive && index < endExclusive || index == additionalToken
                ? logits[index]
                : suppressed;
        return new Vector<T>(result);
    }

    private Vector<T> ExtractPositionLogits(Tensor<T> logits, int position, int allowedVocabulary)
    {
        var values = new T[logits.Shape[2]];
        T suppressed = NumOps.FromDouble(double.NegativeInfinity);
        for (int token = 0; token < values.Length; token++)
            values[token] = token < allowedVocabulary ? logits[0, position, token] : suppressed;
        return new Vector<T>(values);
    }

    private static int Sample(Vector<T> logits, SamplingOptions options)
        => options.IsGreedy
            ? TokenSampler<T>.ArgMax(logits)
            : TokenSampler<T>.Sample(logits, options);

    private static List<int> Tail(IReadOnlyList<int> values, int count)
    {
        int start = Math.Max(0, values.Count - count);
        var result = new List<int>(values.Count - start);
        for (int index = start; index < values.Count; index++) result.Add(values[index]);
        return result;
    }

    private void ValidateCodecTokens(int[,] tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.GetLength(0) != _options.NumCodebooks)
            throw new ArgumentException(
                $"Bark codec input must contain {_options.NumCodebooks} codebooks.",
                nameof(tokens));
        for (int codebook = 0; codebook < tokens.GetLength(0); codebook++)
        {
            for (int frame = 0; frame < tokens.GetLength(1); frame++)
            {
                if (tokens[codebook, frame] < 0 || tokens[codebook, frame] >= _options.CodebookSize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokens),
                        $"Codec token [{codebook}, {frame}]={tokens[codebook, frame]} is outside [0, {_options.CodebookSize}).");
            }
        }
    }

    private int ValidateCodebookToken(int token, string parameterName)
    {
        if (token < 0 || token >= _options.CodebookSize)
            throw new ArgumentOutOfRangeException(
                parameterName,
                $"Bark codec token {token} is outside [0, {_options.CodebookSize}).");
        return token;
    }

    private int ValidateSemanticToken(int token, string parameterName)
    {
        if (token < 0 || token >= _options.SemanticVocabularySize)
            throw new ArgumentOutOfRangeException(
                parameterName,
                $"Bark semantic token {token} is outside [0, {_options.SemanticVocabularySize}).");
        return token;
    }

    private static void WriteStage(BinaryWriter writer, BarkStageOptions stage)
    {
        writer.Write(stage.InputVocabularySize);
        writer.Write(stage.OutputVocabularySize);
        writer.Write(stage.HiddenSize);
        writer.Write(stage.NumberOfLayers);
        writer.Write(stage.NumberOfHeads);
        writer.Write(stage.BlockSize);
        writer.Write(stage.FeedForwardSize);
        writer.Write(stage.Dropout);
        writer.Write(stage.IsCausal);
    }

    private static BarkStageOptions ReadStage(BinaryReader reader) => new()
    {
        InputVocabularySize = reader.ReadInt32(),
        OutputVocabularySize = reader.ReadInt32(),
        HiddenSize = reader.ReadInt32(),
        NumberOfLayers = reader.ReadInt32(),
        NumberOfHeads = reader.ReadInt32(),
        BlockSize = reader.ReadInt32(),
        FeedForwardSize = reader.ReadInt32(),
        Dropout = reader.ReadDouble(),
        IsCausal = reader.ReadBoolean(),
    };

    private static bool StageMatches(BarkStageOptions left, BarkStageOptions right)
        => left.InputVocabularySize == right.InputVocabularySize
           && left.OutputVocabularySize == right.OutputVocabularySize
           && left.HiddenSize == right.HiddenSize
           && left.NumberOfLayers == right.NumberOfLayers
           && left.NumberOfHeads == right.NumberOfHeads
           && left.BlockSize == right.BlockSize
           && left.FeedForwardSize == right.FeedForwardSize
           && Math.Abs(left.Dropout - right.Dropout) < 1e-12
           && left.IsCausal == right.IsCausal;

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BarkModel<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing && _codec.Codec is IDisposable disposable) disposable.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}
