using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of SoundStorm's two networks a training step updates.</summary>
public enum SoundStormTrainingStage
{
    /// <summary>The ByT5 text-to-semantic model's decoder (§5): text bytes to semantic tokens.</summary>
    TextToSemantic,

    /// <summary>The SoundStorm acoustic model (§3): semantic tokens to the SoundStream codes.</summary>
    SemanticToAcoustic,
}

/// <summary>
/// SoundStorm: efficient parallel audio generation (Borsos et al. 2023), a bidirectional Conformer that fills in a
/// SoundStream codec's RVQ codes from time-aligned semantic tokens, level by level and many tokens at a time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Acoustic model</b> (§3.1–3.3, <see cref="SoundStormCore{T}"/>): per frame, the embeddings of the Q = 12 codes
/// and of the semantic token are summed, so the Conformer's sequence is as long as the number of codec frames; a dense
/// head per level predicts that level's codes. A training step samples a level q, a prompt t ~ U{0, T − 1} and a
/// cosine-scheduled mask over the positions after t; level q is masked there, and so is every finer level, and the
/// loss is the cross-entropy on level q's masked positions only. Decoding fills the levels coarse to fine: the first
/// with 16 confidence-based (MaskGIT) iterations, the last of them greedy, and each finer level with one greedy pass.
/// </para>
/// <para>
/// <b>Text to semantic</b> (§5): ByT5-large reads the text's UTF-8 bytes and samples semantic tokens (temperature 0.9,
/// top 125), which are duplicated from 25 Hz to the codec's 50 Hz. Only the decoder trains; the encoder is a
/// text-pretrained ByT5 (<see cref="LoadTextEncoderWeights"/>). The semantic tokens share the decoder's vocabulary
/// after ByT5's 384 byte entries.
/// </para>
/// <para>
/// <b>Voice prompting</b> (§4.2): a voice's recording (<see cref="TtsVoice{T}.ReferenceAudio"/>) and its semantic tokens
/// at 25 Hz (<see cref="TtsVoice{T}.ReferenceTokens"/>) become the unmasked prefix of both models; without a voice the
/// speaker is sampled. Training needs each utterance's codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
/// <c>[frames, 1 + Q]</c>: the semantic token at the codec's frame rate, then the Q acoustic codes), or its semantic
/// tokens there and its recording, which the codec encodes.
/// </para>
/// <para>
/// What the paper leaves unstated, and its authors released no code for, follows lucidrains/soundstorm-pytorch: a
/// learned mask embedding, no dropout, convolution expansion 2, and its trainer's optimizer (see
/// <see cref="SoundStormOptions"/>). The Conformer's weights are drawn from N(0, 0.02) with zero biases, as Pheme's
/// SoundStorm implementation does. MaskGIT's confidence adds Gumbel noise annealed linearly to zero, as MaskGIT's
/// reference does.
/// </para>
/// <para><b>For Beginners:</b> SoundStorm turns "what is said" tokens into audio-codec tokens all at once instead of
/// one by one, which makes it about a hundred times faster than generating them in order.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new SoundStorm&lt;float&gt;(architecture, new SoundStormOptions());
/// model.LoadTextEncoderWeights("byt5-large.safetensors");
/// Tensor&lt;float&gt; audio = model.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SoundStorm: Efficient Parallel Audio Generation",
    "https://arxiv.org/abs/2305.09636",
    Year = 2023,
    Authors = "Borsos et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 0.5, Component = "semantic-to-acoustic", Provenance = RecipeProvenance.NamedReproduction,
    Source = "The paper states no optimizer and released no code. lucidrains/soundstorm-pytorch's SoundStormTrainer: "
             + "lr 3e-4 after a linear warmup from 1e-5, cosine annealing over the training steps, gradient norm "
             + "clipped to 0.5, weight decay 0, which makes audiolm_pytorch's get_optimizer return Adam with betas "
             + "(0.9, 0.99) and eps 1e-8.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 0.5, Component = "text-to-semantic", Provenance = RecipeProvenance.NamedReproduction,
    Source = "The paper states no optimizer for its ByT5 decoder (Section 5) and no reproduction trains one; the "
             + "acoustic model's lucidrains/soundstorm-pytorch recipe is reused.")]
public partial class SoundStorm<T> : TtsModelBase<T>, ICodecTts<T>
{
    // ByT5's vocabulary: <pad> 0, </s> 1, <unk> 2, then byte b at b + 3, padded to 384 entries.
    private const int PadToken = 0, EndToken = 1, ByteOffset = 3, ByteVocabulary = 384;

    private readonly SoundStormOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<SoundStormTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private bool _disposed;

    private readonly List<LayerBase<T>> _textLayers = new();
    private readonly List<LayerBase<T>> _acousticLayers = new();
    private T5Seq2Seq<T>? _textToSemantic;
    private SoundStormCore<T>? _acoustic;
    private AudioCodecLayer<T>? _codec;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed SoundStorm for inference.</summary>
    public SoundStorm(NeuralNetworkArchitecture<T> architecture, string modelPath, SoundStormOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SoundStormOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable SoundStorm.</summary>
    public SoundStorm(NeuralNetworkArchitecture<T> architecture, SoundStormOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SoundStormOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _options.VocabSize = ByteVocabulary + _options.SemanticCodes;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => _options.SampleRate / _options.HopSize;

    /// <summary>The network a training step updates.</summary>
    public SoundStormTrainingStage CurrentStage { get; set; } = SoundStormTrainingStage.TextToSemantic;

    /// <summary>Training steps taken by each network (they set the learning-rate schedules).</summary>
    public int TextToSemanticUpdates { get; private set; }

    /// <summary>Training steps taken by the acoustic network.</summary>
    public int SemanticToAcousticUpdates { get; private set; }

    /// <summary>The SoundStream codec whose codes SoundStorm models.</summary>
    public SoundStream<T> Codec => (SoundStream<T>?)_codec?.Codec ?? throw new InvalidOperationException("SoundStorm's codec exists in native mode only.");

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1 + _options.NumCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => Math.Min(_options.SemanticCodes, _options.CodebookSize);

    private bool HasPaperLayers => _acoustic is not null;

    /// <inheritdoc />
    /// <remarks>ByT5's text ids: the special tokens and the 256 bytes.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(ByteOffset + 256) : base.GetInputDomain(inputShape);

    private SoundStormCore<T> Acoustic =>
        _acoustic ?? throw new InvalidOperationException("SoundStorm's acoustic network exists in native mode only.");

    private T5Seq2Seq<T> TextToSemantic =>
        _textToSemantic ?? throw new InvalidOperationException("SoundStorm's text-to-semantic network exists in native mode only.");

    // The networks' layers, for tests that check a stage updates its own network only.
    internal IReadOnlyList<LayerBase<T>> TextToSemanticLayers => _textLayers;
    internal IReadOnlyList<LayerBase<T>> SemanticToAcousticLayers => _acousticLayers;
    internal SoundStormCore<T>? AcousticCore => _acoustic;
    internal T5Seq2Seq<T>? TextModel => _textToSemantic;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _textToSemantic = new T5Seq2Seq<T>(Engine, _textLayers, new T5Configuration(
            VocabularySize: o.VocabSize, ModelDim: o.TextModelDim, FeedForwardDim: o.TextFeedForwardDim,
            KeyValueDim: o.TextKeyValueDim, Heads: o.NumHeads, EncoderLayers: o.NumEncoderLayers,
            DecoderLayers: o.NumDecoderLayers, Dropout: o.TextDropoutRate, DecoderStartTokenId: PadToken,
            EndTokenId: EndToken, GatedGelu: true, TieWordEmbeddings: false));
        _textToSemantic.InitializeLikeHuggingFace(random);

        double Normal002()
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        // Codes plus the mask row; the heads predict codes only.
        _acoustic = new SoundStormCore<T>(Engine, _acousticLayers, new SoundStormCoreConfiguration(
            Levels: o.NumCodebooks, CodebookSize: o.CodebookSize, EmbeddingRows: o.CodebookSize + 1,
            HeadOutputs: o.CodebookSize, PaddingRow: -1, ConditioningCodes: o.SemanticCodes, ConditioningPaddingRow: -1,
            Conformer: new SoundStormConformerConfiguration(
                Dim: o.HiddenDim, Layers: o.ConformerLayers, Heads: o.NumHeads, HeadDim: o.ConformerHeadDim,
                FeedForwardMultiplier: o.FeedForwardMultiplier, ConvExpansion: o.ConvExpansion,
                ConvKernel: o.ConvKernel, Dropout: o.DropoutRate),
            GlobalConditioningDim: 0, GlobalConditioningDropout: 0.0,
            MaskInput: SoundStormMaskInput.MaskToken, Prompt: SoundStormPromptSampling.Paper,
            FinerPromptLevelsVisible: true, GreedyLastIteration: true), Normal002);

        // SoundStorm only encodes and decodes with its (pretrained, frozen) codec, so the codec's discriminators are
        // never built.
        var codecOptions = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.SoundStream);
        codecOptions.IncludeDiscriminators = false;
        var codec = new SoundStream<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (codec.CodebookSize != o.CodebookSize)
            throw new ArgumentException($"SoundStorm's {o.CodebookSize} codes per level must equal the codec's {codec.CodebookSize}.");
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"SoundStorm's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the SoundStream options.");
        if (codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps) != o.NumCodebooks)
            throw new ArgumentException($"SoundStorm models {o.NumCodebooks} levels; the codec's bandwidth gives " +
                $"{codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps)}.");
        if (CodecFrameRate % o.SemanticFrameRate != 0)
            throw new ArgumentException($"The codec's {CodecFrameRate} frames per second must be a multiple of the semantic tokens' {o.SemanticFrameRate}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_textLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_acousticLayers);
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- front end

    /// <inheritdoc />
    /// <remarks>The text's UTF-8 bytes as ByT5 ids (byte + 3), truncated to <see cref="MaxTextLength"/>, then
    /// <c>&lt;/s&gt;</c>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        if (bytes.Length == 0) throw new ArgumentException("The text is empty.", nameof(text));
        int length = Math.Min(bytes.Length, _options.MaxTextLength - 1);
        var ids = new Tensor<T>(new[] { length + 1 });
        for (int i = 0; i < length; i++) ids[i] = NumOps.FromDouble(bytes[i] + ByteOffset);
        ids[length] = NumOps.FromDouble(EndToken);
        return ids;
    }

    private int UpsampleFactor => CodecFrameRate / _options.SemanticFrameRate;

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, objective, StageOptimizer());
        if (CurrentStage == SoundStormTrainingStage.TextToSemantic) TextToSemanticUpdates++;
        else SemanticToAcousticUpdates++;
        return loss;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, sample.Tokens)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // The utterance's codes [1 + Q, frames]: the semantic token at the codec's frame rate, then the acoustic levels.
    private int[,] CodesOf(TtsTrainingSample<T> sample)
    {
        int codebooks = CodecTokenCodebooks;
        var tokens = sample.CodecTokens ?? throw new ArgumentException(
            "SoundStorm trains on codec tokens [frames, 1 + Q]: the semantic token at the codec's frame rate, then the Q codes.", nameof(sample));
        if (tokens.Rank != 2 || (tokens.Shape[1] != codebooks && tokens.Shape[1] != 1))
            throw new ArgumentException($"Expected codec tokens [frames, {codebooks}], or [frames, 1] of semantic tokens with the recording.", nameof(sample));
        int frames = tokens.Shape[0];
        var codes = new int[codebooks, frames];
        for (int f = 0; f < frames; f++)
            codes[0, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, 0]));
        if (tokens.Shape[1] == codebooks)
        {
            for (int f = 0; f < frames; f++)
                for (int q = 1; q < codebooks; q++) codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
            return codes;
        }
        var audio = sample.Audio ?? throw new ArgumentException("Semantic tokens alone need the recording for the acoustic codes.", nameof(sample));
        var acoustic = Codec.Encode(audio);
        int levels = Math.Min(acoustic.GetLength(0), _options.NumCodebooks), common = Math.Min(frames, acoustic.GetLength(1));
        if (levels != _options.NumCodebooks)
            throw new ArgumentException($"The codec gave {acoustic.GetLength(0)} levels; SoundStorm models {_options.NumCodebooks}.", nameof(sample));
        var aligned = new int[codebooks, common];
        for (int f = 0; f < common; f++)
        {
            aligned[0, f] = codes[0, f];
            for (int q = 0; q < levels; q++) aligned[q + 1, f] = acoustic[q, f];
        }
        return aligned;
    }

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var codes = CodesOf(sample);
        int frames = codes.GetLength(1);

        if (CurrentStage == SoundStormTrainingStage.TextToSemantic)
        {
            // Non-deduplicated semantic tokens at their own rate (every UpsampleFactor-th codec frame), then </s>.
            int factor = UpsampleFactor, count = (frames + factor - 1) / factor;
            var labels = new int[count + 1];
            for (int i = 0; i < count; i++) labels[i] = ByteVocabulary + codes[0, i * factor];
            labels[count] = EndToken;
            var text = sample.Tokens;
            return (_, _) => TextToSemantic.Loss(text, labels, training, random, freezeEncoder: true);
        }

        var o = _options;
        var acoustic = new int[o.NumCodebooks, frames];
        for (int l = 0; l < o.NumCodebooks; l++)
            for (int f = 0; f < frames; f++) acoustic[l, f] = codes[l + 1, f];
        var semantic = new int[frames];
        for (int f = 0; f < frames; f++) semantic[f] = codes[0, f];
        var objective = Acoustic.TrainingObjective(acoustic, semantic, null, training, random);
        return (_, _) => objective();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            var o = _options;
            double peak = o.LearningRate, start = o.WarmupStartLearningRate;
            int warmup = o.WarmupSteps, total = o.TrainingSteps;
            // lucidrains' SoundStormTrainer: a linear warmup from start to peak, then CosineAnnealingLR(T_max = total
            // steps), which it only begins stepping after the warmup.
            var schedule = new LambdaLRScheduler(peak, step =>
            {
                if (step < warmup) return (start + (peak - start) * step / warmup) / peak;
                if (total <= 0) return 1.0;
                return 0.5 * (1 + Math.Cos(Math.PI * Math.Min(step - warmup, total) / total));
            });
            var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = peak,
                Beta1 = 0.9,
                Beta2 = 0.99,
                Epsilon = 1e-8,
                EnableGradientClipping = true,
                MaxGradientNorm = o.MaxGradientNorm,
                LearningRateScheduler = schedule,
            };
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options),
                CurrentStage == SoundStormTrainingStage.TextToSemantic ? "text-to-semantic" : "semantic-to-acoustic");
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var stageLayers = CurrentStage == SoundStormTrainingStage.TextToSemantic ? _textLayers : _acousticLayers;
        var stage = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(stageLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(stage.Contains).ToList();
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is the text as ByT5 ids (<see cref="PreprocessText"/>); the output is the waveform
    /// <c>[samples]</c>. With a <see cref="TtsModelBase{T}.Voice"/>, its recording and semantic tokens are the prompt.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            int factor = UpsampleFactor;

            int[] promptSemantic = Array.Empty<int>();
            int[,] promptCodes = new int[o.NumCodebooks, 0];
            if (Voice is { } voice)
            {
                var recording = voice.ReferenceAudio ?? throw new ArgumentException("A SoundStorm voice needs its recording.");
                var tokens = voice.ReferenceTokens ?? throw new ArgumentException("A SoundStorm voice needs its semantic tokens at 25 Hz.");
                promptSemantic = Enumerable.Range(0, tokens.Length).Select(i => (int)Math.Round(NumOps.ToDouble(tokens[i]))).ToArray();
                if (promptSemantic.Any(code => code < 0 || code >= o.SemanticCodes))
                    throw new ArgumentOutOfRangeException(nameof(input), $"Semantic tokens lie in [0, {o.SemanticCodes}).");
                promptCodes = Codec.Encode(recording);
            }

            // Text to semantic: continue the prompt's semantic tokens.
            var prefix = promptSemantic.Select(code => ByteVocabulary + code).ToList();
            // The decoder shares one vocabulary of bytes and semantic codes but emits only semantic codes (and the end
            // token), and at least one: sampling a byte or ending at once left nothing to vocode.
            var generated = TextToSemantic.Generate(input, prefix, o.Temperature, o.TopK, o.MaxNewSemanticTokens, random,
                    allowed: id => id >= ByteVocabulary || id == EndToken, minNewTokens: 1)
                .Skip(prefix.Count).Where(id => id >= ByteVocabulary).Select(id => id - ByteVocabulary).ToArray();
            if (generated.Length == 0)
                throw new InvalidOperationException("The text-to-semantic model generated no semantic tokens.");

            // The prompt's frames are those both its codes and its upsampled semantic tokens cover.
            int promptFrames = Math.Min(promptCodes.GetLength(1), promptSemantic.Length * factor);
            int frames = promptFrames + generated.Length * factor;
            var semantic = new int[frames];
            for (int f = 0; f < promptFrames; f++) semantic[f] = promptSemantic[f / factor];
            for (int f = promptFrames; f < frames; f++) semantic[f] = generated[(f - promptFrames) / factor];
            var codes = new int[o.NumCodebooks, frames];
            for (int l = 0; l < o.NumCodebooks; l++)
                for (int f = 0; f < frames; f++) codes[l, f] = f < promptFrames ? promptCodes[l, f] : Acoustic.MaskId;

            Acoustic.Decode(codes, semantic, promptFrames, level => level == 0 ? o.FirstLevelIterations : 1, null, random);

            var continuation = new int[o.NumCodebooks, frames - promptFrames];
            for (int l = 0; l < o.NumCodebooks; l++)
                for (int f = promptFrames; f < frames; f++) continuation[l, f - promptFrames] = codes[l, f];
            var audio = Codec.Decode(continuation);
            return audio.Reshape(new[] { audio.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <summary>The codec's codes <c>[frames, Q]</c> of a recording.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        var codes = Codec.Encode(audio);
        int levels = codes.GetLength(0), frames = codes.GetLength(1);
        var tokens = new Tensor<T>(new[] { frames, levels });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < levels; q++) tokens[f, q] = NumOps.FromDouble(codes[q, f]);
        return tokens;
    }

    /// <summary>The waveform of codec codes <c>[frames, Q]</c>.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2)
            throw new ArgumentException("Expected codec tokens [frames, levels].", nameof(tokens));
        int frames = tokens.Shape[0], levels = tokens.Shape[1];
        var codes = new int[levels, frames];
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < levels; q++) codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
        var audio = Codec.Decode(codes);
        return audio.Reshape(new[] { audio.Length });
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>Loads the text-pretrained ByT5 encoder (§5) from a Hugging Face <c>T5ForConditionalGeneration</c>
    /// state dict (safetensors or <c>.pt</c>), such as google/byt5-large: its 384 byte embeddings and the encoder.
    /// The decoder, which the paper trains, keeps its values.</summary>
    public void LoadTextEncoderWeights(string path) =>
        TextToSemantic.LoadHuggingFaceEncoderWeights(Reader(path, prefix: ""), ByteVocabulary);

    /// <summary>Loads the whole text-to-semantic model, as this class saves it in Hugging Face's layout.</summary>
    public void LoadTextToSemanticWeights(string path) =>
        TextToSemantic.LoadHuggingFaceWeights(Reader(path, prefix: ""));

    /// <summary>Loads the acoustic model from a state dict laid out as Pheme's <c>TTSConformer</c>
    /// (<c>embedding.{l}</c>, <c>semantic_embedding</c>, <c>conformer</c>, <c>heads.{l}</c>).</summary>
    public void LoadSemanticToAcousticWeights(string path) =>
        Acoustic.LoadTorchWeights(Reader(path, prefix: "model."), conditioningName: "semantic_embedding", globalName: "spkr_linear");

    private static Func<string, int[], double[]> Reader(string path, string prefix)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return (name, shape) =>
        {
            if (!file.TryGetValue(prefix + name, out var tensor) && !file.TryGetValue(name, out tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{prefix + name}'.");
            var actual = tensor.Shape.ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        };
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SoundStorm-Native" : "SoundStorm-ONNX",
            Description = "SoundStorm: Efficient Parallel Audio Generation (Borsos et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.ConformerLayers + _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "SoundStorm";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SoundStorm<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing) (_codec?.Codec as IDisposable)?.Dispose();
        base.Dispose(disposing);
    }
}
