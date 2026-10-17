using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// AudioLM: audio generation as language modeling over semantic and acoustic tokens (Borsos et al. 2022), in three
/// stages — semantic, coarse acoustic and fine acoustic.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Semantic tokens (w2v-BERT, 25 a second) carry the linguistic content and long-term structure; acoustic tokens
/// (SoundStream, 50 frames of 12 levels a second) carry the voice, the recording conditions and the fine detail. The
/// semantic stage continues the semantic tokens; the coarse stage predicts the first Q′ = 4 acoustic levels from all
/// the semantic tokens; the fine stage predicts the remaining 8 from the coarse ones, 3 seconds at a time (§III-C).
/// </para>
/// <para>
/// Generation (§III-D): <see cref="ContinueAudio"/> continues a prompt recording (its semantic tokens come from the
/// injected <see cref="W2vBert{T}"/>, or pass them to <see cref="ContinueCodes"/>); <see cref="GenerateAcoustic"/>
/// renders given semantic tokens ("acoustic generation"); <see cref="GenerateUnconditional"/> samples everything.
/// AudioLM has no text input, so the text-prompted <see cref="IAudioGenerator{T}"/> methods throw; SPEAR-TTS adds text.
/// </para>
/// <para>
/// Training: <see cref="Train(Tensor{T}, Tensor{T})"/> takes the semantic tokens at 25 Hz as the input and the
/// recording's codes <c>[frames, Q]</c> (or the recording) as the target, crops them as the paper does (30, 10 or
/// 3 seconds) and updates the network <see cref="CurrentStage"/> names. See <see cref="AudioLMOptions"/> for the sizes
/// and for what is taken from lucidrains/audiolm-pytorch where the paper is silent.
/// </para>
/// <para><b>For Beginners:</b> give AudioLM three seconds of someone speaking and it continues in the same voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var audioLm = new AudioLM&lt;float&gt;(architecture, new AudioLMOptions(), semanticTokenizer: w2vBert);
/// Tensor&lt;float&gt; continued = audioLm.ContinueAudio(threeSecondPrompt, extensionSeconds: 7);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AudioLM: a Language Modeling Approach to Audio Generation", "https://arxiv.org/abs/2209.03143",
    Year = 2022, Authors = "Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 0.5, Component = "semantic", Provenance = RecipeProvenance.NamedReproduction,
    Source = "The paper states no optimizer (Section IV-B: 16 TPUv4, batch 256, 1M steps) and released no code. "
             + "lucidrains/audiolm-pytorch's SemanticTransformerTrainer: lr 3e-4 constant, weight decay 0, gradient norm "
             + "clipped to 0.5; audiolm_pytorch's get_optimizer then returns Adam with betas (0.9, 0.99), eps 1e-8.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 0.5, Component = "coarse", Provenance = RecipeProvenance.NamedReproduction,
    Source = "As the semantic stage: lucidrains/audiolm-pytorch's CoarseTransformerTrainer has the same defaults.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 0.5, Component = "fine", Provenance = RecipeProvenance.NamedReproduction,
    Source = "As the semantic stage: lucidrains/audiolm-pytorch's FineTransformerTrainer has the same defaults.")]
public partial class AudioLM<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>, ITrainingObjectiveProvider<T>
{
    private readonly AudioLMOptions _options;
    private readonly W2vBert<T>? _semanticTokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<AudioLMStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private readonly List<LayerBase<T>> _semanticLayers = new();
    private readonly List<LayerBase<T>> _coarseLayers = new();
    private readonly List<LayerBase<T>> _fineLayers = new();
    private AudioLmCore<T>? _core;
    private AudioCodecLayer<T>? _codec;
    private bool _disposed;

    /// <summary>Creates AudioLM.</summary>
    /// <param name="architecture">The architecture (its layers are built here; custom layers are not supported).</param>
    /// <param name="options">The options; the paper's when null.</param>
    /// <param name="semanticTokenizer">A pre-trained w2v-BERT with a fitted semantic tokenizer, for prompts given as
    /// audio. Optional: without it, pass semantic tokens yourself.</param>
    /// <param name="optimizer">An optimizer for every stage instead of the per-stage defaults.</param>
    public AudioLM(NeuralNetworkArchitecture<T> architecture, AudioLMOptions? options = null, W2vBert<T>? semanticTokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioLMOptions();
        _semanticTokenizer = semanticTokenizer;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>The network a training step updates.</summary>
    public AudioLMStage CurrentStage { get; set; } = AudioLMStage.CoarseAcoustic;

    /// <summary>The SoundStream codec whose codes AudioLM models.</summary>
    public SoundStream<T> Codec => (SoundStream<T>?)_codec?.Codec ?? throw new InvalidOperationException("AudioLM's codec is not built.");

    /// <summary>Codec frames per second (50).</summary>
    public double FrameRate => (double)_options.SampleRate / Codec.HopLength;

    internal AudioLmCore<T> Core => _core ?? throw new InvalidOperationException("AudioLM's networks are not built.");

    internal IReadOnlyList<LayerBase<T>> StageLayers(AudioLMStage stage) => stage switch
    {
        AudioLMStage.Semantic => _semanticLayers,
        AudioLMStage.CoarseAcoustic => _coarseLayers,
        _ => _fineLayers,
    };

    private int FramesPerSemanticToken => (int)Math.Round(FrameRate / _options.SemanticFrameRate);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException("AudioLM builds its three Transformers; custom layers in the architecture are not supported.");
        var o = _options;
        var tokens = new AudioLmTokens(o.SemanticCodes, o.NumCodebooks, o.CoarseCodebooks, o.CodebookSize);
        T5Configuration Configuration(int vocabulary) => new(
            VocabularySize: vocabulary, ModelDim: o.ModelDim, FeedForwardDim: o.FeedForwardDim, KeyValueDim: o.KeyValueDim,
            Heads: o.NumHeads, EncoderLayers: 0, DecoderLayers: o.NumLayers, Dropout: o.DropoutRate,
            GatedGelu: true, TieWordEmbeddings: false);
        _core = new AudioLmCore<T>(Engine, _semanticLayers, _coarseLayers, _fineLayers, tokens, Configuration,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.Seed ?? 42));

        // AudioLM encodes and decodes with a pretrained, frozen SoundStream; its discriminators are never built.
        var codecOptions = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.SoundStream);
        codecOptions.IncludeDiscriminators = false;
        var codec = new SoundStream<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (codec.CodebookSize != o.CodebookSize || ((IAudioCodec<T>)codec).SampleRate != o.SampleRate)
            throw new ArgumentException($"AudioLM's {o.SampleRate} Hz audio and {o.CodebookSize} codes per level must be its codec's.");
        if (codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps) != o.NumCodebooks)
            throw new ArgumentException($"AudioLM models {o.NumCodebooks} levels; the codec's bandwidth gives {codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps)}.");
        _codec = new AudioCodecLayer<T>(codec);
        double rate = (double)o.SampleRate / codec.HopLength;
        if (Math.Abs(rate / o.SemanticFrameRate - Math.Round(rate / o.SemanticFrameRate)) > 1e-9)
            throw new ArgumentException($"The codec's {rate} frames per second must be a multiple of the semantic tokens' {o.SemanticFrameRate}.");

        Layers.AddRange(_semanticLayers);
        Layers.AddRange(_coarseLayers);
        Layers.AddRange(_fineLayers);
        // The codec stays out of the sequential layers, as Bark's does: a whole model cannot be rebuilt from a layer's
        // type name, so a serialized AudioLM restores it through its own construction while the base still finds the
        // codec's layers as an extra component of this model.
    }

    /// <inheritdoc />
    /// <remarks>The semantic token ids.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => LayerInputDomain.Indices(_options.SemanticCodes);

    // ---------------------------------------------------------------- tokens

    private int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = NumOps.ToDouble(tensor[i]);
            ids[i] = (int)Math.Round(value);
            if (ids[i] < 0 || ids[i] >= _options.SemanticCodes)
                throw new ArgumentOutOfRangeException(name, $"Semantic tokens lie in [0, {_options.SemanticCodes}).");
        }
        return ids;
    }

    // The codes [Q, frames] of a target given as codes [frames, Q] or as a recording.
    private int[,] CodesOf(Tensor<T> target)
    {
        Guard.NotNull(target);
        if (target.Rank == 2 && target.Shape[1] == _options.NumCodebooks)
        {
            int frames = target.Shape[0];
            var codes = new int[_options.NumCodebooks, frames];
            for (int f = 0; f < frames; f++)
                for (int q = 0; q < _options.NumCodebooks; q++)
                    codes[q, f] = Math.Max(0, Math.Min(_options.CodebookSize - 1, (int)Math.Round(NumOps.ToDouble(target[f, q]))));
            return codes;
        }
        var encoded = Codec.Encode(target.Rank == 1 ? target : target.Reshape(new[] { target.Length }));
        if (encoded.GetLength(0) < _options.NumCodebooks)
            throw new ArgumentException($"The codec gave {encoded.GetLength(0)} levels; AudioLM models {_options.NumCodebooks}.", nameof(target));
        return encoded;
    }

    private int[] SemanticTokensOf(Tensor<T> recording) =>
        (_semanticTokenizer ?? throw new InvalidOperationException(
            "A prompt given as audio needs the semantic tokenizer (a W2vBert with a fitted tokenizer); pass one to the " +
            "constructor, or pass the prompt's semantic tokens to ContinueCodes."))
        .SemanticTokens(recording);

    private static int[,] Slice(int[,] codes, int from, int to, int levels)
    {
        var result = new int[levels, Math.Max(0, to - from)];
        for (int q = 0; q < levels; q++)
            for (int t = from; t < to; t++) result[q, t - from] = codes[q, t];
        return result;
    }

    // ---------------------------------------------------------------- generation

    /// <summary>
    /// Continues a prompt given as codes <c>[Q, P]</c> and its semantic tokens (25 Hz) by
    /// <paramref name="extensionSeconds"/>; returns the prompt's and the continuation's codes <c>[Q, frames]</c>.
    /// </summary>
    public int[,] ContinueCodes(int[,] promptCodes, IReadOnlyList<int> promptSemantic, double extensionSeconds, int? seed = null)
    {
        Guard.NotNull(promptCodes);
        Guard.NotNull(promptSemantic);
        ThrowIfDisposed();
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed ?? o.Seed ?? 0);
        int frames = Math.Max(1, (int)Math.Round(extensionSeconds * FrameRate));
        int semanticTokens = Math.Max(1, (int)Math.Ceiling(extensionSeconds * o.SemanticFrameRate));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var core = Core;
            // Stage 1: continue the semantic tokens. Repetitions are removed, so the continuation is at most as many
            // tokens as the extension's duration holds.
            var semantic = promptSemantic.Concat(core.GenerateSemantic(promptSemantic, semanticTokens, o.SemanticTemperature, random)).ToArray();
            // Stage 2: the coarse levels for the extension, given all the semantic tokens and the prompt's coarse codes.
            var coarse = core.GenerateCoarse(semantic, promptCodes, frames, o.CoarseTemperature, random);
            int prompt = promptCodes.GetLength(1), total = prompt + coarse.GetLength(1);
            var codes = new int[o.NumCodebooks, total];
            for (int q = 0; q < o.NumCodebooks; q++)
                for (int t = 0; t < prompt; t++) codes[q, t] = promptCodes[q, t];
            for (int q = 0; q < o.CoarseCodebooks; q++)
                for (int t = 0; t < coarse.GetLength(1); t++) codes[q, prompt + t] = coarse[q, t];
            // Stage 3: the fine levels of the continuation, in non-overlapping chunks.
            int chunk = Math.Max(1, (int)Math.Round(o.FineCropSeconds * FrameRate));
            core.GenerateFine(codes, prompt, chunk, o.FineTemperature, random);
            return codes;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>Acoustic generation (§III-D): renders semantic tokens (25 Hz) to codes <c>[Q, frames]</c> with the
    /// coarse and fine stages.</summary>
    public int[,] GenerateAcoustic(IReadOnlyList<int> semantic, int? seed = null)
    {
        Guard.NotNull(semantic);
        ThrowIfDisposed();
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed ?? o.Seed ?? 0);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var core = Core;
            int frames = Math.Max(1, semantic.Count * FramesPerSemanticToken);
            var coarse = core.GenerateCoarse(semantic, new int[o.CoarseCodebooks, 0], frames, o.CoarseTemperature, random);
            var codes = new int[o.NumCodebooks, coarse.GetLength(1)];
            for (int q = 0; q < o.CoarseCodebooks; q++)
                for (int t = 0; t < coarse.GetLength(1); t++) codes[q, t] = coarse[q, t];
            core.GenerateFine(codes, 0, Math.Max(1, (int)Math.Round(o.FineCropSeconds * FrameRate)), o.FineTemperature, random);
            return codes;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>Unconditional generation (§III-D): samples semantic tokens from nothing, then renders them.</summary>
    public Tensor<T> GenerateUnconditional(double durationSeconds, int? seed = null) =>
        Decode(ContinueCodes(new int[_options.NumCodebooks, 0], Array.Empty<int>(), durationSeconds, seed));

    /// <summary>The waveform of codes <c>[Q, frames]</c>.</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        Guard.NotNull(codes);
        if (codes.GetLength(1) == 0) return new Tensor<T>(new[] { 0 });
        var audio = Codec.Decode(codes);
        return audio.Reshape(new[] { audio.Length });
    }

    /// <inheritdoc />
    /// <remarks>The prompt's semantic tokens come from the injected w2v-BERT; the result includes the prompt, which the
    /// paper also feeds to the decoder. AudioLM has no text input, so <paramref name="prompt"/> must be null.</remarks>
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        Guard.NotNull(inputAudio);
        if (prompt is not null)
            throw new NotSupportedException("AudioLM has no text input; SPEAR-TTS conditions it on text.");
        var recording = inputAudio.Rank == 1 ? inputAudio : inputAudio.Reshape(new[] { inputAudio.Length });
        return Decode(ContinueCodes(Codec.Encode(recording), SemanticTokensOf(recording), extensionSeconds, seed));
    }

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null) =>
        throw new NotSupportedException("AudioLM has no text input; use GenerateUnconditional, ContinueAudio or SPEAR-TTS.");

    /// <inheritdoc />
    public Task<Tensor<T>> GenerateAudioAsync(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> GenerateMusic(string prompt, string? negativePrompt = null, double durationSeconds = 10.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null) =>
        throw new NotSupportedException("AudioLM has no text input; continue a piano recording with ContinueAudio instead (§IV-I).");

    /// <inheritdoc />
    public Tensor<T> InpaintAudio(Tensor<T> audio, Tensor<T> mask, string? prompt = null, int numInferenceSteps = 100, int? seed = null) =>
        throw new NotSupportedException("AudioLM generates left to right; it does not inpaint.");

    /// <inheritdoc />
    public AudioGenerationOptions<T> GetDefaultOptions() => new() { DurationSeconds = 7.0, Seed = _options.Seed };

    /// <inheritdoc />
    public double MaxDurationSeconds => _options.SemanticCropSeconds;

    /// <inheritdoc />
    public bool SupportsTextToAudio => false;

    /// <inheritdoc />
    public bool SupportsTextToMusic => false;

    /// <inheritdoc />
    public bool SupportsAudioContinuation => _semanticTokenizer is not null;

    /// <inheritdoc />
    public bool SupportsAudioInpainting => false;

    /// <inheritdoc />
    /// <remarks>The input is semantic tokens at 25 Hz; the output is their acoustic generation as a waveform.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) =>
        Decode(GenerateAcoustic(ToIds(input, nameof(input)), _options.Seed ?? 0));

    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    // ---------------------------------------------------------------- training

    /// <summary>
    /// One training step of <see cref="CurrentStage"/>: <paramref name="input"/> is the semantic tokens at 25 Hz and
    /// <paramref name="expectedOutput"/> the recording's codes <c>[frames, Q]</c> or the recording. A random crop of the
    /// stage's length is taken (30, 10 or 3 seconds, §IV-B).
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(input, expectedOutput, _trainingRandom, training: true);
        var placeholder = new Tensor<T>(new[] { 1 });
        SetTrainingMode(true);
        try
        {
            TrainWithCustomObjective(placeholder, placeholder, (_, _) => objective(), StageOptimizer());
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    private Func<Tensor<T>> BuildObjective(Tensor<T> input, Tensor<T> expectedOutput, Random random, bool training)
    {
        Guard.NotNull(input);
        var semantic = ToIds(input, nameof(input));
        var core = Core;
        var o = _options;
        int ratio = FramesPerSemanticToken;
        if (CurrentStage == AudioLMStage.Semantic)
        {
            int tokens = Math.Max(1, (int)Math.Round(o.SemanticCropSeconds * o.SemanticFrameRate));
            int start = semantic.Length > tokens ? random.Next(semantic.Length - tokens + 1) : 0;
            return core.SemanticObjective(semantic.Skip(start).Take(tokens).ToArray(), training, random);
        }
        var codes = CodesOf(expectedOutput);
        // The semantic tokens and the codes cover the same audio, two codec frames per semantic token.
        int frames = Math.Min(codes.GetLength(1), semantic.Length * ratio);
        if (frames < 1) throw new ArgumentException("The sample has no frames.", nameof(expectedOutput));
        double seconds = CurrentStage == AudioLMStage.CoarseAcoustic ? o.CoarseCropSeconds : o.FineCropSeconds;
        int length = Math.Min(frames, Math.Max(ratio, (int)Math.Round(seconds * FrameRate) / ratio * ratio));
        int first = random.Next((frames - length) / ratio + 1) * ratio;
        var crop = Slice(codes, first, first + length, o.NumCodebooks);
        if (CurrentStage == AudioLMStage.FineAcoustic)
            return core.FineObjective(crop, training, random);
        var cropSemantic = semantic.Skip(first / ratio).Take((length + ratio - 1) / ratio).ToArray();
        return core.CoarseObjective(cropSemantic, crop, training, random);
    }

    /// <summary>The current stage's loss on a sample, in inference mode with a fixed crop; for evaluation.</summary>
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var objective = BuildObjective(input, expectedOutput,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 2), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => EvaluateTrainingObjective(input, target);
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> StageOptimizer()
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    Beta1 = 0.9,
                    Beta2 = 0.99,
                    Epsilon = 1e-8,
                    EnableGradientClipping = true,
                    MaxGradientNorm = _options.MaxGradientNorm,
                }), CurrentStage switch
                {
                    AudioLMStage.Semantic => "semantic",
                    AudioLMStage.CoarseAcoustic => "coarse",
                    _ => "fine",
                });
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var stage = new HashSet<Tensor<T>>(
            Training.TapeTrainingStep<T>.CollectParameters(StageLayers(CurrentStage).Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(stage.Contains).ToList();
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => true;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "AudioLM",
            Description = "AudioLM: a Language Modeling Approach to Audio Generation (Borsos et al., 2022)",
            FeatureCount = _options.ModelDim,
            Complexity = 3 * _options.NumLayers,
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioLM<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) (_codec?.Codec as IDisposable)?.Dispose();
        base.Dispose(disposing);
    }
}
