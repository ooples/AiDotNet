using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Resampling;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// MaskGCT (Wang et al. 2024): zero-shot text-to-speech by two masked generative transformers — text and a prompt's
/// semantic tokens to the target's semantic tokens, then semantic tokens and the prompt's acoustic tokens to the target's
/// acoustic tokens, layer by layer — with no text-speech alignment and no phone durations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper fixes the design (§3.2): semantic tokens from w2v-BERT 2.0's 17th hidden layer quantized by a VQ-VAE of
/// 8,192 codes; acoustic tokens from a 24 kHz codec of twelve residual layers; a text-to-semantic transformer reading the
/// phones and the prompt's semantic tokens as a prefix of the masked target, and a SoundStorm-like semantic-to-acoustic
/// transformer; both Llama-style, bidirectional, with adaptive RMSNorm on the step. The release (Amphion
/// <c>models/tts/maskgct</c>, <c>amphion/MaskGCT</c>) fills the rest: its G2P front ends, its decoding, the rule that sets the
/// target's length from the prompt's speaking rate, and the pair of semantic-to-acoustic models it ships — one for the
/// first acoustic layer and one for the other eleven given it. The paper's flow-matching duration predictor (App. A.5)
/// is an option.</para>
/// <para>Synthesis needs a voice: a recording of the speaker and its transcript (<see cref="CreateVoice"/>). The text is
/// phonemized in <see cref="MaskGCTOptions.Language"/> — eSpeak NG (installed separately) for English, French, German and
/// Korean, Open JTalk with its dictionary for Japanese, MaskGCT's Mandarin front end with the release's polyphone model for
/// Chinese.</para>
/// <para><b>For Beginners:</b> Give MaskGCT a few seconds of someone speaking and what they said, and it speaks new text
/// in that voice: it first sketches the speech's content as coarse tokens, then fills in the sound in layers of finer
/// tokens, refining many tokens at once in each pass.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new MaskGCT&lt;float&gt;(architecture, new MaskGCTOptions { Language = "en" });
/// model.LoadReferenceWeights("MaskGCT", "w2v-bert-2.0/model.safetensors", "wav2vec2bert_stats.pt");
/// model.Voice = model.CreateVoice(promptAudio, promptSampleRate, "The prompt's transcript.", "en");
/// var audio = model.Synthesize("Text to speak in that voice.");   // 24 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer",
    "https://arxiv.org/abs/2409.00750",
    Year = 2024,
    Authors = "Wang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WarmupSteps = 32000,
                Schedule = LearningRateSchedulerType.Noam,
                Source = "Wang et al. 2024, Sec. 4: the AdamW optimizer with a learning rate of 1e-4 "
                        + "and 32K warmup steps, following the inverse square root learning schedule, "
                        + "declared here as Noam.")]
public partial class MaskGCT<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly MaskGCTOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private Wav2Vec2BertFeatures<T>? _features;
    private Wav2Vec2BertNetwork<T>? _semanticModel;
    private MaskGctSemanticCodec<T>? _semanticCodec;
    private MaskGctAcousticCodec<T>? _acousticCodec;
    private MaskGctTextToSemantic<T>? _t2s;
    private MaskGctSemanticToAcoustic<T>? _s2aFirst;
    private MaskGctSemanticToAcoustic<T>? _s2aFull;
    private MaskGctDurationPredictor<T>? _duration;

    private readonly List<LayerBase<T>> _semanticModelLayers = new();
    private readonly List<LayerBase<T>> _semanticCodecLayers = new();
    private readonly List<LayerBase<T>> _acousticCodecLayers = new();
    private readonly List<LayerBase<T>> _t2sLayers = new();
    private readonly List<LayerBase<T>> _s2aFirstLayers = new();
    private readonly List<LayerBase<T>> _s2aFullLayers = new();
    private readonly List<LayerBase<T>> _durationLayers = new();

    private AmphionPhonemizer<T>? _phonemizer;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed MaskGCT for inference.</summary>
    public MaskGCT(NeuralNetworkArchitecture<T> architecture, string modelPath, MaskGCTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MaskGCTOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable MaskGCT.</summary>
    public MaskGCT(NeuralNetworkArchitecture<T> architecture, MaskGCTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MaskGCTOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.CodecEncoderRates.Aggregate(1, (a, b) => a * b);
        base.HiddenDim = o.HiddenDim;
        if (o.HopSize != base.HopSize) throw new ArgumentException($"HopSize {o.HopSize} differs from the codec's strides' product {base.HopSize}.");
        if (o.SemanticToAcousticSteps is null || o.SemanticToAcousticSteps.Length != o.NumQuantizers || o.SemanticToAcousticSteps.Any(s => s < 1))
            throw new ArgumentException($"SemanticToAcousticSteps needs a positive step count for each of the {o.NumQuantizers} layers.");
        if (o.TextToSemanticSteps < 1) throw new ArgumentException("TextToSemanticSteps must be positive.");
        // PaperOptimizerFactory ignores a non-positive override and would silently train on the paper warmup instead.
        if (o.WarmupSteps <= 0) throw new ArgumentOutOfRangeException("options", o.WarmupSteps, "WarmupSteps must be positive.");
        if (o.DurationMultiplier <= 0) throw new ArgumentException("DurationMultiplier must be positive.");
        if (o.TargetDurationSeconds is <= 0) throw new ArgumentException("TargetDurationSeconds must be positive.");
        if (o.PromptDropProbability is < 0 or > 1) throw new ArgumentException("PromptDropProbability lies in [0, 1].");
        // The release clips no gradients.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumQuantizers;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <summary>Acoustic frames per second (50 at 24 kHz).</summary>
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)HopSize);

    /// <inheritdoc />
    /// <remarks>Synthesis continues a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _t2s is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal Wav2Vec2BertNetwork<T> SemanticModel => _semanticModel ?? throw NotNative();
    internal MaskGctSemanticCodec<T> SemanticCodec => _semanticCodec ?? throw NotNative();
    internal MaskGctAcousticCodec<T> AcousticCodec => _acousticCodec ?? throw NotNative();
    internal MaskGctTextToSemantic<T> TextToSemantic => _t2s ?? throw NotNative();
    internal MaskGctSemanticToAcoustic<T> FirstLayerModel => _s2aFirst ?? throw NotNative();
    internal MaskGctSemanticToAcoustic<T> FullModel => _s2aFull ?? throw NotNative();
    internal MaskGctDurationPredictor<T> DurationPredictor => _duration ?? throw NotNative();

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.TrainingSeed);
        var semantic = new Wav2Vec2BertConfiguration
        {
            HiddenSize = o.SemanticHiddenSize, Heads = o.SemanticHeads, IntermediateSize = o.SemanticIntermediateSize,
            Layers = Math.Max(o.SemanticLayer, 1),
        };
        _features = new Wav2Vec2BertFeatures<T>(Engine, semantic);
        _semanticModel = new Wav2Vec2BertNetwork<T>(Engine, _semanticModelLayers, semantic, o.SemanticLayer);
        _semanticCodec = new MaskGctSemanticCodec<T>(Engine, _semanticCodecLayers, o.SemanticHiddenSize, o.SemanticCodecDim, o.SemanticCodecIntermediate,
            o.SemanticCodecLayers, o.SemanticCodebookSize, o.SemanticCodebookDim, random);
        _acousticCodec = new MaskGctAcousticCodec<T>(Engine, _acousticCodecLayers, o.CodecEncoderDim, (int[])o.CodecEncoderRates.Clone(), o.CodecLatentDim,
            o.NumQuantizers, o.CodebookSize, o.CodebookDim, o.VocosDim, o.VocosIntermediate, o.VocosLayers, o.VocosFftSize, base.HopSize, random);
        var t2s = new MaskGctTransformerConfiguration { Hidden = o.TextToSemanticDim, Layers = o.TextToSemanticLayers, Heads = o.TextToSemanticHeads, Activation = o.Activation };
        _t2s = new MaskGctTextToSemantic<T>(Engine, _t2sLayers, t2s, o.SemanticCodebookSize, o.VocabSize, random);
        var s2a = new MaskGctTransformerConfiguration { Hidden = o.HiddenDim, Layers = o.SemanticToAcousticLayers, Heads = o.SemanticToAcousticHeads, Activation = o.Activation };
        _s2aFirst = new MaskGctSemanticToAcoustic<T>(Engine, _s2aFirstLayers, s2a, 1, o.CodebookSize, o.SemanticCodebookSize, true, random);
        _s2aFull = new MaskGctSemanticToAcoustic<T>(Engine, _s2aFullLayers, s2a, o.NumQuantizers, o.CodebookSize, o.SemanticCodebookSize, false, random);
        var duration = new MaskGctTransformerConfiguration { Hidden = o.DurationDim, Layers = o.DurationLayers, Heads = o.DurationHeads, Activation = o.Activation };
        _duration = new MaskGctDurationPredictor<T>(Engine, _durationLayers, duration, o.VocabSize, random);
        AddUnsplitLayers(_t2sLayers.Concat(_s2aFirstLayers).Concat(_s2aFullLayers).Concat(_durationLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_semanticModelLayers);
        ComponentLayers.AddRange(_semanticCodecLayers);
        ComponentLayers.AddRange(_acousticCodecLayers);
    }

    // ---------------------------------------------------------------- the front ends

    /// <summary>
    /// MaskGCT's <c>g2p_</c>: the phone string and the phone ids of <paramref name="text"/> in <paramref name="language"/>.
    /// </summary>
    public (string Phonemes, int[] Ids) Phonemize(string text, string language) =>
        (_phonemizer ??= new AmphionPhonemizer<T>(Engine, () => _options.PolyphoneModelPath, () => _options.OpenJTalkDictionaryPath, nameof(MaskGCTOptions)))
        .Phonemize(text, language);

    // ---------------------------------------------------------------- tokens

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    private static Tensor<T> Ids(IReadOnlyList<int> ids) => new(new[] { ids.Count }, new Vector<T>(ids.Select(i => Ops.FromDouble(i)).ToArray()));

    private int[] PhoneIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"Phone ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <summary>
    /// The semantic tokens of a 16 kHz waveform (<c>extract_features</c> and <c>extract_semantic_code</c>): w2v-BERT 2.0's
    /// features padded to whole rows, its hidden state <see cref="MaskGCTOptions.SemanticLayer"/> with the padding masked,
    /// normalized by the feature statistics and quantized by the semantic codec.
    /// </summary>
    internal int[] SemanticTokens(IReadOnlyList<double> audio16k)
    {
        if (!HasPaperLayers) throw NotNative();
        using var _ = new NoGradScope<T>();
        foreach (var layer in _semanticModelLayers.Concat(_semanticCodecLayers)) layer.SetTrainingMode(false);
        var (features, mask) = _features!.ForwardPadded(audio16k);
        var hidden = SemanticModel.HiddenStates(features, mask)[_options.SemanticLayer];
        return SemanticCodec.Encode(hidden);
    }

    /// <inheritdoc />
    /// <remarks>The acoustic codes of a 24 kHz recording (<c>extract_acoustic_code</c>): <c>[frames, 12]</c>, each in
    /// [0, 1024).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var codes = AcousticCodec.Encode(Doubles(audio));
        var tokens = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int q = 0; q < codes.GetLength(1); q++) tokens[f, q] = Ops.FromDouble(codes[f, q]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The 24 kHz audio of acoustic codes <c>[frames, 12]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        return Wave(AcousticCodec.Decode(CodesOf(tokens)));
    }

    private int[,] CodesOf(Tensor<T> tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2 || tokens.Shape[1] != _options.NumQuantizers)
            throw new ArgumentException($"MaskGCT's acoustic codes are [frames, {_options.NumQuantizers}].", nameof(tokens));
        var codes = new int[tokens.Shape[0], tokens.Shape[1]];
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int q = 0; q < codes.GetLength(1); q++)
            {
                codes[f, q] = (int)Math.Round(Ops.ToDouble(tokens[f, q]));
                if (codes[f, q] < 0 || codes[f, q] >= _options.CodebookSize)
                    throw new ArgumentException($"Acoustic codes lie in [0, {_options.CodebookSize}).", nameof(tokens));
            }
        return codes;
    }

    // ---------------------------------------------------------------- the voice

    // What a voice made by CreateVoice keeps beyond TtsVoice's fields: the prompt's length at 16 kHz, which the
    // duration rule reads (the release resamples the file to each rate separately).
    private sealed class PromptInfo
    {
        public int Samples16k;
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();

    /// <summary>
    /// A voice for synthesis from a recording of the speaker at <paramref name="sampleRate"/> and its transcript in
    /// <paramref name="language"/>: the recording resampled to 24 kHz (its acoustic tokens are the prompt) and to 16 kHz
    /// (its semantic tokens are the prompt, its length sets the speaking rate), as the release loads the prompt file at
    /// each rate, and the transcript's phone ids.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string transcript, string language)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (transcript is null) throw new ArgumentNullException(nameof(transcript));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        var samples = Doubles(recording);
        var audio24k = sampleRate == _options.SampleRate ? samples : SoxrResampler.LibrosaResample(samples, sampleRate, _options.SampleRate);
        var audio16k = sampleRate == _options.SemanticSampleRate ? samples : SoxrResampler.LibrosaResample(samples, sampleRate, _options.SemanticSampleRate);
        var voice = new TtsVoice<T>
        {
            ReferenceAudio = Wave(audio24k),
            ReferenceTokens = Ids(Phonemize(transcript, language).Ids),
            ReferenceCodecTokens = HasPaperLayers ? Ids(SemanticTokens(audio16k)) : null,
        };
        _prompts.Add(voice, new PromptInfo { Samples16k = audio16k.Length });
        return voice;
    }

    // The prompt's 16 kHz waveform length and semantic tokens: from CreateVoice, or from the 24 kHz recording.
    private (int Samples16k, int[] Semantic) Prompt(TtsVoice<T> voice)
    {
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} continues a recording; set ReferenceAudio on the voice.");
        int[]? semantic = voice.ReferenceCodecTokens is { } codes ? codes.ToArray().Select(v => (int)Math.Round(Ops.ToDouble(v))).ToArray() : null;
        if (_prompts.TryGetValue(voice, out var info) && semantic is not null) return (info.Samples16k, semantic);
        var audio16k = SoxrResampler.LibrosaResample(Doubles(audio), _options.SampleRate, _options.SemanticSampleRate);
        return (audio16k.Length, semantic ?? SemanticTokens(audio16k));
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's phone ids (<see cref="Phonemize"/> in <see cref="MaskGCTOptions.Language"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Phonemize(text, _options.Language).Ids;
        if (ids.Length == 0) throw new ArgumentException("The text has no phones.", nameof(text));
        return Ids(ids);
    }

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the target text's phone ids; the 24 kHz speech the release's pipeline makes of
    /// them in the voice's (<c>maskgct_inference</c>, returning the target's audio).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var target = PhoneIds(input, nameof(input));
        if (target.Length == 0) throw new ArgumentException("There are no phones.", nameof(input));
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var promptPhones = voice.ReferenceTokens is { } tokens ? PhoneIds(tokens, nameof(voice)) : Array.Empty<int>();
            var (samples16k, promptSemantic) = Prompt(voice);
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            Func<int, double[]> uniforms = count =>
            {
                var u = new double[count];
                for (int i = 0; i < count; i++) u[i] = random.NextDouble();
                return u;
            };
            int length = TargetLength(promptPhones, target, samples16k, random);
            var semantic = Generate(promptPhones, target, promptSemantic, length, uniforms);
            var promptCodes = AcousticCodec.Encode(Doubles(voice.ReferenceAudio!));
            var codes = Acoustic(semantic, promptCodes, uniforms);
            return Wave(AcousticCodec.Decode(codes));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The target's length in semantic frames: <see cref="MaskGCTOptions.TargetDurationSeconds"/>, the release's
    /// rule, or the duration predictor, times <see cref="MaskGCTOptions.DurationMultiplier"/>.</summary>
    internal int TargetLength(IReadOnlyList<int> promptPhones, IReadOnlyList<int> targetPhones, int samples16k, Random random)
    {
        var o = _options;
        const int framesPerSecond = 50;
        double frames;
        if (o.TargetDurationSeconds is double seconds)
            frames = seconds * framesPerSecond;
        else if (o.DurationMode == MaskGctDurationMode.Rule)
        {
            if (promptPhones.Count == 0)
                throw new InvalidOperationException("The duration rule reads the prompt's speaking rate, which needs its transcript (the voice's "
                    + "ReferenceTokens, from CreateVoice); or set TargetDurationSeconds or the duration predictor.");
            frames = samples16k * (double)targetPhones.Count / promptPhones.Count / o.SemanticSampleRate * framesPerSecond;
        }
        else
        {
            var noise = Enumerable.Range(0, targetPhones.Count).Select(_ => Normal(random)).ToArray();
            frames = DurationPredictor.Generate(Array.Empty<int>(), Array.Empty<int>(), targetPhones, o.DurationSteps, o.DurationGuidance, noise).Sum();
        }
        return Math.Max(1, (int)(frames * o.DurationMultiplier));
    }

    internal static double Normal(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());

    private MaskGctDecoding Decoding(double temperature) => new()
    {
        Temperature = temperature, FilterThreshold = _options.FilterThreshold, Guidance = _options.Guidance, GuidanceRescale = _options.GuidanceRescale,
    };

    /// <summary><c>text2semantic</c>: the prompt's semantic tokens followed by the target's, decoded from the phones of the
    /// prompt's transcript and the text.</summary>
    internal int[] Generate(IReadOnlyList<int> promptPhones, IReadOnlyList<int> targetPhones, IReadOnlyList<int> promptSemantic, int length,
        Func<int, double[]> uniforms)
    {
        var phones = promptPhones.Concat(targetPhones).ToArray();
        var predicted = TextToSemantic.Generate(phones, promptSemantic, length,
            Decoding(_options.TextToSemanticTemperature) with { Steps = _options.TextToSemanticSteps }, uniforms);
        return promptSemantic.Concat(predicted).ToArray();
    }

    /// <summary><c>semantic2acoustic</c>: the target's acoustic codes <c>[frames, 12]</c> — the first layer by the
    /// one-layer model, the rest by the full model given it.</summary>
    internal int[,] Acoustic(IReadOnlyList<int> semantic, int[,] promptCodes, Func<int, double[]> uniforms)
    {
        var steps = _options.SemanticToAcousticSteps;
        var decoding = Decoding(_options.SemanticToAcousticTemperature);
        var first = FirstLayerModel.Generate(semantic, promptCodes, null, new[] { steps[0] }, decoding, uniforms);
        return FullModel.Generate(semantic, promptCodes, first, steps, decoding, uniforms);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    private static Dictionary<string, double[]> ReadState(string path)
    {
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        return state;
    }

    private static Func<string, int[], double[]?> Reader(Dictionary<string, double[]> state, HashSet<string> used) => (name, _) =>
    {
        if (!state.TryGetValue(name, out var value)) return null;
        used.Add(name);
        return value;
    };

    private static void RequireAllUsed(string what, Dictionary<string, double[]> state, HashSet<string> used, Func<string, bool>? unused = null)
    {
        var left = state.Keys.Where(k => !used.Contains(k) && (unused is null || !unused(k))).ToList();
        if (left.Count > 0) throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
    }

    /// <summary>
    /// Loads the release: an <c>amphion/MaskGCT</c> directory (<c>semantic_codec/model.safetensors</c>,
    /// <c>acoustic_codec/model.safetensors</c> and <c>model_1.safetensors</c>, <c>t2s_model/model.safetensors</c>,
    /// <c>s2a_model/s2a_model_1layer/model.safetensors</c>, <c>s2a_model/s2a_model_full/model.safetensors</c>), w2v-BERT 2.0's
    /// <c>model.safetensors</c> (<c>facebook/w2v-bert-2.0</c>) and the feature statistics (Amphion's
    /// <c>models/tts/maskgct/ckpt/wav2vec2bert_stats.pt</c>).
    /// </summary>
    public void LoadReferenceWeights(string directory, string semanticModelPath, string statisticsPath)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        LoadSemanticModel(semanticModelPath);
        LoadStatistics(statisticsPath);
        var semanticCodec = ReadState(Path.Combine(directory, "semantic_codec", "model.safetensors"));
        var used = new HashSet<string>(StringComparer.Ordinal);
        var read = Reader(semanticCodec, used);
        SemanticCodec.Load((name, _) => read(name, Array.Empty<int>()));
        // RepCodec's decoder reconstructs the features; MaskGCT only quantizes.
        RequireAllUsed("semantic codec", semanticCodec, used, k => k.StartsWith("decoder.", StringComparison.Ordinal));

        var encoder = ReadState(Path.Combine(directory, "acoustic_codec", "model.safetensors"));
        var decoder = ReadState(Path.Combine(directory, "acoustic_codec", "model_1.safetensors"));
        var usedEncoder = new HashSet<string>(StringComparer.Ordinal);
        var usedDecoder = new HashSet<string>(StringComparer.Ordinal);
        var readEncoder = Reader(encoder, usedEncoder);
        var readDecoder = Reader(decoder, usedDecoder);
        AcousticCodec.Load((name, _) => readEncoder(name, Array.Empty<int>()), (name, _) => readDecoder(name, Array.Empty<int>()));
        RequireAllUsed("acoustic encoder", encoder, usedEncoder);
        RequireAllUsed("acoustic decoder", decoder, usedDecoder);

        void Generator(string relative, Action<Func<string, int[], double[]?>> load, Func<string, bool>? unused = null)
        {
            var state = ReadState(Path.Combine(directory, relative));
            var usedNames = new HashSet<string>(StringComparer.Ordinal);
            load(Reader(state, usedNames));
            RequireAllUsed(relative, state, usedNames, unused);
        }
        // The acoustic models' DiffLlama is a LlamaModel whose token table its forward never reads (it is given
        // embeddings); the release saves it anyway.
        static bool TokenTable(string name) => name == "diff_estimator.embed_tokens.weight";
        Generator(Path.Combine("t2s_model", "model.safetensors"), TextToSemantic.Load);
        Generator(Path.Combine("s2a_model", "s2a_model_1layer", "model.safetensors"), FirstLayerModel.Load, TokenTable);
        Generator(Path.Combine("s2a_model", "s2a_model_full", "model.safetensors"), FullModel.Load, TokenTable);
    }

    /// <summary>Loads w2v-BERT 2.0 (<c>facebook/w2v-bert-2.0</c>'s <c>model.safetensors</c>); the layers past
    /// <see cref="MaskGCTOptions.SemanticLayer"/> are never run and not read.</summary>
    public void LoadSemanticModel(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = ReadState(path);
        SemanticModel.LoadTorchWeights((name, _) => state.TryGetValue(name, out var v) ? v : throw new InvalidDataException($"The w2v-BERT checkpoint has no tensor '{name}'."));
    }

    /// <summary>Loads the semantic feature statistics (<c>wav2vec2bert_stats.pt</c>: per-channel <c>mean</c> and
    /// <c>var</c>).</summary>
    public void LoadStatistics(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = ReadState(path);
        SemanticCodec.LoadStatistics(
            state.TryGetValue("mean", out var mean) ? mean : throw new InvalidDataException("The statistics have no 'mean'."),
            state.TryGetValue("var", out var variance) ? variance : throw new InvalidDataException("The statistics have no 'var'."));
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MaskGCT-Native" : "MaskGCT-ONNX",
            Description = "MaskGCT: zero-shot TTS by masked generative codec transformers (Wang et al., 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MaskGCT";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Language"] = _options.Language;
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MaskGCT<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
