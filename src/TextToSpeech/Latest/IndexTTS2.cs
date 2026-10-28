using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Resampling;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// IndexTTS2 (Zhou et al. 2025): zero-shot autoregressive text-to-speech with precise duration control and emotion
/// control separated from the speaker's timbre — text to semantic tokens by a GPT-2 conditioned on a speaker prompt and a
/// style prompt, semantic tokens (with the GPT's latents) to a mel spectrogram by flow matching, and BigVGAN-v2 to audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The text-to-semantic model reads <c>[c + e, p, BT text ET, BA semantic]</c>: c, 32 latents a conformer and
/// perceiver pool from the speaker prompt's w2v-BERT 2.0 features; e, the emotion vector pooled the same way from a style
/// prompt (or mixed from an emotion table by weights, or by weights a text-to-emotion model reads off a sentence); p, the
/// duration token — by default the paper's <c>W_sem h(T)</c> naming the number of semantic tokens to generate, or zeros to
/// generate freely. The semantic-to-mel module conditions seed-vc's DiT on the semantic codec's features plus the GPT's
/// last hidden states (the paper's GPT latent enhancement), the prompt's mel and its CAM++ style.</para>
/// <para>The paper leaves most sizes to the release (<c>IndexTeam/IndexTTS-2</c>), whose code fills the gaps; where the
/// two differ the paper is the default and the release an option (<see cref="IndexTTS2Options.Release"/>), which the
/// released weights need.</para>
/// <para><b>For Beginners:</b> Give IndexTTS2 a few seconds of someone speaking and it reads new text in that voice. You
/// can also give it a second recording whose emotion to copy, a set of emotion weights, or a sentence describing the
/// emotion, and say how many tokens (roughly how long) the speech should be.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var options = IndexTTS2Options.Release();
/// options.BpeModelPath = "IndexTTS-2/bpe.model";
/// var model = new IndexTTS2&lt;float&gt;(architecture, options);
/// model.LoadReferenceWeights(new IndexTts2Checkpoints { Directory = "IndexTTS-2", SemanticModel = "w2v-bert-2.0/model.safetensors",
///     SemanticCodec = "MaskGCT/semantic_codec/model.safetensors", SpeakerEncoder = "campplus_cn_common.bin",
///     Vocoder = "bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt" });
/// model.Voice = model.CreateVoice(promptAudio, promptSampleRate);
/// var audio = model.Synthesize("Text to speak in that voice.");   // 22.05 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech",
    "https://arxiv.org/abs/2506.21619",
    Year = 2025,
    Authors = "Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4,
                Source = "Zhou et al. 2025, Training Hyperparameter Details: the AdamW optimizer with "
                        + "an initial learning rate of 2e-4 on 8 A100 GPUs.")]
public partial class IndexTTS2<T> : TtsModelBase<T>
{
    private readonly IndexTTS2Options _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private Wav2Vec2BertFeatures<T>? _features;
    private Wav2Vec2BertNetwork<T>? _semanticModel;
    private MaskGctSemanticCodec<T>? _semanticCodec;
    private IndexTtsUnifiedVoice<T>? _gpt;
    private IndexTtsSemanticToMel<T>? _s2mel;
    private CamPlusPlusEncoder<T>? _speakerEncoder;
    private BigVganGenerator<T>? _vocoder;
    private DenseLayer<T>? _speakerClassifier;
    private IndexTtsTextToEmotion<T>? _t2e;

    private readonly List<LayerBase<T>> _semanticModelLayers = new();
    private readonly List<LayerBase<T>> _semanticCodecLayers = new();
    private readonly List<LayerBase<T>> _gptLayers = new();
    private readonly List<LayerBase<T>> _s2melLayers = new();
    private readonly List<LayerBase<T>> _speakerEncoderLayers = new();
    private readonly List<LayerBase<T>> _classifierLayers = new();
    private readonly List<LayerBase<T>> _t2eLayers = new();

    private IndexTtsTokenizer? _tokenizer;
    private KaldiFbank<T>? _fbank;
    private DifferentiableMel<T>? _mel;
    private double[][]? _emotionRows;
    private double[][]? _speakerRows;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed IndexTTS2 for inference.</summary>
    public IndexTTS2(NeuralNetworkArchitecture<T> architecture, string modelPath, IndexTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new IndexTTS2Options();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable IndexTTS2.</summary>
    public IndexTTS2(NeuralNetworkArchitecture<T> architecture, IndexTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new IndexTTS2Options();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.VocoderUpsampleRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException($"The vocoder's upsampling rates must multiply to HopSize {o.HopSize}.");
        if ((o.VocoderInitialChannels >> o.VocoderUpsampleRates.Length) < 1)
            throw new ArgumentException("Each vocoder upsampling stage halves the channels: VocoderInitialChannels must keep at least one through them all.");
        if (o.WavenetHidden != o.DitHidden) throw new ArgumentException("The release's final layer reads the DiT's time embedding: WavenetHidden must equal DitHidden.");
        if (o.Beams < 1) throw new ArgumentException("Beams must be positive.");
        if (o.SegmentTokens < 1) throw new ArgumentException("SegmentTokens must be positive.");
        if (o.DurationDropProbability is < 0 or > 1) throw new ArgumentException("DurationDropProbability lies in [0, 1].");
        if (o.LatentFusionProbability is < 0 or > 1) throw new ArgumentException("LatentFusionProbability lies in [0, 1].");
        if (o.EmotionLabels is null || o.EmotionLabels.Length == 0) throw new ArgumentException("EmotionLabels needs at least one emotion.");
        if (o.EmotionGroupSizes is { } groups && groups.Length != o.EmotionLabels.Length)
            throw new ArgumentException("EmotionGroupSizes needs one group size per emotion label.");
        if (o.TargetSemanticTokens is <= 0) throw new ArgumentException("TargetSemanticTokens must be positive.");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _gpt is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal Wav2Vec2BertNetwork<T> SemanticModel => _semanticModel ?? throw NotNative();
    internal MaskGctSemanticCodec<T> SemanticCodec => _semanticCodec ?? throw NotNative();
    internal IndexTtsUnifiedVoice<T> TextToSemantic => _gpt ?? throw NotNative();
    internal IndexTtsSemanticToMel<T> SemanticToMel => _s2mel ?? throw NotNative();
    internal CamPlusPlusEncoder<T> SpeakerEncoder => _speakerEncoder ?? throw NotNative();
    internal BigVganGenerator<T> Vocoder => _vocoder ?? throw NotNative();
    internal DenseLayer<T> SpeakerClassifier => _speakerClassifier ?? throw NotNative();

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
        _gpt = new IndexTtsUnifiedVoice<T>(Engine, _gptLayers, VoiceConfiguration(o), random);
        _s2mel = new IndexTtsSemanticToMel<T>(Engine, _s2melLayers, new IndexTtsSemanticToMelConfiguration(o.MelChannels, o.DitHidden, o.DitHeads,
            o.DitDepth, o.StyleDim, o.SemanticHiddenSize, o.RegulatorChannels, o.RegulatorStages, o.WavenetHidden, o.WavenetLayers, o.WavenetKernel,
            o.WavenetDropout, o.ClassDropout, o.HiddenDim), random);
        _speakerEncoder = new CamPlusPlusEncoder<T>(Engine, _speakerEncoderLayers, new CamPlusPlusConfiguration(
            EmbeddingSize: o.StyleDim, GrowthRate: o.SpeakerEncoderGrowthRate, InitialChannels: o.SpeakerEncoderChannels), random);
        _speakerEncoder.SetTrainingMode(false);
        _vocoder = new BigVganGenerator<T>(Engine, random, o.MelChannels, o.VocoderInitialChannels, o.VocoderUpsampleRates, o.VocoderUpsampleKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, snakeBeta: true, tanhAtFinal: false, biasAtFinal: false);
        _speakerClassifier = IndexTtsOps.Linear(Engine, _classifierLayers, random, o.HiddenDim, Math.Max(1, o.SpeakerCount));
        AddUnsplitLayers(_gptLayers.Concat(_s2melLayers).Concat(_classifierLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_semanticModelLayers);
        ComponentLayers.AddRange(_semanticCodecLayers);
        ComponentLayers.AddRange(_speakerEncoderLayers);
        ComponentLayers.AddRange(_vocoder.Layers);
    }

    internal static IndexTtsUnifiedVoiceConfiguration VoiceConfiguration(IndexTTS2Options o) => new(
        o.HiddenDim, o.NumHeads, o.Layers, o.MaxTextLength, o.MaxMelTokens, o.VocabSize, o.MelCodes, 0, 1, o.MelCodes - 2, o.MelCodes - 1,
        o.ConditionLatents, new IndexTtsConformerConfiguration(o.SemanticHiddenSize, o.ConditionDim, o.ConditionLinearUnits, o.ConditionHeads, o.ConditionBlocks),
        o.PerceiverMultiple, new IndexTtsConformerConfiguration(o.SemanticHiddenSize, o.EmotionConditionDim, o.EmotionConditionLinearUnits,
            o.EmotionConditionHeads, o.EmotionConditionBlocks), o.PerceiverMultiple, o.SemanticHiddenSize, o.Duration, o.DropoutRate);

    // ---------------------------------------------------------------- text

    /// <summary>The release's tokenizer (normalizer, CJK splitting, <c>bpe.model</c>).</summary>
    internal IndexTtsTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = _options.BpeModelPath
                ?? throw new InvalidOperationException("Text needs the release's tokenizer; set IndexTTS2Options.BpeModelPath to bpe.model.");
            return _tokenizer = new IndexTtsTokenizer(new SentencePieceModel(File.ReadAllBytes(path)), new IndexTtsTextNormalizer());
        }
    }

    /// <summary>The token ids of <paramref name="text"/> (normalized, CJK-split, BPE).</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Tokenizer.ToIds(Tokenizer.Tokenize(text));
    }

    // ---------------------------------------------------------------- the prompt

    private sealed class PromptInfo
    {
        public Tensor<T> Features = null!;         // normalized w2v-BERT features [T, 1024]
        public Tensor<T> Quantized = null!;        // the semantic codec's features of them [T, 1024]
        public Tensor<T> Mel = null!;              // [frames, mel] at the output rate
        public Tensor<T> Style = null!;            // CAM++ [style]
        public double[] Samples = null!;           // the cut recording at its own rate
        public int SampleRate;
        public Tensor<T>? EmotionFeatures;         // the recording read as a style prompt (librosa's 16 kHz load)
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    private Tensor<T> ResampleTorchaudio(double[] samples, int from, int to) =>
        from == to ? Wave(samples) : AudioHelper<T>.ResampleBandLimited(Engine, Wave(samples), from, to);

    private double[] Cut(double[] samples, int sampleRate)
    {
        int max = (int)(_options.MaxPromptSeconds * sampleRate);
        return samples.Length > max ? samples.Take(max).ToArray() : samples;
    }

    /// <summary>
    /// The normalized w2v-BERT 2.0 features <c>[frames, 1024]</c> of 16 kHz audio (<c>get_emb</c>): SeamlessM4T's
    /// filterbank features, hidden state <see cref="IndexTTS2Options.SemanticLayer"/>, minus the mean over the standard
    /// deviation.
    /// </summary>
    internal Tensor<T> SemanticFeatures(IReadOnlyList<double> audio16k)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _semanticModelLayers.Concat(_semanticCodecLayers)) layer.SetTrainingMode(false);
        var (features, mask) = _features!.ForwardPadded(audio16k);
        var hidden = SemanticModel.HiddenStates(features, mask)[_options.SemanticLayer];
        return SemanticCodec.Normalize(hidden);
    }

    /// <summary>The CAM++ style <c>[192]</c> of 16 kHz audio: 80-band Kaldi filterbank (no dither) minus its mean.</summary>
    internal Tensor<T> Style(Tensor<T> audio16k)
    {
        using var _ = new NoGradScope<T>();
        _fbank ??= new KaldiFbank<T>(Engine, _options.SemanticSampleRate, 80, WindowType.Povey);
        var features = _fbank.Forward(audio16k);
        features = Engine.TensorSubtract(features, Engine.TensorBroadcastTo(Engine.ReduceMean(features, new[] { 0 }, true), features.Shape.ToArray()));
        var style = SpeakerEncoder.Forward(Engine.Reshape(features, new[] { 1, features.Shape[0], features.Shape[1] }));
        return Engine.Reshape(style, new[] { _options.StyleDim });
    }

    /// <summary>The log-mel spectrogram <c>[frames, mel]</c> of audio at the output rate (BigVGAN's
    /// <c>mel_spectrogram</c>: n_fft 1024, hop 256, 0 Hz to Nyquist, not centred).</summary>
    internal Tensor<T> MelSpectrogram(Tensor<T> audio)
    {
        var o = _options;
        _mel ??= new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0, o.SampleRate / 2.0);
        return _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
    }

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/> (the release's prompt handling): the first
    /// 15 seconds, resampled by torchaudio to 16 kHz (its w2v-BERT features condition the GPT, their codes and CAM++ style
    /// condition the flow) and to the output rate (its mel is the flow's prompt).
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var samples = Cut(Doubles(recording), sampleRate);
        var audio = ResampleTorchaudio(samples, sampleRate, _options.SampleRate);
        var voice = new TtsVoice<T> { ReferenceAudio = audio };
        _prompts.Add(voice, PromptOf(audio, samples, sampleRate));
        return voice;
    }

    private PromptInfo PromptOf(Tensor<T> audioAtRate, double[] samples, int sampleRate)
    {
        using var _ = new NoGradScope<T>();
        var audio16k = ResampleTorchaudio(samples, sampleRate, _options.SemanticSampleRate);
        var features = SemanticFeatures(Doubles(audio16k));
        return new PromptInfo
        {
            Samples = samples,
            SampleRate = sampleRate,
            Features = features,
            Quantized = SemanticCodec.Embed(SemanticCodec.EncodeNormalized(features)),
            Mel = MelSpectrogram(audioAtRate),
            Style = Style(audio16k),
        };
    }

    private PromptInfo Prompt(TtsVoice<T> voice)
    {
        if (_prompts.TryGetValue(voice, out var info)) return info;
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        info = PromptOf(audio, Cut(Doubles(audio), _options.SampleRate), _options.SampleRate);
        _prompts.Add(voice, info);
        return info;
    }

    // ---------------------------------------------------------------- emotion

    /// <summary>A recording whose emotion to imitate (the release's <c>emo_audio_prompt</c>), at
    /// <see cref="EmotionReferenceSampleRate"/>; null takes the speaker's.</summary>
    public Tensor<T>? EmotionReference { get; set; }

    /// <summary>The sample rate of <see cref="EmotionReference"/>.</summary>
    public int EmotionReferenceSampleRate { get; set; } = 22050;

    /// <summary>How strongly the emotion reference (or the weights) applies (<c>emo_alpha</c>, 1).</summary>
    public double EmotionAlpha { get; set; } = 1.0;

    /// <summary>Emotion weights, one per <see cref="IndexTTS2Options.EmotionLabels"/> (the release's <c>emo_vector</c>);
    /// they mix the emotion table's vectors and override the emotion reference.</summary>
    public double[]? EmotionWeights { get; set; }

    /// <summary>A sentence whose emotion the text-to-emotion model reads into weights (<c>use_emo_text</c>).</summary>
    public string? EmotionText { get; set; }

    private Tensor<T>? _emotionCacheAudio;
    private Tensor<T>? _emotionCacheFeatures;

    // The release reads a style prompt with librosa at 16 kHz (soxr_hq), cut to 15 seconds — the speaker's recording too
    // when no other is given, unlike the speaker prompt's torchaudio resampling.
    private Tensor<T> StyleFeatures(double[] samples, int sampleRate)
    {
        var audio16k = sampleRate == _options.SemanticSampleRate ? samples : SoxrResampler.LibrosaResample(samples, sampleRate, _options.SemanticSampleRate);
        return SemanticFeatures(Cut(audio16k, _options.SemanticSampleRate));
    }

    private Tensor<T> EmotionFeatures(Tensor<T> recording)
    {
        if (ReferenceEquals(recording, _emotionCacheAudio) && _emotionCacheFeatures is not null) return _emotionCacheFeatures;
        _emotionCacheAudio = recording;
        return _emotionCacheFeatures = StyleFeatures(Doubles(recording), EmotionReferenceSampleRate);
    }

    /// <summary>
    /// The emotion vector e <c>[1, d]</c> for a voice (the release's <c>infer_generator</c>): the style prompt's emotion
    /// moved toward from the speaker's by <see cref="EmotionAlpha"/>; with weights (given, or read from
    /// <see cref="EmotionText"/>), their mix of the emotion table plus (1 − Σw) of that.
    /// </summary>
    internal Tensor<T> EmotionVector(TtsVoice<T> voice, Random random)
    {
        var o = _options;
        var prompt = Prompt(voice);
        double[]? weights = EmotionWeights;
        if (EmotionText is { } text) weights = TextToEmotion().Scores(text, o.EmotionMaxTokens);
        double alpha = EmotionAlpha;
        if (weights is not null)
        {
            if (weights.Length != o.EmotionLabels.Length) throw new InvalidOperationException($"Give one emotion weight per label ({o.EmotionLabels.Length}).");
            double scale = Math.Max(0.0, Math.Min(1.0, alpha));
            if (scale != 1.0) weights = weights.Select(w => Math.Truncate(w * scale * 10000) / 10000).ToArray();
        }
        var emotionAudio = weights is null ? EmotionReference : null;
        if (emotionAudio is null) alpha = 1.0;
        var gpt = TextToSemantic;
        var baseVector = gpt.EmotionVector(prompt.Features);
        var styleFeatures = emotionAudio is not null ? EmotionFeatures(emotionAudio)
            : prompt.EmotionFeatures ??= StyleFeatures(prompt.Samples, prompt.SampleRate);
        var emotion = gpt.EmotionVector(styleFeatures);
        var e = Engine.TensorAdd(baseVector, Engine.TensorMultiplyScalar(Engine.TensorSubtract(emotion, baseVector), Ops.FromDouble(alpha)));
        if (weights is null) return e;
        var mixed = EmotionTableMix(weights, prompt.Style, random);
        return Engine.TensorAdd(mixed, Engine.TensorMultiplyScalar(e, Ops.FromDouble(1 - weights.Sum())));
    }

    /// <summary>The weighted mix <c>[1, d]</c> of the emotion table: per emotion its group's mean (Eq. 5), the row whose
    /// speaker vector is nearest the style, or a random row.</summary>
    private Tensor<T> EmotionTableMix(double[] weights, Tensor<T> style, Random random)
    {
        var o = _options;
        var rows = _emotionRows ?? throw new InvalidOperationException("Emotion weights need the emotion table; load it with LoadEmotionTable.");
        var groups = o.EmotionGroupSizes ?? throw new InvalidOperationException("Emotion weights need IndexTTS2Options.EmotionGroupSizes.");
        int d = o.HiddenDim, start = 0;
        var mix = new double[d];
        var styleValues = Doubles(style);
        for (int g = 0; g < groups.Length; g++)
        {
            var members = Enumerable.Range(start, groups[g]).ToArray();
            start += groups[g];
            double[] vector;
            switch (o.EmotionMode)
            {
                case IndexTts2EmotionMode.Paper:
                    vector = new double[d];
                    foreach (int r in members)
                        for (int k = 0; k < d; k++) vector[k] += rows[r][k] / members.Length;
                    break;
                case IndexTts2EmotionMode.ReleaseRandomRow:
                    vector = rows[members[random.Next(members.Length)]];
                    break;
                default:
                    var speakers = _speakerRows ?? throw new InvalidOperationException("Nearest-speaker rows need the emotion table's speaker vectors.");
                    vector = rows[members.OrderByDescending(r => Cosine(speakers[r], styleValues)).ThenBy(r => r).First()];
                    break;
            }
            for (int k = 0; k < d; k++) mix[k] += weights[g] * vector[k];
        }
        return new Tensor<T>(new[] { 1, d }, new Vector<T>(mix.Select(v => Ops.FromDouble(v)).ToArray()));
    }

    private static double Cosine(double[] a, double[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        // F.cosine_similarity clamps each norm at 1e-8.
        return dot / (Math.Max(Math.Sqrt(na), 1e-8) * Math.Max(Math.Sqrt(nb), 1e-8));
    }

    /// <summary>Sets the emotion table: one emotion vector <c>[d]</c> per row (the release's <c>feat2.pt</c>), grouped by
    /// <see cref="IndexTTS2Options.EmotionGroupSizes"/>, and optionally each row's speaker vector <c>[style]</c>
    /// (<c>feat1.pt</c>) for the nearest-speaker choice.</summary>
    public void LoadEmotionTable(double[][] emotionRows, double[][]? speakerRows = null)
    {
        ThrowIfDisposed();
        if (emotionRows is null) throw new ArgumentNullException(nameof(emotionRows));
        var o = _options;
        if (o.EmotionGroupSizes is not { } groups || groups.Sum() != emotionRows.Length)
            throw new ArgumentException("EmotionGroupSizes must sum to the table's rows.", nameof(emotionRows));
        if (emotionRows.Any(r => r.Length != o.HiddenDim)) throw new ArgumentException($"Emotion rows are {o.HiddenDim} wide.", nameof(emotionRows));
        if (speakerRows is not null && (speakerRows.Length != emotionRows.Length || speakerRows.Any(r => r.Length != o.StyleDim)))
            throw new ArgumentException($"Give one {o.StyleDim}-wide speaker row per emotion row.", nameof(speakerRows));
        _emotionRows = emotionRows;
        _speakerRows = speakerRows;
    }

    private IndexTtsTextToEmotion<T> TextToEmotion()
    {
        if (_t2e is not null) return _t2e;
        var o = _options;
        var tokenizer = o.EmotionModelPath is { } dir && File.Exists(Path.Combine(dir, "tokenizer.json"))
            ? OuteTTSTokenizers.FromFile(Path.Combine(dir, "tokenizer.json")) : OuteTTSTokenizers.Qwen3;
        _t2e = new IndexTtsTextToEmotion<T>(Engine, _t2eLayers, EmotionConfiguration(o), tokenizer, o.EmotionLabels,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.TrainingSeed + 11));
        ComponentLayers.AddRange(_t2eLayers);
        if (o.EmotionModelPath is { } path && File.Exists(Path.Combine(path, "model.safetensors"))) LoadTextToEmotionState(_t2e, path);
        return _t2e;
    }

    internal static LlamaConfiguration EmotionConfiguration(IndexTTS2Options o) => new()
    {
        VocabSize = o.EmotionVocabulary, Hidden = o.EmotionHidden, Layers = o.EmotionLayers, Heads = o.EmotionHeads,
        KeyValueHeads = o.EmotionKeyValueHeads, HeadDim = o.EmotionHeadDim, Intermediate = o.EmotionIntermediate, RmsEpsilon = 1e-6,
        RopeTheta = 1e6, QueryKeyNorm = true, TieWordEmbeddings = true, LoraRank = o.EmotionLoraRank, LoraAlpha = 2 * o.EmotionLoraRank, LoraDropout = 0,
    };

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
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

    private int[] TokenIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] > _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"Text token ids lie in [0, {_options.VocabSize}].");
        }
        return ids;
    }

    /// <summary>The text's segments (<c>split_segments</c> at <see cref="IndexTTS2Options.SegmentTokens"/>) when the
    /// tokenizer is loaded; otherwise the ids in runs of that many.</summary>
    private List<int[]> Segments(int[] ids)
    {
        if (_options.BpeModelPath is not null)
        {
            var pieces = ids.Select(Tokenizer.Model.IdToPiece).ToList();
            return IndexTtsTokenizer.SplitSegments(pieces, _options.SegmentTokens).Select(s => Tokenizer.ToIds(s)).ToList();
        }
        var segments = new List<int[]>();
        for (int i = 0; i < ids.Length; i += _options.SegmentTokens) segments.Add(ids.Skip(i).Take(_options.SegmentTokens).ToArray());
        return segments;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is the speech at
    /// <see cref="TtsModelOptions.SampleRate"/> in the voice's timbre (the release's <c>infer</c>: per segment, semantic
    /// tokens, their GPT latents, the flow's mel after the prompt, BigVGAN; segments joined by silence, the end faded).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = TokenIds(input, nameof(input));
        if (ids.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            var prompt = Prompt(voice);
            var emotion = EmotionVector(voice, random);
            var gpt = TextToSemantic;
            var speaker = gpt.SpeakerConditioning(prompt.Features);
            var decoding = new IndexTtsDecoding(o.Sample, o.Beams, o.TopK, o.TopP, o.Temperature, o.RepetitionPenalty, o.MaxGeneratedTokens);
            var promptCondition = SemanticToMel.Regulator.Forward(prompt.Quantized, prompt.Mel.Shape[0]);
            var waves = new List<double[]>();
            foreach (var segment in Segments(ids))
            {
                var conditions = gpt.Conditions(speaker, emotion, o.TargetSemanticTokens);
                var codes = gpt.Generate(conditions, segment, decoding, random);
                if (codes.Length == 0) continue;
                waves.Add(SegmentAudio(conditions, segment, codes, prompt, promptCondition, random));
            }
            return Wave(Join(waves));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>One segment's audio: the semantic features plus the fused GPT latents, regulated to
    /// ⌊1.72 n⌋ frames after the prompt's, the flow's mel and BigVGAN.</summary>
    private double[] SegmentAudio(Tensor<T> conditions, int[] text, int[] codes, PromptInfo prompt, Tensor<T> promptCondition, Random random)
    {
        var o = _options;
        var latents = Engine.TensorNarrow(TextToSemantic.Latents(conditions, text, codes), 0, 0, codes.Length);
        var semantic = SemanticToMel.Fuse(SemanticCodec.Embed(codes), latents);
        int frames = (int)(codes.Length * o.FramesPerToken);
        if (frames < 1) return Array.Empty<double>();
        var condition = Engine.TensorConcatenate(new[] { promptCondition, SemanticToMel.Regulator.Forward(semantic, frames) }, 0);
        int total = condition.Shape[0];
        var noise = new Tensor<T>(new[] { total, o.MelChannels });
        for (int i = 0; i < noise.Length; i++) noise[i] = Ops.FromDouble(IndexTtsOps.Gaussian(random));
        var mel = SemanticToMel.Solve(condition, prompt.Mel, prompt.Style, noise, o.DiffusionSteps, o.Guidance);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, o.MelChannels, mel.Shape[0] }));
        return Doubles(wave);
    }

    /// <summary>Segments joined by <see cref="IndexTTS2Options.SegmentSilenceMilliseconds"/> of silence, the last 20 ms
    /// faded by a raised cosine (<c>fade_out_tail</c>).</summary>
    private double[] Join(List<double[]> waves)
    {
        int silence = (int)(_options.SampleRate * _options.SegmentSilenceMilliseconds / 1000.0);
        var joined = new List<double>();
        for (int i = 0; i < waves.Count; i++)
        {
            if (i > 0) joined.AddRange(new double[silence]);
            joined.AddRange(waves[i]);
        }
        var result = joined.ToArray();
        int n = Math.Min((int)(_options.SampleRate * 20 / 1000.0), result.Length);
        if (n > 1)
            for (int i = 0; i < n; i++)
                result[result.Length - n + i] *= 0.5 * (1 + Math.Cos(Math.PI * i / (n - 1)));
        return result;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "IndexTTS2-Native" : "IndexTTS2-ONNX",
            Description = "IndexTTS2: emotionally expressive, duration-controlled autoregressive zero-shot TTS (Zhou et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "IndexTTS2";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Duration"] = _options.Duration.ToString();
        metadata.AdditionalInfo["EmotionMode"] = _options.EmotionMode.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IndexTTS2<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
