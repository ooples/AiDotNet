using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// GPT-SoVITS (RVC-Boss, 2024): few-shot voice cloning in which a GPT writes 25 Hz semantic tokens for the text,
/// continuing a reference recording's, and a synthesizer speaks them in the reference voice: SoVITS's VITS-style GAN for
/// v1 to v2ProPlus, conditional flow matching to mels and a vocoder for v3, v4 and V5.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The pipeline is the release's <c>inference_webui.get_tts_wav</c>: the target text cut into sentences, each
/// read by the language front ends into phones (and, for Chinese, chinese-roberta's features), the reference's
/// chinese-hubert features quantized into semantic tokens, the GPT (<c>Text2SemanticDecoder</c>) sampling new tokens
/// after the prompt's phones and tokens, and the synthesizer decoding them with the reference's style (plus, for v2Pro,
/// an ERes2NetV2 speaker embedding). Sentences are joined by silence.</para>
/// <para>There is no paper; the release (RVC-Boss/GPT-SoVITS) is the reference, each <see cref="GptSoVitsVersion"/>
/// with its configuration. Training follows its three scripts: s1 for the GPT, s2 for the GAN synthesizer and the
/// flow-matching synthesizer's LoRA (<see cref="GPTSoVITSOptions.CurrentStage"/>).</para>
/// <para><b>For Beginners:</b> Record a few seconds of someone speaking, write down what they said, and GPT-SoVITS reads
/// new text in that voice. It works in Chinese, English, Japanese, Korean and Cantonese.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var options = new GPTSoVITSOptions { TextLanguage = GptSoVitsTextLanguage.English, PromptLanguage = GptSoVitsTextLanguage.English };
/// var model = new GPTSoVITS&lt;float&gt;(architecture, options);
/// model.LoadReferenceWeights(new GptSoVitsCheckpoints { Directory = "GPT-SoVITS/pretrained_models" });
/// model.Voice = model.CreateVoice(referenceAudio, 32000, "What the reference recording says.");
/// var audio = model.Synthesize("Text to speak in that voice.");   // 32 kHz for v2ProPlus
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GPT-SoVITS: 1 min voice data can also be used to train a good TTS model", "https://github.com/RVC-Boss/GPT-SoVITS",
    Year = 2024, Authors = "RVC-Boss")]
[PaperOptimizer(OptimizerKind.ScaledAdam, LearningRate = 0.01, Beta1 = 0.9, Beta2 = 0.95, Component = "t2s",
                Source = "RVC-Boss/GPT-SoVITS AR/models/t2s_lightning_module.py: ScaledAdam (lr 0.01, betas 0.9/0.95, clipping "
                        + "scale 2, clipping update period 1000), stepped every fourth batch, WarmupCosineLRSchedule locking the "
                        + "rate at 0.002 after its first step.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, Component = "sovits",
                Source = "RVC-Boss/GPT-SoVITS s2_train.py and configs/s2.json: AdamW (betas 0.8/0.99, eps 1e-9, PyTorch's default "
                        + "weight decay) at 1e-4, the text embedding, text encoder and MRTE at 0.4 times it, ExponentialLR 0.999875 "
                        + "per epoch.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, Component = "discriminator",
                Source = "RVC-Boss/GPT-SoVITS s2_train.py: the multi-period discriminator's AdamW, as the generator's.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, Component = "cfm",
                Source = "RVC-Boss/GPT-SoVITS s2_train_v3_lora.py: AdamW over the trainable parameters (LoRA rank 32 by default) "
                        + "at s2.json's 1e-4, ExponentialLR 0.999875 per epoch.")]
public partial class GPTSoVITS<T> : TtsModelBase<T>
{
    private readonly GPTSoVITSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private GptSoVitsT2S<T>? _t2s;
    private GptSoVitsSynthesizer<T>? _sovits;
    private GptSoVitsCfmSynthesizer<T>? _cfm;
    private HiFiGanDiscriminators<T>? _discriminators;
    private BigVganGenerator<T>? _bigVgan;
    private HiFiGanGenerator<T>? _hifiGan;
    private HubertNetwork<T>? _hubert;
    private BertNetwork<T>? _bert;
    private ERes2NetV2Encoder<T>? _speaker;
    private G2pwConverter<T>? _g2pw;
    private GptSoVitsLangSegmenter? _segmenter;
    private GptSoVitsJapaneseFrontEnd? _japanese;
    private GptSoVitsText? _text;

    private readonly List<LayerBase<T>> _t2sLayers = new();
    private readonly List<LayerBase<T>> _synthesizerLayers = new();
    private readonly List<LayerBase<T>> _loraLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private readonly List<LayerBase<T>> _bertLayers = new();
    private readonly List<LayerBase<T>> _g2pwLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed GPT-SoVITS for inference.</summary>
    public GPTSoVITS(NeuralNetworkArchitecture<T> architecture, string modelPath, GPTSoVITSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GPTSoVITSOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable GPT-SoVITS.</summary>
    public GPTSoVITS(NeuralNetworkArchitecture<T> architecture, GPTSoVITSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GPTSoVITSOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        // The output rate is the version's: SoVITS's 32 kHz, v3's 24 kHz BigVGAN, v4 and V5's 48 kHz vocoder.
        o.SampleRate = IsCfm ? (o.Version == GptSoVitsVersion.V3 ? 24000 : 48000) : 32000;
        base.SampleRate = o.SampleRate;
        base.MelChannels = IsCfm ? 100 : o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        // The release clips nothing: ScaledAdam bounds s1's steps itself and s2 calls clip_grad_value_(…, None).
        MaxGradNorm = NumOps.Zero;
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.UpsampleRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException($"The GAN generator's upsampling rates must multiply to HopSize {o.HopSize}.");
        if (o.SegmentSize % o.HopSize != 0) throw new ArgumentException("SegmentSize must be a whole number of hops.");
        if (o.T2SAccumulation < 1) throw new ArgumentException("T2SAccumulation must be positive.");
        if (o.LoraRank < 0) throw new ArgumentException("LoraRank must be 0 (no LoRA) or positive.");
        if (o.Speed <= 0) throw new ArgumentException("Speed must be positive.");
        // ERes2NetV2's Res2Net splits are ⌊m_channels · 24/64⌋ wide.
        if (IsV2Pro && o.SpeakerChannels * 24 / 64 < 1) throw new ArgumentException("SpeakerChannels must be at least 3 (ERes2NetV2's splits are ⌊m·24/64⌋ wide).");
        if (o.CurrentStage == GptSoVitsStage.SoVits && IsCfm)
            throw new ArgumentException($"{VersionName} synthesizes by flow matching; train it with GptSoVitsStage.FlowMatching.");
        if (o.CurrentStage == GptSoVitsStage.FlowMatching && !IsCfm)
            throw new ArgumentException($"{VersionName} synthesizes with the GAN; train it with GptSoVitsStage.SoVits.");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool IsCfm => _options.Version is GptSoVitsVersion.V3 or GptSoVitsVersion.V4 or GptSoVitsVersion.V5Dev or GptSoVitsVersion.V5Turbo;

    private bool IsV2Pro => _options.Version is GptSoVitsVersion.V2Pro or GptSoVitsVersion.V2ProPlus;

    private bool IsV1 => _options.Version == GptSoVitsVersion.V1;

    /// <summary>The release's name of the version ("v1" … "v5turbo").</summary>
    internal string VersionName => _options.Version switch
    {
        GptSoVitsVersion.V1 => "v1",
        GptSoVitsVersion.V2 => "v2",
        GptSoVitsVersion.V2Pro => "v2Pro",
        GptSoVitsVersion.V2ProPlus => "v2ProPlus",
        GptSoVitsVersion.V3 => "v3",
        GptSoVitsVersion.V4 => "v4",
        GptSoVitsVersion.V5Dev => "v5dev",
        _ => "v5turbo",
    };

    private bool HasPaperLayers => _t2s is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal GptSoVitsT2S<T> TextToSemantic => _t2s ?? throw NotNative();
    internal GptSoVitsSynthesizer<T> SoVits => _sovits ?? throw new InvalidOperationException($"{VersionName} has no GAN synthesizer.");
    internal GptSoVitsCfmSynthesizer<T> FlowSynthesizer => _cfm ?? throw new InvalidOperationException($"{VersionName} has no flow-matching synthesizer.");
    internal HubertNetwork<T> Hubert => _hubert ?? throw NotNative();
    internal BertNetwork<T> Bert => _bert ?? throw NotNative();

    /// <summary>The synthesizer's configuration: the version's s2 configuration with the options' sizes.</summary>
    internal GptSoVitsSynthesizerConfiguration SynthesizerConfiguration()
    {
        var o = _options;
        var c = GptSoVitsSynthesizerConfiguration.For(VersionName);
        return c with
        {
            InterChannels = o.InterChannels, Hidden = o.SynthesizerHidden, Filter = o.SynthesizerFilter, Heads = o.SynthesizerHeads,
            Layers = o.SynthesizerLayers, Gin = o.StyleChannels ?? c.Gin, MrteChannels = o.MrteChannels, StyleHidden = o.StyleHidden,
            Dropout = o.SynthesizerDropout ?? c.Dropout, UpsampleRates = o.UpsampleRates, UpsampleKernels = o.UpsampleKernels ?? c.UpsampleKernels,
            UpsampleInitialChannels = o.UpsampleInitialChannels ?? c.UpsampleInitialChannels, ResblockKernels = o.ResblockKernels,
            ResblockDilations = o.ResblockDilations, SslDim = o.HubertHidden, Bins = o.SemanticCodebookSize, SvDim = SpeakerEmbeddingSize,
        };
    }

    private int SpeakerEmbeddingSize => _options.SpeakerChannels * 8 * 4 * 10;

    internal GptSoVitsT2SConfiguration T2SConfiguration() =>
        new(_options.HiddenDim, _options.NumHeads, _options.T2SLayers, _options.SemanticCodebookSize + 1, SynthesizerConfiguration().Symbols,
            BertDim: _options.BertHidden);

    internal GptSoVitsDiTConfiguration DiTConfiguration() =>
        new(_options.DitDim, _options.DitDepth, _options.DitHeads, _options.DitHeadDim, TextDim: _options.BridgeChannels,
            ConvLayers: _options.DitConvLayers, UseStepEmbedding: _options.Version is not (GptSoVitsVersion.V5Dev or GptSoVitsVersion.V5Turbo));

    internal HubertConfiguration HubertConfiguration() => new()
    {
        ConvDims = _options.HubertConvDims, LayerNormFeatureEncoder = false, HiddenSize = _options.HubertHidden, Layers = _options.HubertLayers,
        Heads = _options.HubertHeads, IntermediateSize = _options.HubertIntermediate, PositionalConvKernel = _options.HubertPositionKernel,
        PositionalConvGroups = _options.HubertPositionGroups, LayerNormFirst = false,
    };

    internal BertConfiguration BertConfiguration() => new()
    {
        VocabSize = 21128, Hidden = _options.BertHidden, Layers = _options.BertLayers, Heads = _options.BertHeads, Intermediate = _options.BertIntermediate,
    };

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(TextToSemantic.Configuration.PhonemeVocab) : base.GetInputDomain(inputShape);

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
        _t2s = new GptSoVitsT2S<T>(Engine, _t2sLayers, T2SConfiguration(), random);
        var synthesizer = SynthesizerConfiguration();
        if (IsCfm)
        {
            _cfm = new GptSoVitsCfmSynthesizer<T>(Engine, _synthesizerLayers, synthesizer, DiTConfiguration(), random, o.BridgeChannels);
            if (o.Version == GptSoVitsVersion.V3)
                _bigVgan = new BigVganGenerator<T>(Engine, random, 100, o.VocoderInitialChannels ?? 1536, o.VocoderUpsampleRates ?? new[] { 4, 4, 2, 2, 2, 2 },
                    o.VocoderUpsampleKernels ?? new[] { 8, 8, 4, 4, 4, 4 }, o.VocoderResblockKernels, o.VocoderResblockDilations, snakeBeta: true,
                    tanhAtFinal: false, biasAtFinal: false);
            else
                _hifiGan = new HiFiGanGenerator<T>(Engine, 100, o.VocoderInitialChannels ?? 512, o.VocoderUpsampleRates ?? new[] { 10, 6, 2, 2, 2 },
                    o.VocoderUpsampleKernels ?? new[] { 20, 12, 4, 4, 4 }, o.VocoderResblockKernels, o.VocoderResblockDilations, resblockType1: true,
                    plainEnds: true, initialization: random, plainOutputBias: true);
            if (o.LoraRank > 0) _cfm.Cfm.Estimator.AddLora(_loraLayers, o.LoraRank, o.LoraRank, random);
        }
        else
        {
            _sovits = new GptSoVitsSynthesizer<T>(Engine, _synthesizerLayers, synthesizer, random);
            var periods = IsV2Pro ? new[] { 2, 3, 5, 7, 11, 17, 23 } : new[] { 2, 3, 5, 7, 11 };
            _discriminators = new HiFiGanDiscriminators<T>(Engine, periods, 1, true, 1, spectralFirstScale: false, vitsScale: true);
        }
        _hubert = new HubertNetwork<T>(Engine, _hubertLayers, HubertConfiguration());
        _bert = new BertNetwork<T>(Engine, _bertLayers, BertConfiguration());
        if (IsV2Pro)
            _speaker = new ERes2NetV2Encoder<T>(Engine, _ => null, o.SpeakerChannels, blocks: o.SpeakerBlocks, initialization: random);
        AddUnsplitLayers(_t2sLayers.Concat(_synthesizerLayers).Concat(_loraLayers).Cast<ILayer<T>>().ToList());
        if (_discriminators is not null) ComponentLayers.AddRange(_discriminators.Layers);
        if (_bigVgan is not null) ComponentLayers.AddRange(_bigVgan.Layers);
        if (_hifiGan is not null) ComponentLayers.AddRange(_hifiGan.Layers);
        ComponentLayers.AddRange(_hubertLayers);
        ComponentLayers.AddRange(_bertLayers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The release's text processing with this model's front ends (made on first use).</summary>
    internal GptSoVitsText Text => _text ??= new GptSoVitsText(Segmenter, Japanese, IsV1 ? null : new Func<List<string>, List<List<string?>>>(G2pw), BertFeatures, IsV1);

    private GptSoVitsLangSegmenter Segmenter() => _segmenter ??= new GptSoVitsLangSegmenter(_options.LanguageModelPath
        ?? throw new InvalidOperationException("Reading mixed-language text needs fastText's lid.176.bin; set GPTSoVITSOptions.LanguageModelPath."));

    private GptSoVitsJapaneseFrontEnd Japanese() => _japanese ??= new GptSoVitsJapaneseFrontEnd(_options.OpenJTalkDictionaryPath
        ?? throw new InvalidOperationException("Japanese needs Open JTalk's dictionary; set GPTSoVITSOptions.OpenJTalkDictionaryPath."));

    private List<List<string?>> G2pw(List<string> sentences)
    {
        if (_g2pw is null)
        {
            string dir = _options.G2pwDirectory
                ?? throw new InvalidOperationException("Chinese after v1 reads G2PW; set GPTSoVITSOptions.G2pwDirectory to G2PWModel_1.1.");
            var model = new G2pwModel<T>(Engine, _g2pwLayers, new G2pwConfiguration());
            var weights = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(Path.Combine(dir, "g2pW.onnx"));
            model.Load((name, _) =>
            {
                if (!weights.TryGetValue(name, out var tensor)) throw new InvalidDataException($"g2pW.onnx has no tensor '{name}'.");
                weights.Remove(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            _g2pw = new G2pwConverter<T>(model);
        }
        using var _ = new NoGradScope<T>();
        return _g2pw.Predict(sentences);
    }

    /// <summary><c>get_bert_feature</c>: chinese-roberta's hidden state −3 for each character of the normalized text
    /// (without [CLS] and [SEP]), repeated for the character's phones.</summary>
    private double[][] BertFeatures(string normalized, IReadOnlyList<int> wordToPhones)
    {
        var ids = MeloTextPipeline.BertTokenIds(MeloLanguage.Chinese, normalized);
        if (ids.Length - 2 < wordToPhones.Count)
            throw new ArgumentException($"BERT reads {ids.Length - 2} tokens of the text but the front end gave {wordToPhones.Count} characters.", nameof(normalized));
        var input = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) input[0, i] = ids[i];
        Tensor<T> hidden;
        bool wasTraining = Bert.Training;
        Bert.Training = false;
        try
        {
            using var _ = new NoGradScope<T>();
            hidden = Bert.Encode(input, layerCount: Bert.Configuration.Layers - 2);
        }
        finally
        {
            Bert.Training = wasTraining;
        }
        int width = Bert.Configuration.Hidden;
        var rows = new List<double[]>();
        for (int i = 0; i < wordToPhones.Count; i++)
        {
            var row = new double[width];
            for (int h = 0; h < width; h++) row[h] = Ops.ToDouble(hidden[0, i + 1, h]);
            for (int r = 0; r < wordToPhones[i]; r++) rows.Add(row);
        }
        return rows.ToArray();
    }

    private Tensor<T> BertTensor(IReadOnlyList<double[]> rows)
    {
        int width = _options.BertHidden;
        var t = new Tensor<T>(new[] { rows.Count, width });
        for (int i = 0; i < rows.Count; i++)
            for (int h = 0; h < width; h++) t[i, h] = Ops.FromDouble(rows[i][h]);
        return t;
    }

    // ---------------------------------------------------------------- audio features

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(IReadOnlyList<double> samples) =>
        new(new[] { samples.Count }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    // torchaudio's Resample (get_spepc, the v3+ reference mel, 2-get-sv) — the release's default resampler.
    private Tensor<T> ResampleTorchaudio(double[] samples, int from, int to) =>
        from == to ? Wave(samples) : AudioHelper<T>.ResampleBandLimited(Engine, Wave(samples), from, to);

    /// <summary>chinese-hubert-base's last hidden state <c>[1, 768, frames]</c> of 16 kHz audio (<c>ssl_model.model</c>,
    /// with no feature-extractor normalization, as the release calls it).</summary>
    internal Tensor<T> SslFeatures(IReadOnlyList<double> audio16k)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _hubertLayers) layer.SetTrainingMode(false);
        var hidden = Hubert.Forward(Wave(audio16k), _options.HubertLayers)[_options.HubertLayers];   // [frames, hidden]
        return Engine.Reshape(Engine.TensorTranspose(hidden), new[] { 1, hidden.Shape[1], hidden.Shape[0] });
    }

    /// <summary>The semantic tokens of SSL features (<c>vq_model.extract_latent</c>).</summary>
    internal int[] SemanticTokens(Tensor<T> ssl)
    {
        using var _ = new NoGradScope<T>();
        return IsCfm ? FlowSynthesizer.Quantizer.Encode(FlowSynthesizer.SslProjection.Forward(ssl)) : SoVits.ExtractLatent(ssl);
    }

    private DifferentiableMel<T>? _spectrogram;

    private DifferentiableMel<T> SpectrogramTransform => _spectrogram ??=
        new DifferentiableMel<T>(Engine, 32000, 2048, 640, 2048, _options.MelChannels, 0, 16000, singlePrecisionBasis: true);

    /// <summary><c>spectrogram_torch(audio, 2048, 32000, 640, 2048, center=False)</c>: <c>[1, 1025, frames]</c>.</summary>
    internal Tensor<T> LinearSpectrogram(Tensor<T> audio32k)
    {
        var magnitude = SpectrogramTransform.Magnitude(Engine.Reshape(audio32k, new[] { audio32k.Length }), 1e-8);   // [frames, 1025]
        return Engine.Reshape(Engine.TensorTranspose(magnitude), new[] { 1, magnitude.Shape[1], magnitude.Shape[0] });
    }

    /// <summary>ERes2NetV2's embedding <c>[20480]</c> of 16 kHz audio (<c>compute_embedding3</c>).</summary>
    internal Tensor<T> SpeakerEmbedding(Tensor<T> audio16k)
    {
        var speaker = _speaker ?? throw new InvalidOperationException($"{VersionName} has no speaker-verification model.");
        using var _ = new NoGradScope<T>();
        return speaker.Embedding(audio16k);
    }

    // ---------------------------------------------------------------- the voice

    private sealed class PromptInfo
    {
        public int Generation;               // the weights it was computed with
        public double[] Samples = Array.Empty<double>();
        public int SampleRate;
        public string? PromptText;
        public int[]? Codes;                 // the prompt's semantic tokens (null: reference-free)
        public int[] Phones = Array.Empty<int>();
        public double[][] Bert = Array.Empty<double[]>();
        public Tensor<T> Spectrogram = null!; // [1, 1025, frames] at 32 kHz
        public Tensor<T>? SpeakerEmbedding;  // v2Pro: [20480]
        public Tensor<T>? Style;             // v3+: ge
        public Tensor<T>? Features;          // v3+: decode_encp of the prompt [1, 512, T]
        public Tensor<T>? Mel;               // v3+: the normalized reference mel [1, 100, T]
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();
    private int _weightsGeneration;

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/> and what it says
    /// (<paramref name="promptText"/>, in <see cref="GPTSoVITSOptions.PromptLanguage"/>), as <c>get_tts_wav</c> reads its
    /// reference: resampled to 16 kHz (resampy kaiser_best) with 0.3 s of silence appended for chinese-hubert's semantic
    /// tokens, to 32 kHz (torchaudio) for the style spectrogram and, for v2Pro, the speaker embedding, and for v3 to V5 to
    /// the mel rate for the flow's prompt. Without prompt text v1 to v2ProPlus synthesize reference-free (the style only);
    /// v3 to V5 need it.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? promptText = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var samples = Doubles(recording);
        var voice = new TtsVoice<T> { ReferenceAudio = recording };
        _prompts.Add(voice, PromptOf(samples, sampleRate, promptText));
        return voice;
    }

    private PromptInfo PromptOf(double[] samples, int sampleRate, string? promptText)
    {
        using var _ = new NoGradScope<T>();
        if (string.IsNullOrEmpty(promptText) && IsCfm) throw new ArgumentException($"{VersionName} needs the reference's text.", nameof(promptText));
        var info = new PromptInfo { Generation = _weightsGeneration, Samples = samples, SampleRate = sampleRate, PromptText = promptText };
        if (!string.IsNullOrEmpty(promptText))
        {
            var wav16k = GptSoVitsAudio.Resample(samples, sampleRate, 16000).Concat(new double[(int)(32000 * _options.PauseSeconds)]).ToArray();
            info.Codes = SemanticTokens(SslFeatures(wav16k));
            string text = GptSoVitsText.EndSentence(promptText!.Trim('\n'), _options.PromptLanguage);
            var (phones, bert, _) = Text.PhonesAndBert(text, _options.PromptLanguage);
            info.Phones = phones;
            info.Bert = bert;
        }
        // get_spepc: torchaudio to 32 kHz, scaled down when it peaks above 1 (by at most half).
        var audio32k = Doubles(ResampleTorchaudio(samples, sampleRate, 32000));
        double peak = audio32k.Length == 0 ? 0 : audio32k.Max(Math.Abs);
        if (peak > 1) audio32k = audio32k.Select(v => v / Math.Min(2, peak)).ToArray();
        info.Spectrogram = LinearSpectrogram(Wave(audio32k));
        if (IsV2Pro) info.SpeakerEmbedding = SpeakerEmbedding(ResampleTorchaudio(audio32k, 32000, 16000));
        if (IsCfm)
        {
            var cfm = FlowSynthesizer;
            info.Style = cfm.Style(info.Spectrogram);
            info.Features = cfm.DecodeFeatures(info.Codes!, info.Phones, info.Style);
            info.Mel = cfm.ReferenceMel(ResampleTorchaudio(samples, sampleRate, cfm.MelSampleRate));
        }
        return info;
    }

    private PromptInfo Prompt(TtsVoice<T> voice)
    {
        if (_prompts.TryGetValue(voice, out var info))
        {
            if (info.Generation == _weightsGeneration) return info;
            // Computed with weights since replaced: recompute from the same recording and text.
            var fresh = PromptOf(info.Samples, info.SampleRate, info.PromptText);
            _prompts.Remove(voice);
            _prompts.Add(voice, fresh);
            return fresh;
        }
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        info = PromptOf(Doubles(audio), _options.SampleRate, null);
        _prompts.Add(voice, info);
        return info;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's phone ids (<see cref="GptSoVitsText"/>'s routing in <see cref="GPTSoVITSOptions.TextLanguage"/>);
    /// fed to <see cref="PredictCore"/> they are synthesized with zero BERT features, as the release reads every language
    /// but Chinese.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var (phones, _, _) = Text.PhonesAndBert(GptSoVitsText.EndSentence(text, _options.TextLanguage), _options.TextLanguage);
        return Wave(phones.Select(p => (double)p).ToArray());
    }

    /// <inheritdoc />
    /// <remarks><c>get_tts_wav</c>: the text cut into sentences (<see cref="GPTSoVITSOptions.Cut"/>), each read into
    /// phones and BERT features, given semantic tokens by the GPT and spoken by the synthesizer; each sentence's audio,
    /// scaled down when it peaks above 1, followed by <see cref="GPTSoVITSOptions.PauseSeconds"/> of silence.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(base.PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var voice = RequireVoice();
        var prompt = Prompt(voice);
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        var output = new List<double>();
        foreach (var sentence in GptSoVitsText.Sentences(text.Trim('\n'), o.Cut))
        {
            if (PythonText.Strip(sentence).Length == 0) continue;
            var (phones, bert, _) = Text.PhonesAndBert(GptSoVitsText.EndSentence(sentence, o.TextLanguage), o.TextLanguage);
            AppendSentence(output, SentenceAudio(phones, bert, prompt, random));
        }
        return Wave(output);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is one sentence's phone ids (zero BERT features); the result is its audio at
    /// <see cref="TtsModelOptions.SampleRate"/> followed by the pause.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        int vocabulary = TextToSemantic.Configuration.PhonemeVocab;
        var phones = new int[input.Length];
        for (int i = 0; i < phones.Length; i++)
        {
            phones[i] = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (phones[i] < 0 || phones[i] >= vocabulary) throw new ArgumentOutOfRangeException(nameof(input), $"Phone ids lie in [0, {vocabulary}).");
        }
        if (phones.Length == 0) throw new ArgumentException("There are no phones.", nameof(input));
        var prompt = Prompt(RequireVoice());
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var output = new List<double>();
        AppendSentence(output, SentenceAudio(phones, phones.Select(_ => new double[_options.BertHidden]).ToArray(), prompt, random));
        return Wave(output);
    }

    private void AppendSentence(List<double> output, double[] audio)
    {
        double peak = audio.Length == 0 ? 0 : audio.Max(Math.Abs);
        output.AddRange(peak > 1 ? audio.Select(v => v / peak) : audio);
        // zero_wav is sized at the s2 configuration's 32 kHz whatever the version's output rate.
        output.AddRange(new double[(int)(32000 * _options.PauseSeconds)]);
    }

    /// <summary>One sentence: the GPT's tokens after the prompt's, then the synthesizer.</summary>
    private double[] SentenceAudio(int[] phones, double[][] bert, PromptInfo prompt, Random random)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            bool referenceFree = prompt.Codes is null;
            var allPhones = referenceFree ? phones : prompt.Phones.Concat(phones).ToArray();
            var allBert = referenceFree ? bert : prompt.Bert.Concat(bert).ToArray();
            var sampling = new GptSoVitsSampling(o.TopK, o.TopP, o.Temperature, o.RepetitionPenalty, EarlyStopTokens: 50 * o.MaxSeconds);
            var tokens = TextToSemantic.Generate(TextToSemantic.TextRows(allPhones, BertTensor(allBert)), prompt.Codes, sampling, random);
            if (tokens.Count == 0) return Array.Empty<double>();
            Tensor<T> Gaussian(int[] shape)
            {
                var noise = new Tensor<T>(shape);
                for (int i = 0; i < noise.Length; i++) noise.SetFlat(i, Ops.FromDouble(AiDotNet.TextToSpeech.Latest.IndexTtsOps.Gaussian(random)));
                return noise;
            }
            if (!IsCfm)
                return Doubles(SoVits.Decode(tokens, phones, new[] { prompt.Spectrogram }, IsV2Pro ? new[] { prompt.SpeakerEmbedding } : null,
                    o.NoiseScale, o.Speed, Gaussian));
            var cfm = FlowSynthesizer;
            var target = cfm.DecodeFeatures(tokens, phones, prompt.Style!, o.Speed);
            var mel = cfm.SynthesizeMel(prompt.Features!, prompt.Mel!, target, o.SampleSteps ?? cfm.DefaultSteps, o.CfgRate ?? cfm.DefaultCfgRate, Gaussian);
            mel = cfm.Denormalize(mel);
            return Doubles(_bigVgan is not null ? _bigVgan.Forward(mel) : _hifiGan!.Forward(mel));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
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
            Name = _useNativeMode ? "GPTSoVITS-Native" : "GPTSoVITS-ONNX",
            Description = "GPT-SoVITS: few-shot voice cloning with a semantic-token GPT and a SoVITS or flow-matching synthesizer (RVC-Boss, 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "GPTSoVITS";
        metadata.AdditionalInfo["Version"] = VersionName;
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Stage"] = _options.CurrentStage.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GPTSoVITS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
