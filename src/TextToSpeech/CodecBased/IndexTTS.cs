using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// IndexTTS (Deng et al. 2025): industrial zero-shot text-to-speech — a GPT-2 predicts a recording's DVAE mel codes from
/// BPE text tokens, conditioned on a conformer-perceiver encoding of a prompt recording, and BigVGAN turns the GPT's
/// latents into audio in the prompt speaker's voice.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The GPT reads <c>[c, BT text ET, BM codes]</c>: c, 32 latents a conformer and perceiver pool from the prompt's
/// 100-band log mel; the text, the release's normalized and CJK-split BPE tokens; the codes, the DVAE's (one per 1024
/// samples of 24 kHz audio). Generation is the release's beam search; the latents of the generated codes, after long
/// silences are shrunk, go to BigVGAN with an ECAPA-TDNN embedding of the prompt's mel added before and after every
/// upsampling.</para>
/// <para>The paper describes the system and its choices; its sizes, the codebook (a VQ DVAE in the release, where the
/// paper weighs FSQ) and the decoding come from the release (<c>IndexTeam/Index-TTS</c>), which the released weights need.
/// Neither publishes training code; training follows XTTS's GPT recipe (<see cref="IndexTTSOptions"/>).</para>
/// <para><b>For Beginners:</b> Give IndexTTS a few seconds of someone speaking and it reads new text in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new IndexTTS&lt;float&gt;(architecture, new IndexTTSOptions());
/// model.LoadReferenceWeights("Index-TTS");             // gpt.pth, bigvgan_generator.pth, dvae.pth, bpe.model
/// model.Voice = model.CreateVoice(promptAudio, promptSampleRate);
/// var audio = model.Synthesize("Text to speak in that voice.");   // 24 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System",
    "https://arxiv.org/abs/2502.05512", Year = 2025, Authors = "Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-6, Beta1 = 0.9, Beta2 = 0.96, Epsilon = 1e-8, WeightDecay = 0.01,
    Source = "The paper and the release publish no training recipe. IndexTTS's UnifiedVoice is XTTS's GPT, whose recipe "
             + "(coqui TTS, recipes/ljspeech/xtts_v2/train_gpt_xtts.py) trains with AdamW at 5e-6, betas (0.9, 0.96), eps 1e-8, "
             + "weight decay 0.01 on weights only, MultiStepLR halving at steps 900k, 2.7M and 5.4M, and losses "
             + "0.01 x text CE + 1.0 x mel CE.")]
public partial class IndexTTS<T> : TtsModelBase<T>
{
    private readonly IndexTTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private IndexTts1UnifiedVoice<T>? _gpt;
    private IndexTtsEcapa<T>? _speakerEncoder;
    private IndexTtsDvae<T>? _dvae;
    private BigVganGenerator<T>? _vocoder;
    private readonly List<LayerBase<T>> _gptLayers = new();
    private readonly List<LayerBase<T>> _speakerEncoderLayers = new();
    private readonly List<LayerBase<T>> _dvaeLayers = new();

    private IndexTtsTokenizer? _tokenizer;
    private CenteredLogMel<T>? _mel;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed IndexTTS for inference.</summary>
    public IndexTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, IndexTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new IndexTTSOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable IndexTTS.</summary>
    public IndexTTS(NeuralNetworkArchitecture<T> architecture, IndexTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new IndexTTSOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.MelCodes < 3) throw new ArgumentException("MelCodes holds the DVAE's codes and the start and stop codes.");
        if (o.VocoderUpsampleRates.Aggregate(1, (a, b) => a * b) != o.SamplesPerCode)
            throw new ArgumentException($"BigVGAN's upsampling rates must multiply to SamplesPerCode {o.SamplesPerCode}.");
        if ((o.VocoderInitialChannels >> o.VocoderUpsampleRates.Length) < 1)
            throw new ArgumentException("Each BigVGAN stage halves the channels: VocoderInitialChannels must keep at least one through them all.");
        if (o.VocoderResBlockDilations.Length != o.VocoderResBlockKernels.Length || o.VocoderResBlockDilations.Any(d => d.Length != 3))
            throw new ArgumentException("The release's AMP block takes three dilations per kernel.");
        if (o.SamplesPerCode % o.HopSize != 0 || (1 << o.DvaeLayers) != o.SamplesPerCode / o.HopSize)
            throw new ArgumentException("The DVAE's stride-2 layers must reduce the mel frames to one per SamplesPerCode samples.");
        if (o.Beams < 1) throw new ArgumentException("Beams must be positive.");
        if (o.SegmentTokens < 1) throw new ArgumentException("SegmentTokens must be positive.");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.SamplesPerCode;
        base.HiddenDim = o.HiddenDim;
        // XTTS's GPT recipe does not clip the gradient.
        MaxGradNorm = NumOps.Zero;
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _gpt is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal IndexTts1UnifiedVoice<T> TextToMel => _gpt ?? throw NotNative();
    internal IndexTtsEcapa<T> SpeakerEncoder => _speakerEncoder ?? throw NotNative();
    internal IndexTtsDvae<T> Dvae => _dvae ?? throw NotNative();
    internal BigVganGenerator<T> Vocoder => _vocoder ?? throw NotNative();

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
        _gpt = new IndexTts1UnifiedVoice<T>(Engine, _gptLayers, VoiceConfiguration(o), random);
        _speakerEncoder = new IndexTtsEcapa<T>(Engine, _speakerEncoderLayers, new IndexTtsEcapaConfiguration(o.MelChannels, o.SpeakerEmbeddingDim,
            o.SpeakerEncoderChannels, o.SpeakerEncoderKernels, o.SpeakerEncoderDilations, o.SpeakerEncoderAttentionChannels,
            o.SpeakerEncoderSeChannels, o.SpeakerEncoderScale), random);
        _dvae = new IndexTtsDvae<T>(Engine, _dvaeLayers, new IndexTtsDvaeConfiguration(o.MelChannels, o.DvaeHidden, o.DvaeLayers, o.DvaeResBlocks,
            o.DvaeCodebookDim, o.MelCodes - 2, o.DvaeKernel), random);
        _vocoder = new BigVganGenerator<T>(Engine, random, o.HiddenDim, o.VocoderInitialChannels, o.VocoderUpsampleRates, o.VocoderUpsampleKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, snakeBeta: true, tanhAtFinal: true, biasAtFinal: true,
            speakerChannels: o.SpeakerEmbeddingDim);
        // The GPT trains; the DVAE (the codes' tokenizer), the speaker encoder and BigVGAN are pretrained and frozen.
        AddUnsplitLayers(_gptLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_speakerEncoderLayers);
        ComponentLayers.AddRange(_dvaeLayers);
        ComponentLayers.AddRange(_vocoder.Layers);
    }

    internal static IndexTts1VoiceConfiguration VoiceConfiguration(IndexTTSOptions o) => new(
        o.HiddenDim, o.NumHeads, o.Layers, o.MaxTextLength, o.MaxMelTokens, o.VocabSize, o.MelCodes, 0, 1, o.MelCodes - 2, o.MelCodes - 1,
        o.ConditionLatents, new IndexTtsConformerConfiguration(o.MelChannels, o.ConditionDim, o.ConditionLinearUnits, o.ConditionHeads, o.ConditionBlocks),
        o.PerceiverMultiple, o.DropoutRate);

    // ---------------------------------------------------------------- text

    /// <summary>The release's tokenizer (normalizer, CJK splitting, <c>bpe.model</c>).</summary>
    internal IndexTtsTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = _options.BpeModelPath
                ?? throw new InvalidOperationException("Text needs the release's tokenizer; set IndexTTSOptions.BpeModelPath to bpe.model.");
            return _tokenizer = new IndexTtsTokenizer(new SentencePieceModel(File.ReadAllBytes(path)), new IndexTtsTextNormalizer());
        }
    }

    /// <summary>The token ids of <paramref name="text"/> (normalized, CJK-split, BPE).</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Tokenizer.ToIds(Tokenizer.Tokenize(text));
    }

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
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

    /// <summary>The text's segments (<c>split_segments</c> at <see cref="IndexTTSOptions.SegmentTokens"/>) when the
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

    // ---------------------------------------------------------------- audio features

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <summary>The 100-band log mel <c>[frames, mel]</c> of audio at the model's rate (the release's
    /// <c>MelSpectrogramFeatures</c>: torchaudio's centred magnitude mel, HTK scale, <c>log(max(x, 1e-7))</c>).</summary>
    internal Tensor<T> MelSpectrogram(Tensor<T> audio)
    {
        var o = _options;
        _mel ??= new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0.0, o.SampleRate / 2.0, 1e-7,
            htkScale: true, naturalLog: true);
        var mel = _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        return Engine.TensorTranspose(Engine.Reshape(mel, new[] { o.MelChannels, mel.Shape[2] }));
    }

    /// <summary>The DVAE's codes of a recording at the model's rate (<c>get_codebook_indices</c> of its log mel).</summary>
    public int[] EncodeToCodes(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (!HasPaperLayers) throw NotNative();
        using var _ = new NoGradScope<T>();
        return Dvae.Codes(MelSpectrogram(audio));
    }

    /// <summary>The ECAPA-TDNN speaker embedding <c>[SpeakerEmbeddingDim]</c> of a prompt's log mel.</summary>
    internal Tensor<T> SpeakerEmbedding(Tensor<T> mel)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
        return SpeakerEncoder.Forward(mel);
    }

    // ---------------------------------------------------------------- the prompt

    private sealed class PromptInfo
    {
        public Tensor<T> Mel = null!;           // [frames, mel] at the model's rate
        public Tensor<T> Speaker = null!;       // ECAPA-TDNN [SpeakerEmbeddingDim]
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/> (the release's prompt handling: the
    /// channels averaged, resampled by torchaudio to 24 kHz, its log mel conditioning the GPT and BigVGAN).
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var audio = sampleRate == _options.SampleRate ? Wave(Doubles(recording))
            : AudioHelper<T>.ResampleBandLimited(Engine, Wave(Doubles(recording)), sampleRate, _options.SampleRate);
        var voice = new TtsVoice<T> { ReferenceAudio = audio };
        _prompts.Add(voice, PromptOf(audio));
        return voice;
    }

    private PromptInfo PromptOf(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        var mel = MelSpectrogram(audio);
        return new PromptInfo { Mel = mel, Speaker = SpeakerEmbedding(mel) };
    }

    private PromptInfo Prompt(TtsVoice<T> voice)
    {
        if (_prompts.TryGetValue(voice, out var info)) return info;
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        info = PromptOf(audio);
        _prompts.Add(voice, info);
        return info;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    /// <summary>
    /// The release's <c>remove_long_silence</c>: when the codes hold more than <see cref="IndexTTSOptions.MaxSilentCodes"/>
    /// silent codes, every run of them is cut to its first <see cref="IndexTTSOptions.KeptSilentCodes"/>.
    /// </summary>
    internal int[] RemoveLongSilence(IReadOnlyList<int> codes)
    {
        var o = _options;
        if (codes.Count(c => c == o.SilentCode) <= o.MaxSilentCodes) return codes.ToArray();
        var kept = new List<int>(codes.Count);
        int run = 0;
        foreach (int code in codes)
        {
            if (code != o.SilentCode)
            {
                kept.Add(code);
                run = 0;
            }
            else if (run < o.KeptSilentCodes)
            {
                kept.Add(code);
                run++;
            }
        }
        return kept.ToArray();
    }

    /// <summary>The release's <c>fade_out_tail</c>: the last 20 ms scaled by a raised cosine from 1 to 0.</summary>
    internal static double[] FadeOutTail(double[] wave, int sampleRate)
    {
        var result = (double[])wave.Clone();
        int n = Math.Min((int)(sampleRate * 20.0 / 1000.0), result.Length);
        if (n <= 1) return result;
        for (int i = 0; i < n; i++)
            result[result.Length - n + i] *= 0.5 * (1 + Math.Cos(Math.PI * i / (n - 1)));
        return result;
    }

    /// <summary>One segment's audio: BigVGAN of the GPT's latents of the codes, with the prompt's speaker embedding.</summary>
    internal double[] Decode(Tensor<T> conditions, IReadOnlyList<int> text, IReadOnlyList<int> codes, Tensor<T> speaker)
    {
        int d = _options.HiddenDim;
        var latents = TextToMel.Latents(conditions, text, codes);                                    // [n, d]
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(latents), new[] { 1, d, codes.Count }), speaker);
        return Doubles(wave).Select(v => Math.Max(-1.0, Math.Min(1.0, v))).ToArray();
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is the speech at 24 kHz in the voice's
    /// timbre (the release's <c>infer</c>: per segment, mel codes by beam search, long silences shrunk, BigVGAN of their
    /// latents; the segments joined and the end faded).</remarks>
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
            var gpt = TextToMel;
            var conditions = gpt.SpeakerConditioning(prompt.Mel);
            var decoding = new IndexTtsDecoding(o.Sample, o.Beams, o.TopK, o.TopP, o.Temperature, o.RepetitionPenalty, o.MaxGeneratedTokens);
            var wave = new List<double>();
            foreach (var segment in Segments(ids))
            {
                var codes = RemoveLongSilence(gpt.Generate(conditions, segment, decoding, random));
                if (codes.Length == 0) continue;
                wave.AddRange(Decode(conditions, segment, codes, prompt.Speaker));
            }
            return Wave(FadeOutTail(wave.ToArray(), o.SampleRate));
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
            Name = _useNativeMode ? "IndexTTS-Native" : "IndexTTS-ONNX",
            Description = "IndexTTS: industrial zero-shot TTS, a GPT over DVAE mel codes and a speaker-conditioned BigVGAN (Deng et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "IndexTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IndexTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
