using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// NaturalSpeech 3 (Ju et al. 2024): zero-shot speech synthesis with a factorized codec and factorized diffusion. FACodec
/// splits speech into prosody, content and acoustic-detail codes and a timbre vector; masked discrete diffusions generate
/// the durations, prosody, content and detail of new text, each following the corresponding part of a speech prompt; the
/// codec decodes them in the prompt's timbre.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Inference: the prompt's codes and timbre from FACodec; the prompt's and the text's phonemes through the phoneme
/// encoder; durations by diffusion after the prompt's (no guidance); then prosody, content and detail by diffusion after the
/// prompt's, each with classifier-free guidance; FACodec's decoder on their embeddings with the prompt's timbre.</para>
/// <para>Training (<see cref="NaturalSpeech3Options.CurrentStage"/>): FACodec with its reconstruction, adversarial,
/// codebook, supervision and gradient-reversal losses; the voice-conversion redecoder; the diffusion model on a frozen codec.
/// FACodec follows the released code and weights; the diffusion model follows the paper, with MaskGCT's conventions where
/// the paper is silent.</para>
/// <para><b>For Beginners:</b> give it a few seconds of someone speaking, with what they said, and new text; it says the
/// text in that voice and speaking style. <see cref="ConvertVoice"/> re-speaks one recording in another's voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new NaturalSpeech3&lt;float&gt;(architecture, new NaturalSpeech3Options());
/// model.LoadFACodec("ns3_facodec_encoder.bin", "ns3_facodec_decoder.bin");
/// model.Voice = model.CreateVoice(prompt, model.TextToPhonemes("what the prompt says"), promptDurations);   // 16 kHz
/// var audio = model.Synthesize("The quick brown fox.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models",
    "https://arxiv.org/abs/2403.03100",
    Year = 2024,
    Authors = "Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, Zhizheng Wu, Tao Qin, Xiang-Yang Li, Wei Ye, Shikun Zhang, Jiang Bian, Lei He, Jinyu Li, Sheng Zhao"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.98,
                WarmupSteps = 5000, Schedule = LearningRateSchedulerType.Noam,
                Source = "Ju et al. 2024, App. A.2: AdamW with a learning rate of 1e-4, beta1 0.9 and beta2 0.98, and 5K warmup "
                        + "steps following the inverse square root schedule, declared here as Noam. FACodec trains separately "
                        + "with Adam at 2e-4, betas 0.5/0.9 (App. B.1).")]
public partial class NaturalSpeech3<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    /// <summary>Calls the fully specified overload with the omitted parameters' defaults.</summary>
    public NaturalSpeech3(NeuralNetworkArchitecture<T> architecture)
        : this(architecture, (NaturalSpeech3Options?)null)
    {
    }


    private readonly NaturalSpeech3Options _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private FACodecModel<T>? _codec;
    private Ns3Core<T>? _core;

    private readonly List<LayerBase<T>> _codecEncoderLayers = new();
    private readonly List<LayerBase<T>> _codecDecoderLayers = new();
    private readonly List<LayerBase<T>> _codecHeadLayers = new();
    private readonly List<LayerBase<T>> _redecoderLayers = new();
    private readonly List<LayerBase<T>> _diffusionLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed NaturalSpeech 3 for inference.</summary>
    public NaturalSpeech3(NeuralNetworkArchitecture<T> architecture, string modelPath, NaturalSpeech3Options? options = null)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeech3Options();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable NaturalSpeech 3.</summary>
    public NaturalSpeech3(NeuralNetworkArchitecture<T> architecture, NaturalSpeech3Options? options)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeech3Options();
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
        if (o.CodecEncoderRates.Aggregate(1, (a, b) => a * b) != o.HopSize || o.CodecDecoderRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException($"FACodec's encoder and decoder strides must each multiply to HopSize {o.HopSize}.");
        if (o.HiddenDim % o.NumHeads != 0 || o.DiffusionDim % o.DiffusionHeads != 0) throw new ArgumentException("Every width must divide by its heads.");
        if (o.DiffusionDim % 2 != 0 || o.DiffusionDim < 4) throw new ArgumentException("DiffusionDim must be even and at least 4.");
        if (o.SamplingSteps < 1 || o.TopK < 1) throw new ArgumentException("SamplingSteps and TopK must be positive.");
        // A non-positive warm-up would silently train on another schedule.
        if (o.WarmupSteps <= 0) throw new ArgumentOutOfRangeException("options", o.WarmupSteps, "WarmupSteps must be positive.");
        if (o.DetailDropout is < 0 or > 1 || o.PromptDropProbability is < 0 or > 1) throw new ArgumentException("Probabilities lie in [0, 1].");
        if ((double.IsNaN(o.CodecBeta1) || double.IsInfinity(o.CodecBeta1)) || o.CodecBeta1 is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(o.CodecBeta1), "CodecBeta1 must be finite and lie in [0, 1).");
        if ((double.IsNaN(o.CodecBeta2) || double.IsInfinity(o.CodecBeta2)) || o.CodecBeta2 is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(o.CodecBeta2), "CodecBeta2 must be finite and lie in [0, 1).");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The iterations of each diffusion.</remarks>
    public int NumFlowSteps => _options.SamplingSteps;

    private bool HasPaperLayers => _core is not null;

    // The training and sampling seed (Options.Seed, 1234 when unset).
    private int SeedValue => _options.Seed ?? 1234;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal FACodecModel<T> Codec => _codec ?? throw NotNative();
    internal Ns3Core<T> Core => _core ?? throw NotNative();

    internal FACodecConfiguration CodecConfiguration()
    {
        var o = _options;
        return new FACodecConfiguration
        {
            SampleRate = o.SampleRate,
            EncoderDim = o.CodecEncoderDim,
            EncoderRates = o.CodecEncoderRates,
            Dim = o.CodecDim,
            DecoderInitial = o.CodecDecoderChannels,
            DecoderRates = o.CodecDecoderRates,
            ProsodyQuantizers = o.ProsodyQuantizers,
            ContentQuantizers = o.ContentQuantizers,
            DetailQuantizers = o.DetailQuantizers,
            CodebookDim = o.CodebookDim,
            CodebookSize = o.CodebookSize,
            Phones = o.CodecPhones,
            Speakers = o.CodecSpeakers,
            DetailF0Reversal = o.DetailF0Reversal,
            DetailPhoneReversal = o.DetailPhoneReversal,
            TimbreReversal = o.TimbreReversal,
            ContentF0Reversal = o.ContentF0Reversal,
            ProsodyPhoneReversal = o.ProsodyPhoneReversal,
            MelProsody = o.MelProsody,
            Redecoder = o.IncludeRedecoder,
        };
    }

    private Ns3Configuration Configuration()
    {
        var o = _options;
        return new Ns3Configuration
        {
            Vocab = o.VocabSize,
            PhonemeDim = o.HiddenDim,
            PhonemeHeads = o.NumHeads,
            PhonemeLayers = o.NumEncoderLayers,
            PhonemeFilter = o.FilterChannels,
            PhonemeKernel = o.EncoderKernel,
            Dropout = o.DropoutRate,
            Hidden = o.DiffusionDim,
            Heads = o.DiffusionHeads,
            Filter = o.DiffusionFilter,
            Kernel = o.DiffusionKernel,
            FrameLayers = o.FrameDiffusionLayers,
            PhonemeDiffusionLayers = o.PhonemeDiffusionLayers,
            MaxDuration = o.MaxDuration,
            ProsodyCodebooks = o.ProsodyQuantizers,
            ContentCodebooks = o.ContentQuantizers,
            DetailCodebooks = o.DetailQuantizers,
            CodebookSize = o.CodebookSize,
            PhonemeProsody = o.PhonemeProsody,
        };
    }

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        _codec = new FACodecModel<T>(Engine, CodecConfiguration(), random, _codecEncoderLayers, _codecDecoderLayers, _codecHeadLayers, _redecoderLayers);
        _core = new Ns3Core<T>(Engine, Configuration(), random, _diffusionLayers);
        BuildDiscriminators();
        AddUnsplitLayers(_diffusionLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecEncoderLayers);
        ComponentLayers.AddRange(_codecDecoderLayers);
        ComponentLayers.AddRange(_codecHeadLayers);
        ComponentLayers.AddRange(_redecoderLayers);
        ComponentLayers.AddRange(DiscriminatorLayers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The text's phone ids (ARPAbet; the front end NaturalSpeech 2 uses, see <see cref="NaturalSpeech2{T}.TextToPhonemes"/>).</summary>
    public int[] TextToPhonemes(string text) => NaturalSpeechArpabet.Phonemes(text, _options.LexiconPath);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return IdTensor(TextToPhonemes(text));
    }

    private static double[] Doubles(Tensor<T> x) => x.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(IReadOnlyList<double> samples) =>
        new(new[] { samples.Count }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    private int[] PhonemeIds(Tensor<T> input, string name)
    {
        var ids = RoundedIds(input);
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(name, $"Phoneme ids lie in [0, {_options.VocabSize}).");
        }
        if (ids.Length == 0) throw new ArgumentException("There are no phonemes.", name);
        return ids;
    }

    // ---------------------------------------------------------------- FACodec

    private Tensor<T> AsWave(Tensor<T> audio) => Engine.Reshape(audio, new[] { 1, 1, audio.Length });

    /// <summary>FACodec's factors of a recording at <see cref="TtsModelOptions.SampleRate"/> (inference: no gradients).</summary>
    internal FACodecFactors<T> Factorize(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        var wave = AsWave(audio);
        var prosody = _options.MelProsody ? Codec.ProsodyFeature(Doubles(audio)) : null;
        var latent = Codec.Encoder.Forward(wave);
        foreach (var q in Codec.Quantizers) q.SetTrainingMode(false);
        return Codec.Factorize(latent, prosody, false, null);
    }

    /// <summary>
    /// FACodec's codes of a recording at <see cref="TtsModelOptions.SampleRate"/>: <c>[codebooks, frames]</c>, prosody first,
    /// then content, then acoustic detail; and its timbre vector.
    /// </summary>
    public (int[,] Codes, Tensor<T> Timbre) EncodeFactors(Tensor<T> recording)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var f = Factorize(recording);
        return (f.Codes, f.Speaker);
    }

    /// <summary>FACodec's waveform of codes <c>[codebooks, frames]</c> in a timbre (the decoder's <c>inference</c>).</summary>
    public Tensor<T> DecodeFactors(int[,] codes, Tensor<T> timbre, bool includeDetail = true)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        var wave = Codec.Decode(Codec.Embed(codes, includeDetail), Engine.Reshape(timbre, new[] { 1, _options.CodecDim }));
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    /// <summary>
    /// Zero-shot voice conversion: the source's prosody and content in the target speaker's timbre, through the redecoder
    /// (<see cref="NaturalSpeech3Options.IncludeRedecoder"/>) or, without one, FACodec's decoder; the detail codes are left
    /// out, as the release's examples do.
    /// </summary>
    public Tensor<T> ConvertVoice(Tensor<T> source, Tensor<T> target)
    {
        ThrowIfDisposed();
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (target is null) throw new ArgumentNullException(nameof(target));
        var a = Factorize(source);
        var b = Factorize(target);
        using var _ = new NoGradScope<T>();
        var wave = Codec.RedecoderEmbeddings is not null
            ? Codec.Redecode(Codec.RedecoderEmbed(a.Codes, b.Speaker, detail: false), b.Speaker)
            : Codec.Decode(Codec.Embed(a.Codes, detail: false), b.Speaker);
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Synthesis follows a prompt recording with its phones and their durations (<see cref="CreateVoice"/>). A
    /// recording alone also works: the durations then follow no prompt, and the prompt's frames carry no phone condition.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <summary>
    /// A voice from a prompt recording at <see cref="TtsModelOptions.SampleRate"/>, its phone ids and each phone's duration
    /// in codec frames (the paper's prompts carry the alignment its duration diffusion follows).
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, IReadOnlyList<int> phonemes, IReadOnlyList<int> durations)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (phonemes is null || durations is null) throw new ArgumentNullException(phonemes is null ? nameof(phonemes) : nameof(durations));
        if (phonemes.Count != durations.Count || phonemes.Count == 0) throw new ArgumentException("Give one duration per phone, and at least one phone.");
        return new TtsVoice<T>
        {
            ReferenceAudio = Engine.Reshape(recording, new[] { recording.Length }),
            ReferenceTokens = IdTensor(phonemes),
            ReferenceDurations = durations.ToArray(),
        };
    }

    // The prompt's phones and durations (empty without a transcript), its codes [frames, codebooks] and timbre.
    private (int[] Phonemes, int[] Durations, int[,] Codes, Tensor<T> Timbre) Prompt(TtsVoice<T> voice)
    {
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} follows a prompt recording; set ReferenceAudio.");
        var f = Factorize(audio);
        int frames = f.Codes.GetLength(1);
        if (voice.ReferenceTokens is null || voice.ReferenceDurations is null)
            return (Array.Empty<int>(), Array.Empty<int>(), Transposed(f.Codes, frames), f.Speaker);
        var phonemes = PhonemeIds(voice.ReferenceTokens, nameof(voice));
        var durations = (int[])voice.ReferenceDurations.Clone();
        if (durations.Length != phonemes.Length) throw new ArgumentException("The prompt needs one duration per phone.", nameof(voice));
        Align(durations, frames);
        return (phonemes, durations, Transposed(f.Codes, durations.Sum()), f.Speaker);
    }

    // The durations cut to a frame count, the excess taken from the last ones.
    private static void Align(int[] durations, int frames)
    {
        for (int excess = durations.Sum() - frames, j = durations.Length - 1; excess > 0 && j >= 0; j--)
        {
            int cut = Math.Min(excess, durations[j]);
            durations[j] -= cut;
            excess -= cut;
        }
    }

    // [codebooks, frames] → [frames, codebooks], the first `frames`.
    private static int[,] Transposed(int[,] codes, int frames)
    {
        int q = codes.GetLength(0);
        var t = new int[frames, q];
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < q; k++) t[f, k] = codes[k, f];
        return t;
    }

    private static int[,] Columns(int[,] codes, int first, int count)
    {
        var c = new int[codes.GetLength(0), count];
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int k = 0; k < count; k++) c[f, k] = codes[f, first + k];
        return c;
    }

    private static int[,] Stack(int[,] top, int[,] bottom)
    {
        int q = top.GetLength(1);
        var s = new int[top.GetLength(0) + bottom.GetLength(0), q];
        for (int f = 0; f < top.GetLength(0); f++)
            for (int k = 0; k < q; k++) s[f, k] = top[f, k];
        for (int f = 0; f < bottom.GetLength(0); f++)
            for (int k = 0; k < q; k++) s[top.GetLength(0) + f, k] = bottom[f, k];
        return s;
    }

    private MaskGctDecoding Decoding(double guidance, int vocabulary) => new()
    {
        Steps = _options.SamplingSteps,
        Temperature = _options.Temperature,
        FilterThreshold = 1.0 - Math.Min(_options.TopK, vocabulary) / (double)vocabulary,
        Guidance = guidance,
        GuidanceRescale = 1.0,
    };

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's phone ids (<see cref="TextToPhonemes"/>); the result is the waveform at
    /// <see cref="TtsModelOptions.SampleRate"/> in <see cref="TtsModelBase{T}.Voice"/>'s voice and style.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var phonemes = PhonemeIds(input, nameof(input));
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var prompt = Prompt(voice);
            using var _ = new NoGradScope<T>();
            var core = Core;
            var o = _options;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
            double[] Uniforms(int n) => Enumerable.Range(0, n).Select(_ => random.NextDouble()).ToArray();
            var all = prompt.Phonemes.Concat(phonemes).ToArray();
            int lp = prompt.Phonemes.Length, promptFrames = prompt.Codes.GetLength(0);
            var cph = core.Phonemes(all, false, null);

            int[]? phonemeProsody = null;
            if (core.PhonemeProsody is not null)
            {
                var promptProsody = Column(PhonemeProsodyCodes(prompt.Codes, prompt.Durations));
                var generated = core.PhonemeProsody.Generate(cph, promptProsody, Decoding(o.GuidanceScale, o.CodebookSize), Uniforms);
                phonemeProsody = Flatten(promptProsody).Concat(Flatten(generated)).ToArray();
            }
            var phonemeCondition = core.WithPhonemeProsody(cph, phonemeProsody);
            var promptDurations = Column(prompt.Durations.Select(d => Math.Min(d, o.MaxDuration)).ToArray());
            var durations = Flatten(core.Duration.Generate(phonemeCondition, promptDurations, Decoding(0, o.MaxDuration + 1), Uniforms));
            if (durations.Sum() == 0) for (int i = 0; i < durations.Length; i++) durations[i] = 1;
            var allDurations = prompt.Durations.Concat(durations).ToArray();

            // A prompt without its transcript has no phone condition on its frames.
            Tensor<T> Frames(Tensor<T> rows)
            {
                var expanded = core.Expand(rows, allDurations);
                return lp > 0 || promptFrames == 0 ? expanded
                    : Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { promptFrames, o.DiffusionDim }), expanded }, 0);
            }
            int np = o.ProsodyQuantizers, nc = o.ContentQuantizers, nd = o.DetailQuantizers;
            var prosody = core.Prosody.Generate(Frames(phonemeCondition), Columns(prompt.Codes, 0, np), Decoding(o.GuidanceScale, o.CodebookSize), Uniforms);
            var withProsody = Stack(Columns(prompt.Codes, 0, np), prosody);
            var contentCondition = core.Plus(Frames(cph), withProsody, 0, core.ProsodyCondition);
            var content = core.Content.Generate(contentCondition, Columns(prompt.Codes, np, nc), Decoding(o.GuidanceScale, o.CodebookSize), Uniforms);
            var withContent = Stack(Columns(prompt.Codes, np, nc), content);
            var detailCondition = core.Plus(contentCondition, withContent, 0, core.ContentCondition);
            var detail = core.Detail.Generate(detailCondition, Columns(prompt.Codes, np + nc, nd), Decoding(o.GuidanceScale, o.CodebookSize), Uniforms);

            int n = prosody.GetLength(0);
            var codes = new int[np + nc + nd, n];
            for (int f = 0; f < n; f++)
            {
                for (int k = 0; k < np; k++) codes[k, f] = prosody[f, k];
                for (int k = 0; k < nc; k++) codes[np + k, f] = content[f, k];
                for (int k = 0; k < nd; k++) codes[np + nc + k, f] = detail[f, k];
            }
            var wave = Codec.Decode(Codec.Embed(codes), prompt.Timbre);
            return Engine.Reshape(wave, new[] { wave.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static int[,] Column(int[] values)
    {
        var c = new int[values.Length, 1];
        for (int i = 0; i < values.Length; i++) c[i, 0] = values[i];
        return c;
    }

    private static int[] Flatten(int[,] column) => Enumerable.Range(0, column.GetLength(0)).Select(i => column[i, 0]).ToArray();

    /// <summary>Each phoneme's most frequent prosody code (the first prosody codebook) over its frames; 0 for a phoneme
    /// without frames. The paper names a phoneme-level prosody but never defines it; this is the port's definition.</summary>
    internal static int[] PhonemeProsodyCodes(int[,] codes, IReadOnlyList<int> durations)
    {
        var result = new int[durations.Count];
        int frame = 0;
        for (int j = 0; j < durations.Count; j++)
        {
            var counts = new Dictionary<int, int>();
            for (int r = 0; r < durations[j]; r++, frame++)
            {
                int c = codes[frame, 0];
                counts[c] = counts.TryGetValue(c, out var k) ? k + 1 : 1;
            }
            result[j] = counts.Count == 0 ? 0 : counts.OrderByDescending(p => p.Value).ThenBy(p => p.Key).First().Key;
        }
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
            Name = _useNativeMode ? "NaturalSpeech3-Native" : "NaturalSpeech3-ONNX",
            Description = "NaturalSpeech 3: FACodec attribute factorization and factorized masked discrete diffusion (Ju et al., 2024)",
            FeatureCount = _options.DiffusionDim,
        };
        metadata.AdditionalInfo["Architecture"] = "NaturalSpeech3";
        metadata.AdditionalInfo["Stage"] = _options.CurrentStage.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech3<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
