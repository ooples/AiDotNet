using AiDotNet.Attributes;
using AiDotNet.Audio.Effects;
using AiDotNet.Audio.Resampling;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// MegaTTS 3 (Jiang et al. 2025): zero-shot speech from a sparse-alignment latent diffusion transformer. A prompt recording
/// is aligned by a Whisper aligner into phones, tones and durations and encoded by the WaveVAE; the text is normalized,
/// chunked and turned into phones by a Qwen2 grapheme-to-phoneme model; an autoregressive duration model continues the
/// prompt's durations; the diffusion transformer generates the WaveVAE latent after the prompt with multi-condition
/// classifier-free guidance; the WaveVAE decodes 24 kHz audio, matched to the prompt's loudness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Inference is the release's (bytedance/MegaTTS3, <c>tts/infer_cli.py</c> and <c>tts/frontend_function.py</c>), with
/// WeTextProcessing's normalizers, langdetect, librosa's soxr resampling and pyloudnorm ported; training is the paper's,
/// which the release does not include: rectified flow on the latent's masked target region, sparse alignment anchors, and
/// the prompt and text dropped for classifier-free guidance.</para>
/// <para><b>For Beginners:</b> give MegaTTS 3 a few seconds of someone speaking (<see cref="TtsModelBase{T}.Voice"/>'s
/// reference recording) and a sentence, and it says the sentence in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new MegaTTS3&lt;float&gt;(architecture, new MegaTTS3Options());
/// model.LoadReferenceWeights("checkpoints");   // ByteDance/MegaTTS3: diffusion_transformer, duration_lm, aligner_lm, g2p, wavvae
/// model.PromptLatent = MegaTTS3&lt;float&gt;.LoadLatent("prompt.npy");   // the release withholds the WaveVAE encoder
/// model.Voice = new TtsVoice&lt;float&gt; { ReferenceAudio = prompt24kHz };
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MegaTTS 3: Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis",
    "https://arxiv.org/abs/2502.18924",
    Year = 2025,
    Authors = "Jiang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, WarmupSteps = 10000,
    Source = "Jiang et al. 2025, Appendix A.1: MegaTTS 3 is trained with the Adam optimizer at a learning rate of 5e-5, "
             + "beta1 0.9, beta2 0.999 and 10K warmup steps, for 1M steps. No reference batch size is declared because the "
             + "paper gives the batch as 10K latent frames per GPU rather than as a count of examples.")]
public partial class MegaTTS3<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MegaTTS3Options _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private MegaTts3Dit<T>? _dit;
    private MegaTts3Duration<T>? _duration;
    private MegaTts3Aligner<T>? _aligner;
    private MegaTts3G2p<T>? _g2p;
    private MegaTts3WaveVae<T>? _vae;

    private readonly List<LayerBase<T>> _ditLayers = new();
    private readonly List<LayerBase<T>> _durationLayers = new();
    private readonly List<LayerBase<T>> _alignerLayers = new();
    private readonly List<LayerBase<T>> _g2pLayers = new();
    private readonly List<LayerBase<T>> _vaeEncoderLayers = new();
    private readonly List<LayerBase<T>> _vaeDecoderLayers = new();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed MegaTTS 3 for inference.</summary>
    public MegaTTS3(NeuralNetworkArchitecture<T> architecture, string modelPath, MegaTTS3Options? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTS3Options();
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable MegaTTS 3.</summary>
    public MegaTTS3(NeuralNetworkArchitecture<T> architecture, MegaTTS3Options? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTS3Options();
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.VaeEncoderRatios.Aggregate(1, (a, b) => a * b) != o.HopSize * FrameMultiple / 2)
            throw new ArgumentException("The WaveVAE's frame (the product of its encoder ratios) must span four mel frames of HopSize samples.");
        base.SampleRate = o.SampleRate;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
    }

    // The release's frame bookkeeping: durations in 100 Hz mel frames (HopSize at 24 kHz, Whisper's hop at 16 kHz), cut to
    // multiples of 8; the latent at a quarter of that rate.
    internal const int FrameMultiple = 8, VaeStride = 4;

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int NumFlowSteps => _options.Steps;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>The prompt's WaveVAE latent <c>[frames, channels]</c> to use instead of encoding the voice's recording (the
    /// release's <c>.npy</c> latents, for its decoder-only checkpoint); null encodes the recording.</summary>
    public Tensor<T>? PromptLatent { get; set; }

    /// <summary>Reads a prompt latent saved by the release (a <c>.npy</c> of <c>[1, frames, channels]</c> or
    /// <c>[frames, channels]</c>).</summary>
    public static Tensor<T> LoadLatent(string path)
    {
        var loaded = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        if (loaded.Count != 1) throw new InvalidDataException("A latent file holds one array.");
        var array = loaded.Values.First();
        var values = array.ToVector().Select(v => MathHelper.GetNumericOperations<T>().FromDouble(v)).ToArray();
        int channels = array.Shape[array.Shape.Length - 1];
        return new Tensor<T>(new[] { values.Length / channels, channels }, new Vector<T>(values));
    }

    /// <inheritdoc />
    /// <remarks>MegaTTS 3 is zero-shot: it reads the voice's recording (24 kHz) as its prompt.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    /// <remarks>Training reads the recording (24 kHz), which the aligner turns into phones, tones and durations.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _dit is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal MegaTts3Dit<T> Dit => _dit ?? throw NotNative();
    internal MegaTts3Duration<T> Duration => _duration ?? throw NotNative();
    internal MegaTts3Aligner<T> Aligner => _aligner ?? throw NotNative();
    internal MegaTts3G2p<T> G2p => _g2p ?? throw NotNative();
    internal MegaTts3WaveVae<T> Vae => _vae ?? throw NotNative();

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.G2pVocabSize) : base.GetInputDomain(inputShape);

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
        _dit = new MegaTts3Dit<T>(Engine, _ditLayers, new MegaTts3DitConfiguration
        {
            Width = o.HiddenDim, Layers = o.DitLayers, Heads = o.DitHeads, LatentChannels = o.LatentChannels, LocalConditionDim = o.LocalConditionDim,
            ContextMaskDim = o.ContextMaskDim, PhoneVocabulary = o.DitPhoneVocabulary, ToneVocabulary = o.DitToneVocabulary,
            PhoneEncoderLayers = o.PhoneEncoderLayers, PhoneEncoderHeads = o.PhoneEncoderHeads,
        });
        _duration = new MegaTts3Duration<T>(Engine, _durationLayers, new MegaTts3DurationConfiguration
        {
            PhoneVocabulary = o.DurationPhoneVocabulary, EncoderWidth = o.DurationEncoderWidth, EncoderFilter = o.DurationEncoderFilter,
            EncoderHeads = o.DurationEncoderHeads, EncoderLayers = o.DurationEncoderLayers, ToneVocabulary = o.DurationToneVocabulary,
            DecoderWidth = o.DurationWidth, DecoderLayers = o.DurationLayers, DecoderHeads = o.DurationHeads, CodeSize = o.DurationCodeSize,
        });
        _aligner = new MegaTts3Aligner<T>(Engine, _alignerLayers, new WhisperDimensions(80, 3000, o.AlignerWidth, o.AlignerHeads, o.AlignerLayers,
            o.AlignerVocabulary, 2048, o.AlignerWidth, o.AlignerHeads, o.AlignerLayers) { UntiedOutput = true, VariableAudioLength = true })
        { MaxTokens = o.AlignerMaxTokens };
        var g2pConfiguration = new LlamaConfiguration
        {
            VocabSize = o.G2pVocabSize, Hidden = o.G2pHidden, Intermediate = o.G2pIntermediate, Layers = o.G2pLayers, Heads = o.G2pHeads,
            KeyValueHeads = o.G2pKeyValueHeads, RmsEpsilon = 1e-6, RopeTheta = o.G2pRopeTheta, AttentionBias = true, TieWordEmbeddings = true,
        };
        _g2p = o.G2pTokenizerJson is null
            ? new MegaTts3G2p<T>(Engine, _g2pLayers, g2pConfiguration, null, 0)
            : MegaTts3G2p<T>.FromTokenizer(Engine, _g2pLayers, g2pConfiguration, o.G2pTokenizerJson);
        _g2p.MaxNewTokens = o.G2pMaxNewTokens;
        _vae = new MegaTts3WaveVae<T>(Engine, _vaeEncoderLayers, _vaeDecoderLayers, new MegaTts3WaveVaeConfiguration
        {
            EncoderRatios = o.VaeEncoderRatios, EncoderFilters = o.VaeEncoderFilters, EncoderDimension = o.VaeEncoderDimension,
            LatentChannels = o.LatentChannels, UpsamplerChannels = o.VaeUpsamplerChannels, LatentUpsampling = o.VaeLatentUpsampling,
            GeneratorFilters = o.VaeGeneratorFilters, GeneratorRatios = o.VaeGeneratorRatios,
        });
        AddUnsplitLayers(_ditLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_durationLayers);
        ComponentLayers.AddRange(_alignerLayers);
        ComponentLayers.AddRange(_g2pLayers);
        ComponentLayers.AddRange(_vaeEncoderLayers);
        ComponentLayers.AddRange(_vaeDecoderLayers);
    }

    // ---------------------------------------------------------------- text

    /// <inheritdoc />
    /// <remarks>The text's ids in the g2p model's tokenizer (synthesis reads the text back from them).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private ByteLevelBpeTokenizer Tokenizer => G2p.Tokenizer
        ?? throw new InvalidOperationException("The g2p tokenizer comes with the checkpoint: call LoadReferenceWeights first, or set G2pTokenizerJson.");

    private string TextOf(Tensor<T> ids) => Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))));

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks><c>MegaTTS3DiTInfer.forward</c> after <c>preprocess</c> of the voice's recording: the text normalized by the
    /// language langdetect gives (English, or else Chinese), chunked, and each chunk synthesized and joined.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Wave(SynthesizeText(text));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's g2p token ids (<see cref="PreprocessText"/>); the speech of that text
    /// in the voice's recording.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.G2pVocabSize))
            throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.G2pVocabSize}).");
        return Wave(SynthesizeText(TextOf(input)));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    private double[] SynthesizeText(string text)
    {
        var o = _options;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var voice = RequireVoice();
            var recording = voice.ReferenceAudio!;
            if (!_prompts.TryGetValue(recording, out var held) || !ReferenceEquals(held.Latent, PromptLatent))
            {
                if (_prompts.Count > 8) _prompts.Clear();
                _prompts[recording] = held = (PromptLatent, Prepare(Doubles(recording)));
            }
            var prompt = held.Prompt;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            bool english = new LangDetector { Seed = o.SamplingSeed }.Detect(text) == "en";
            var segments = english
                ? MegaTts3Text.ChunkEnglish(WeTextNormalizer.English.Normalize(text), o.EnglishChunkBytes)
                : MegaTts3Text.ChunkChinese(WeTextNormalizer.Chinese.Normalize(text), o.ChineseChunkLimit);
            if (segments.Count == 0) throw new ArgumentException("The text has nothing to say.", nameof(text));
            var waves = new List<double[]>();
            for (int i = 0; i < segments.Count; i++)
            {
                var (phones, tones) = G2p.Convert(segments[i]);
                var mel2ph = PredictMel2Ph(prompt, phones, tones, i == 0, i == segments.Count - 1, random.NextDouble);
                waves.Add(SynthesizeSegment(prompt, phones, tones, mel2ph, random));
            }
            return MegaTts3Text.Combine(waves, 0.16, o.SampleRate);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>A prepared prompt: its phones, tones and frame-to-phone map from the aligner, its latent, the duration model's
    /// state after it and its duration tokens, and its loudness.</summary>
    internal sealed record Prompt(int[] Phones, int[] Tones, int[] Mel2Ph, Tensor<T> Latent, MegaTts3DurationCache<T> DurationState,
        int[] DurationTokens, double Loudness);

    private readonly Dictionary<Tensor<T>, (Tensor<T>? Latent, Prompt Prompt)> _prompts = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary><c>preprocess</c>: the recording padded, its loudness, the aligner's phones and map, the prompt latent and the
    /// duration prompt.</summary>
    internal Prompt Prepare(double[] wav)
    {
        var o = _options;
        int ws = o.FftSize;
        if (wav.Length % ws < ws - 1) Array.Resize(ref wav, wav.Length + ws - 1 - wav.Length % ws);
        Array.Resize(ref wav, wav.Length + 12000);
        double loudness = LoudnessMeter.IntegratedLoudness(wav, o.SampleRate);
        // align(): Whisper's log-mel of the recording at 16 kHz.
        var (phones, tones, mel2ph) = Aligner.Align(SoxrResampler.LibrosaResample(wav, o.SampleRate, 16000));
        int latentFrames = mel2ph.Length / VaeStride;
        Tensor<T> latent;
        if (PromptLatent is not null)
        {
            if (PromptLatent.Shape.Length != 2 || PromptLatent.Shape[1] != o.LatentChannels)
                throw new ArgumentException($"The prompt latent is [frames, {o.LatentChannels}].");
            latent = Engine.Reshape(PromptLatent, new[] { 1, PromptLatent.Shape[0], o.LatentChannels });
        }
        else
        {
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            latent = Vae.EncodeLatent(Wave(wav), shape => Gaussian(shape, random));
        }
        if (latent.Shape[1] < latentFrames) throw new ArgumentException("The prompt latent is shorter than the prompt.");
        latent = Engine.TensorSlice(latent, new[] { 0, 0, 0 }, new[] { 1, latentFrames, o.LatentChannels });
        return PromptFrom(phones, tones, mel2ph, latent, loudness);
    }

    /// <summary><c>make_dur_prompt</c> on an aligned prompt: the duration tokens and the duration model's state.</summary>
    internal Prompt PromptFrom(int[] phones, int[] tones, int[] mel2ph, Tensor<T> latent, double loudness)
    {
        var o = _options;
        // make_dur_prompt: each phone's frames, clamped to the code size and shifted by one, the teacher-forced prompt pass.
        var counts = new int[phones.Length];
        foreach (int p in mel2ph)
            if (p > 0) counts[p - 1]++;
        var tokens = Enumerable.Range(0, phones.Length).Where(i => phones[i] > 0).Select(i => Math.Min(counts[i], o.DurationCodeSize - 1) + 1).ToArray();
        var state = new MegaTts3DurationCache<T>();
        Duration.Infer(Duration.EncodeLinguistic(phones, tones), state, tokens, null, Enumerable.Range(0, tokens.Length).ToArray());
        return new Prompt(phones, tones, mel2ph, latent, state, tokens, loudness);
    }

    // The phones the release treats as pauses: 。！？ and silence get at least 64 frames, ，； at least 32.
    private static readonly int[] LongPauses = { 148, 153, 166, 145 }, ShortPauses = { 163, 165 };

    /// <summary>
    /// <c>dur_pred</c>: the duration model continued from the prompt's state; each duration less one, clamped to [0, 127];
    /// the last lengthened by 32 frames (for the crossfade) unless final, where it is clamped to [64, 128]; each scaled up or
    /// down by a random factor up to 1 + <see cref="MegaTTS3Options.DurationDisturbance"/> (two uniform draws per phone, all
    /// the coin flips first), by <see cref="MegaTTS3Options.DurationScale"/>, rounded and clamped; pauses lengthened; the
    /// first phone of the first chunk 8 frames; the total padded to a multiple of 8; the frame-to-phone map.
    /// </summary>
    internal int[] PredictMel2Ph(Prompt prompt, int[] phones, int[] tones, bool first, bool final, Func<double> uniform)
    {
        var o = _options;
        int p = prompt.DurationTokens.Length, n = phones.Length;
        var (raw, _) = Duration.Infer(Duration.EncodeLinguistic(phones, tones), prompt.DurationState.Copy(), null, prompt.DurationTokens[p - 1],
            Enumerable.Range(p, n).ToArray());
        var dur = raw.Select(d => (float)Math.Min(Math.Max(d - 1, 0), o.DurationCodeSize - 1)).ToArray();
        if (!final) dur[n - 1] += 32;
        else dur[n - 1] = Math.Min(Math.Max(dur[n - 1], 64), 128);
        // torch.rand_like twice, in float32: the coin flips, then the factors.
        var choice = new float[n];
        var factor = new float[n];
        for (int i = 0; i < n; i++) choice[i] = (float)uniform() > 0.5f ? 1f : 0f;
        for (int i = 0; i < n; i++) factor[i] = 1f + (float)uniform() * (float)o.DurationDisturbance;
        for (int i = 0; i < n; i++)
        {
            float disturbed = dur[i] * factor[i] * choice[i] + dur[i] / factor[i] * (1 - choice[i]);
            dur[i] = Math.Min(Math.Max((float)Math.Round(disturbed * (float)o.DurationScale, MidpointRounding.ToEven), 0f), 127f);
        }
        for (int i = 0; i < n; i++)
        {
            if (Array.IndexOf(LongPauses, phones[i]) >= 0) dur[i] = Math.Max(dur[i], 64);
            if (Array.IndexOf(ShortPauses, phones[i]) >= 0) dur[i] = Math.Max(dur[i], 32);
        }
        if (first) dur[0] = 8;
        float sum = dur.Sum();
        float pad = FrameMultiple - sum % FrameMultiple;
        if (pad < FrameMultiple) dur[n - 1] += pad;
        return MegaTts3Aligner<T>.LengthRegulate(dur.Select(d => (int)Math.Round(d, MidpointRounding.ToEven)).ToArray());
    }

    /// <summary><c>prepare_inputs_for_dit</c>: the prompt's and the chunk's phones, tones and maps joined (non-Chinese tones
    /// to 3), cut to a multiple of 8 frames, as three rows (with the prompt, without it, and without it or the text); the
    /// prompt latent and its mask over the target's frames.</summary>
    internal (int[,] Phones, int[,] Tones, int[,] Mel2Ph, Tensor<T> LatentContext, Tensor<T> ContextMask) DitInputs(Prompt prompt, int[] mel2phPred,
        int[] phones, int[] tones)
    {
        var o = _options;
        int pr = prompt.Phones.Length;
        var map = prompt.Mel2Ph.Concat(mel2phPred.Select(m => m + pr)).ToArray();
        map = map.Take(map.Length / FrameMultiple * FrameMultiple).ToArray();
        var ph = prompt.Phones.Concat(phones).ToArray();
        var tone = prompt.Tones.Concat(tones).Select(t => t == 4 || (t >= 11 && t <= 15) || t == 0 ? t : 3).ToArray();
        int l = ph.Length, frames = map.Length;
        var phRows = new int[3, l];
        var toneRows = new int[3, l];
        var mapRows = new int[3, frames];
        for (int r = 0; r < 3; r++)
        {
            for (int j = 0; j < l; j++)
            {
                phRows[r, j] = r < 2 ? ph[j] : o.DitPhoneVocabulary - 1;
                toneRows[r, j] = r < 2 ? tone[j] : o.DitToneVocabulary - 1;
            }
            for (int f = 0; f < frames; f++) mapRows[r, f] = map[f];
        }
        int target = frames / VaeStride, promptFrames = prompt.Latent.Shape[1];
        if (target < promptFrames) throw new InvalidOperationException("The target is shorter than the prompt.");
        var latent = new Tensor<T>(new[] { 3, target, o.LatentChannels });
        var mask = new Tensor<T>(new[] { 3, target, 1 });
        for (int f = 0; f < promptFrames; f++)
        {
            for (int c = 0; c < o.LatentChannels; c++) latent[0, f, c] = prompt.Latent[0, f, c];
            for (int r = 0; r < 3; r++) mask[r, f, 0] = NumOps.One;
        }
        return (phRows, toneRows, mapRows, latent, mask);
    }

    /// <summary>The DiT's latent for a chunk, the prompt latent written over its start, decoded; the prompt's audio cut;
    /// matched to the prompt's loudness and kept below full scale.</summary>
    internal double[] SynthesizeSegment(Prompt prompt, int[] phones, int[] tones, int[] mel2phPred, Random random)
    {
        var o = _options;
        var (ph, tone, map, latentContext, mask) = DitInputs(prompt, mel2phPred, phones, tones);
        var linguistic = Dit.LinguisticFeatures(ph, tone, map);
        var local = Dit.LocalCondition(latentContext, mask);
        var x = Dit.Sample(mask, local, linguistic, o.Steps, o.TextGuidance, o.SpeakerGuidance, shape => Gaussian(shape, random));
        int promptFrames = prompt.Latent.Shape[1], frames = x.Shape[1];
        var prefix = prompt.Latent;
        var rest = Engine.TensorSlice(x, new[] { 0, promptFrames, 0 }, new[] { 1, frames - promptFrames, o.LatentChannels });
        var wav = Doubles(Vae.Decode(Engine.TensorConcatenate(new[] { prefix, rest }, 1)));
        int cut = promptFrames * VaeStride * o.HopSize;
        wav = wav.Skip(Math.Min(cut, wav.Length)).ToArray();
        double loudness = LoudnessMeter.IntegratedLoudness(wav, o.SampleRate);
        wav = LoudnessMeter.NormalizeLoudness(wav, loudness, prompt.Loudness);
        double peak = wav.Length == 0 ? 0 : wav.Max(Math.Abs);
        if (peak >= 1) wav = wav.Select(v => v / peak * 0.95).ToArray();
        return wav;
    }

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new T[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return new Tensor<T>(shape, new Vector<T>(values));
    }

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => MathHelper.GetNumericOperations<T>().ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) =>
        new(new[] { samples.Length }, new Vector<T>(samples.Select(v => MathHelper.GetNumericOperations<T>().FromDouble(v)).ToArray()));

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads the release's checkpoint folder (ByteDance/MegaTTS3): <c>diffusion_transformer/model_only_last.ckpt</c>,
    /// <c>duration_lm/model_only_last.ckpt</c>, <c>aligner_lm/model_only_last.ckpt</c>, <c>g2p/</c> (Qwen2's
    /// <c>model.safetensors</c> and <c>tokenizer.json</c>) and <c>wavvae/</c> (<c>model_only_last.ckpt</c> with the encoder,
    /// or the published <c>decoder.ckpt</c>).
    /// </summary>
    public void LoadReferenceWeights(string root)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        Func<string, double[]> Required(Dictionary<string, double[]> state, string what) =>
            name => state.TryGetValue(name, out var v) ? v : throw new InvalidDataException($"The {what} checkpoint has no tensor '{name}'.");
        Dit.LoadTorch(Engine, Required(ReadCheckpoint(Path.Combine(root, "diffusion_transformer", "model_only_last.ckpt"), "state_dict.dit."), "diffusion"));
        Duration.LoadTorch(Engine, Required(ReadCheckpoint(Path.Combine(root, "duration_lm", "model_only_last.ckpt"), "state_dict.dur_model."), "duration"));
        var aligner = ReadCheckpoint(Path.Combine(root, "aligner_lm", "model_only_last.ckpt"), "state_dict.model.");
        Aligner.LoadTorch(Required(aligner, "aligner"));
        string g2p = Path.Combine(root, "g2p");
        _options.G2pTokenizerJson = File.ReadAllText(Path.Combine(g2p, "tokenizer.json"));
        G2p.LoadTokenizer(_options.G2pTokenizerJson);
        var safetensors = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(g2p, "*.safetensors"))
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                safetensors[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        G2p.Network.LoadTorchWeights((name, _) => Required(safetensors, "g2p")(name));
        string vaeFull = Path.Combine(root, "wavvae", "model_only_last.ckpt");
        bool full = File.Exists(vaeFull);
        var vae = ReadCheckpoint(full ? vaeFull : Path.Combine(root, "wavvae", "decoder.ckpt"), "state_dict.model_gen.");
        Vae.LoadTorch(Engine, name => vae.TryGetValue(name, out var v) ? v : null, includeEncoder: full);
    }

    // A pytorch-lightning checkpoint's tensors under a prefix, with DistributedDataParallel's "module." dropped.
    private static Dictionary<string, double[]> ReadCheckpoint(string path, string prefix)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
        {
            if (!entry.Key.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string name = entry.Key.Substring(prefix.Length);
            if (name.StartsWith("module.", StringComparison.Ordinal)) name = name.Substring("module.".Length);
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        if (state.Count == 0) throw new InvalidDataException($"The checkpoint {path} has no tensors under '{prefix}'.");
        return state;
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MegaTTS3-Native" : "MegaTTS3-ONNX",
            Description = "MegaTTS 3: sparse-alignment latent diffusion transformer for zero-shot speech (Jiang et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MegaTTS3";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Steps"] = _options.Steps.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MegaTTS3<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
