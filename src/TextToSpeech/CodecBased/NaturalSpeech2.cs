using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Generation;
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

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// NaturalSpeech 2 (Shen et al. 2023): zero-shot speech synthesis as latent diffusion over a neural codec's continuous
/// quantized vectors, conditioned on a prior of phonemes, durations and pitch, with a speech prompt for in-context
/// learning in the duration/pitch predictors and the diffusion model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Inference: the reference recording's codec latent → the prompt encoder → query tokens; the phonemes → the
/// phoneme encoder → predicted durations (rounded) → frame-level hidden states + the embedding of the predicted pitch →
/// the condition c; z₁ ~ N(0, τ⁻¹ I) → the reverse ODE solved with the WaveNet denoiser predicting ẑ₀ → the codec's
/// decoder.</para>
/// <para>Training (<see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/>): a random span of the utterance becomes the
/// prompt and the rest the target; L = L_diff + L_dur + L_pitch with the ground-truth durations and pitch in the prior.
/// The codec is trained first, on its own (<see cref="Codec"/>). Details the paper leaves out come from Amphion's
/// implementation, whose departures are options (<see cref="NaturalSpeech2Options.Amphion"/>).</para>
/// <para><b>For Beginners:</b> give it a few seconds of someone speaking and some text; it says the text in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new NaturalSpeech2&lt;float&gt;(architecture, new NaturalSpeech2Options());
/// // train model.Codec on recordings, then model.Train(new TtsTrainingSample&lt;float&gt; { Tokens = ids, Durations = frames, Audio = wave })
/// model.Voice = model.CreateVoice(reference);           // 16 kHz
/// var audio = model.Synthesize("The quick brown fox.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers",
    "https://arxiv.org/abs/2304.09116",
    Year = 2023,
    Authors = "Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, Jiang Bian"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, WarmupSteps = 32000,
                Schedule = LearningRateSchedulerType.Noam,
                Source = "Shen et al. 2023, Sec. 5.1: AdamW with a 5e-4 learning rate and 32k warmup "
                        + "steps following the inverse square root schedule, declared here as Noam. The "
                        + "separate 2e-4 Adam setting in the same section belongs to the audio codec, "
                        + "which follows SoundStream, not to this model.")]
public partial class NaturalSpeech2<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    /// <summary>Calls the fully specified overload with the omitted parameters' defaults.</summary>
    public NaturalSpeech2(NeuralNetworkArchitecture<T> architecture)
        : this(architecture, (NaturalSpeech2Options?)null)
    {
    }


    private readonly NaturalSpeech2Options _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private NaturalSpeech2Core<T>? _core;
    private AudioCodecLayer<T>? _codec;

    private readonly List<LayerBase<T>> _priorLayers = new();
    private readonly List<LayerBase<T>> _promptLayers = new();
    private readonly List<LayerBase<T>> _denoiserLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed NaturalSpeech 2 for inference.</summary>
    public NaturalSpeech2(NeuralNetworkArchitecture<T> architecture, string modelPath, NaturalSpeech2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeech2Options();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable NaturalSpeech 2.</summary>
    public NaturalSpeech2(NeuralNetworkArchitecture<T> architecture, NaturalSpeech2Options? options)
        : base(architecture)
    {
        _options = options ?? new NaturalSpeech2Options();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        // Older serialized options contain the inherited mel default.
        if (o.MelChannels is not (0 or 80))
            throw new ArgumentException("NaturalSpeech 2 consumes waveform codec latents and has no mel frontend.", nameof(o.MelChannels));
        base.MelChannels = 0;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        MaxGradNorm = NumOps.FromDouble(o.GradientClipNorm);
        if (o.Codec is null) throw new ArgumentException("NaturalSpeech 2 needs a codec (NaturalSpeech2Options.Codec).");
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.HiddenDim % 2 != 0 || o.HiddenDim < 4) throw new ArgumentException("HiddenDim must be even and at least 4 (the step embedding's halves).");
        if (o.PitchBins < 2 || o.PitchMin <= 0 || o.PitchMax <= o.PitchMin) throw new ArgumentException("The pitch buckets need 0 < PitchMin < PitchMax and two or more bins.");
        if (o.PromptMinFraction < 0 || o.PromptMaxFraction < o.PromptMinFraction || o.PromptMaxFraction >= 1)
            throw new ArgumentException("The prompt fractions satisfy 0 <= PromptMinFraction <= PromptMaxFraction < 1.");
        if (o.Temperature <= 0) throw new ArgumentException("Temperature must be positive.");
        if (!(o.TimeEpsilon >= 0 && o.TimeEpsilon < .5))
            throw new ArgumentOutOfRangeException(nameof(o.TimeEpsilon), "Diffusion time clipping requires 0 <= TimeEpsilon < 0.5.");
        if (o.SamplingSteps < 1) throw new ArgumentException("SamplingSteps must be positive.");
        if (o.DenoiserAttentionEvery < 1 || o.PredictorAttentionEvery < 1) throw new ArgumentException("The attention intervals must be positive.");
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The reverse ODE's steps.</remarks>
    public int NumFlowSteps => _options.SamplingSteps;

    private bool HasPaperLayers => _core is not null;

    // The training and sampling seed (Options.Seed, 1234 when unset).
    private int SeedValue => _options.Seed ?? 1234;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal NaturalSpeech2Core<T> Core => _core ?? throw NotNative();

    /// <summary>
    /// The codec (<see cref="NaturalSpeech2Options.Codec"/>): a model of its own. The paper trains it first (following
    /// SoundStream); train it through its own <c>Train</c> before the diffusion model, which keeps it frozen.
    /// </summary>
    public NeuralAudioCodecBase<T> Codec => (NeuralAudioCodecBase<T>?)_codec?.Codec ?? throw NotNative();

    internal ResidualVectorQuantizerLayer<T> Quantizer => Codec.CodecQuantizer as ResidualVectorQuantizerLayer<T>
        ?? throw new InvalidOperationException("NaturalSpeech 2 reads a residual vector quantizer's codebooks; the codec has none.");

    internal int Codebooks => Codec.QuantizersForBandwidth(_options.Codec.TargetBandwidthKbps);

    private Ns2Configuration Configuration()
    {
        var o = _options;
        return new Ns2Configuration
        {
            Vocab = o.VocabSize,
            Latent = CodecDimension(o.Codec),
            Hidden = o.HiddenDim,
            Heads = o.NumHeads,
            EncoderLayers = o.NumEncoderLayers,
            EncoderFilter = o.FilterChannels,
            EncoderKernel = o.EncoderKernel,
            EncoderDropout = o.DropoutRate,
            StyleAdaptiveEncoder = o.StyleAdaptivePhonemeEncoder,
            PromptLayers = o.NumPromptLayers,
            Positions = o.Positions,
            DurationLayers = o.DurationLayers,
            DurationKernel = o.DurationKernel,
            PitchLayers = o.PitchLayers,
            PitchKernel = o.PitchKernel,
            PredictorAttentionEvery = o.PredictorAttentionEvery,
            PredictorDropout = o.PredictorDropout,
            PitchMin = o.PitchMin,
            PitchMax = o.PitchMax,
            PitchBins = o.PitchBins,
            QueryTokens = o.QueryTokens,
            DenoiserLayers = o.DenoiserLayers,
            DenoiserDilation = o.DenoiserDilation,
            DenoiserDilationCycle = o.DenoiserDilationCycle,
            DenoiserAttentionEvery = o.DenoiserAttentionEvery,
            DenoiserAttentionOffset = o.DenoiserAttentionOffset,
            DenoiserAttentionDropout = o.DenoiserAttentionDropout,
        };
    }

    private static int CodecDimension(NeuralAudioCodecOptions codec) => codec switch
    {
        SoundStreamOptions s => s.Dimension,
        EnCodecOptions e => e.Dimension,
        _ => throw new ArgumentException($"NaturalSpeech 2 builds a SoundStream or EnCodec codec, not {codec.GetType().Name}."),
    };

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        _core = new NaturalSpeech2Core<T>(Engine, Configuration(), random, _priorLayers, _promptLayers, _denoiserLayers);

        var codecOptions = (NeuralAudioCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = Architecture.RandomSeed };
        NeuralAudioCodecBase<T> codec = codecOptions switch
        {
            SoundStreamOptions s => new SoundStream<T>(architecture, s),
            EnCodecOptions e => new EnCodec<T>(architecture, e),
            _ => throw new ArgumentException($"NaturalSpeech 2 builds a SoundStream or EnCodec codec, not {codecOptions.GetType().Name}."),
        };
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"NaturalSpeech 2's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's ({codecRate} Hz, " +
                $"{codec.HopLength}-sample hop).");
        _codec = new AudioCodecLayer<T>(codec);
        AddUnsplitLayers(_priorLayers.Concat(_promptLayers).Concat(_denoiserLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The phone set (Amphion's): CMUdict's 84 ARPAbet symbols, then sp, spn, sil, &lt;s&gt; and &lt;/s&gt;.</summary>
    public static IReadOnlyList<string> Symbols => NaturalSpeechArpabet.Symbols;

    /// <summary>
    /// The text's phone ids (Amphion's <c>preprocess_english</c>): trailing punctuation stripped, the text split at
    /// punctuation and spaces, each piece looked up in the lexicon (<see cref="NaturalSpeech2Options.LexiconPath"/>) or
    /// pronounced by g2p_en, a punctuation mark becoming <c>sp</c>.
    /// </summary>
    public int[] TextToPhonemes(string text) => NaturalSpeechArpabet.Phonemes(text, _options.LexiconPath);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return IdTensor(TextToPhonemes(text));
    }

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

    // ---------------------------------------------------------------- codec latents

    /// <summary>The codec's codes <c>[codebooks, frames]</c> of audio at <see cref="TtsModelOptions.SampleRate"/>.</summary>
    internal int[,] Encode(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        return Codec.Encode(Engine.Reshape(audio, new[] { audio.Length }));
    }

    /// <summary>The quantized latent <c>[1, dim, frames]</c> of codes <c>[codebooks, frames]</c>: the sum of their code
    /// vectors (Eq. 1).</summary>
    internal Tensor<T> Latent(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        return Quantizer.Decode(codes);
    }

    /// <summary>Codes <c>[frames, codebooks]</c> (a sample's or a voice's codec tokens) as <c>[codebooks, frames]</c>.</summary>
    private int[,] CodesOf(Tensor<T> tokens, string name)
    {
        if (tokens.Rank != 2) throw new ArgumentException("Codec tokens are [frames, codebooks].", name);
        int frames = tokens.Shape[0], codebooks = tokens.Shape[1];
        if (codebooks != Codebooks) throw new ArgumentException($"Expected {Codebooks} codebooks, got {codebooks}.", name);
        var codes = LevelMajorCodes(tokens);
        foreach (int c in codes)
            if (c < 0 || c >= Codec.CodebookSize) throw new ArgumentOutOfRangeException(name, $"Codes lie in [0, {Codec.CodebookSize}).");
        return codes;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <summary>A voice from a recording of the speaker at <see cref="TtsModelOptions.SampleRate"/> (the speech prompt).</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (recording.Length < _options.HopSize) throw new ArgumentException("The recording is shorter than one codec frame.", nameof(recording));
        return new TtsVoice<T> { ReferenceAudio = Engine.Reshape(recording, new[] { recording.Length }) };
    }

    private Tensor<T> PromptLatent(TtsVoice<T> voice) =>
        voice.ReferenceCodecTokens is { } tokens ? Latent(CodesOf(tokens, nameof(voice))) : Latent(Encode(voice.ReferenceAudio!));

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phone ids (<see cref="TextToPhonemes"/>); the result is the waveform at
    /// <see cref="TtsModelOptions.SampleRate"/> in <see cref="TtsModelBase{T}.Voice"/>'s voice.</remarks>
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
            using var _ = new NoGradScope<T>();
            var o = _options;
            var core = Core;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
            var (prompt, query) = core.Prompt(PromptLatent(voice), false, null);
            var (hidden, logDurations) = core.Phonemes(phonemes, prompt, false, null);
            // Amphion lets every duration round to zero and then expands to nothing; such a sequence gets one frame per
            // phoneme instead.
            var durations = NaturalSpeech2Core<T>.RoundDurations(logDurations);
            if (durations.Sum() == 0) for (int i = 0; i < durations.Length; i++) durations[i] = 1;
            var (condition, _) = core.Condition(hidden, durations, null, prompt, false, null);
            int frames = condition.Shape[0];
            var start = Gaussian(new[] { 1, CodecDimension(o.Codec), frames }, random, 1.0 / Math.Sqrt(o.Temperature));
            var x0 = Reverse(start, condition, query, o.SamplingSteps);
            var wave = Codec.DecodeEmbeddings(x0);
            return Engine.Reshape(wave, new[] { wave.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Reverse diffusion from z₁ (Eq. 5's ODE, or the flow's) in <paramref name="steps"/> steps at the midpoints
    /// t = 1 − (i + ½)h: Euler, or Amphion's midpoint solver.
    /// </summary>
    internal Tensor<T> Reverse(Tensor<T> start, Tensor<T> condition, Tensor<T> query, int steps)
    {
        var o = _options;
        double h = 1.0 / Math.Max(steps, 1);
        var xt = start;
        var half = Ops.FromDouble(0.5);
        for (int i = 0; i < steps; i++)
        {
            double t = 1.0 - (i + 0.5) * h;
            var dxt = Drift(xt, condition, query, t, h);
            if (o.Diffusion == NaturalSpeech2Diffusion.ScoreSde && o.Solver == NaturalSpeech2Solver.Midpoint)
            {
                var middle = Engine.TensorMultiplyScalar(Engine.TensorAdd(Engine.TensorSubtract(xt, dxt), xt), half);

                dxt = Drift(middle, condition, query, t + 0.5 * h, h);
            }
            xt = Engine.TensorSubtract(xt, dxt);
        }
        return xt;
    }

    // cal_dxt: −½ h β_t (∇log p + z_t / σ²) with ∇log p = −(z_t − ρ(ẑ₀, t)) / Σ_t; the flow's h · (ε − z₀)̂.
    private Tensor<T> Drift(Tensor<T> xt, Tensor<T> condition, Tensor<T> query, double t, double h)
    {
        var o = _options;
        var prediction = Core.Denoiser.Forward(xt, t, condition, query, false, null);
        if (o.Diffusion == NaturalSpeech2Diffusion.FlowMatching) return Engine.TensorMultiplyScalar(prediction, Ops.FromDouble(h));
        double cumBeta = NaturalSpeech2Core<T>.CumulativeBeta(t, o.BetaMin, o.BetaMax), beta = NaturalSpeech2Core<T>.Beta(t, o.BetaMin, o.BetaMax);
        var meanPrediction = Engine.TensorMultiplyScalar(prediction, Ops.FromDouble(Math.Exp(-0.5 * cumBeta)));
        double variance = 1.0 - Math.Exp(-cumBeta);
        var score = Engine.TensorMultiplyScalar(Engine.TensorSubtract(xt, meanPrediction), Ops.FromDouble(-1.0 / (variance + 1e-8)));
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(score, xt), Ops.FromDouble(-0.5 * h * beta));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    private static Tensor<T> Gaussian(int[] shape, Random random, double std = 1.0)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, Ops.FromDouble(std * AiDotNet.TextToSpeech.Latest.IndexTtsOps.Gaussian(random)));
        return t;
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "NaturalSpeech2-Native" : "NaturalSpeech2-ONNX",
            Description = "NaturalSpeech 2: latent diffusion over a neural codec's continuous vectors with speech prompting (Shen et al., 2023)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "NaturalSpeech2";
        metadata.AdditionalInfo["Diffusion"] = _options.Diffusion.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech2<T>));
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
