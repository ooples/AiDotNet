using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Voicebox: a text-guided speech infilling model with separate regression-duration and conditional-flow audio models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The native path implements Le et al. (2023): position-dependent phones with ghost silences, a duration-context
/// Transformer, an 80-bin normalized log-mel optimal-transport conditional flow, midpoint ODE integration with
/// classifier-free guidance, and the paper's 16 kHz HiFi-GAN V1 adaptation. It supports text-only diverse synthesis,
/// reference-conditioned zero-shot synthesis, and arbitrary mel infilling. Voicebox is not a neural-codec language
/// model; the previous amplitude quantizer and sinusoidal "decoder" were placeholders and have been removed.
/// </para>
/// <para>
/// Meta did not release Voicebox weights or source code. Native checkpoints therefore use an explicit AiDotNet
/// conversion schema and are never described as official weights. Architecture details stated in the paper are exact;
/// unpublished ordinary Transformer details use the documented pre-norm implementation in this class.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale",
    "https://arxiv.org/abs/2306.15687",
    Year = 2023,
    Authors = "Le et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, WarmupSteps = 5000,
    Schedule = LearningRateSchedulerType.LinearWarmup,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    MinLearningRate = 0, MaxGradientNorm = 0.2,
    Provenance = RecipeProvenance.Stated,
    Source = "Le et al. 2023 Sec. 5.1 and Tables D13-D14: Adam at peak 1e-4, 5k linear warmup and linear decay; "
        + "the audio model clips gradient norm to 0.2.")]
public partial class Voicebox<T> : TtsModelBase<T>, AiDotNet.TextToSpeech.Interfaces.ITtsModel<T>
{
    private readonly VoiceboxOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly Random _trainingRandom;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private VoiceboxPhoneTokenizer? _tokenizer;
    private VoiceboxAudioModel<T>? _audioModel;
    private VoiceboxDurationModel<T>? _durationModel;
    private CenteredLogMel<T>? _melFeatures;
    private HiFiGanGenerator<T>? _vocoder;

    /// <summary>Creates an ONNX-backed Voicebox inference wrapper.</summary>
    public Voicebox(NeuralNetworkArchitecture<T> architecture, string modelPath, VoiceboxOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VoiceboxOptions();
        ValidateOptions(_options);
        Options = _options;
        InitializePhoneFrontend();
        _trainingRandom = new Random(_options.SamplingSeed);
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Creates the paper-native duration, audio-flow, mel, and vocoder graph.</summary>
    public Voicebox(
        NeuralNetworkArchitecture<T> architecture,
        VoiceboxOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VoiceboxOptions();
        ValidateOptions(_options);
        Options = _options;
        InitializePhoneFrontend();
        _useNativeMode = true;
        _trainingRandom = new Random(_options.SamplingSeed);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ConfigureBaseProperties();
        InitializeLayers();
    }

    /// <summary>Maximum number of phones accepted by the text front end.</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>Whether the native instance owns the complete paper pipeline.</summary>
    public bool HasPaperPipeline => _audioModel is not null && _durationModel is not null && _vocoder is not null;

    /// <summary>Audio vector-field Transformer block count.</summary>
    public int AudioLayerCount => _audioModel?.LayerCount ?? 0;

    /// <summary>Regression duration Transformer block count.</summary>
    public int DurationLayerCount => _durationModel?.LayerCount ?? 0;

    /// <summary>Default vector-field evaluations (two per midpoint step).</summary>
    public int NumberOfFunctionEvaluations => 2 * _options.OdeSteps;

    /// <summary>The configured phone inventory size, or zero until a text frontend is selected.</summary>
    public int PhoneTableSize => _phoneVocabulary is not null || _options.UseNativeEnglishFallback
        ? _tokenizer?.TableSize ?? 0 : 0;

    /// <summary>Voicebox predicts continuous mels and has no codec codebooks.</summary>
    public override int CodecTokenCodebooks => 0;

    /// <summary>Voicebox predicts continuous mels and has no codec vocabulary.</summary>
    public override int CodecTokenVocabulary => 0;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected override int TargetWindowSize => _options.WindowLength;

    /// <inheritdoc />
    protected override int TargetFftSize => _options.FftSize;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperPipeline ? LayerInputDomain.Indices(_options.PhoneVocabularySize) : base.GetInputDomain(inputShape);

    /// <summary>Returns the paper's ghost-silence and word-position-dependent phone sequence.</summary>
    public IReadOnlyList<string> Phonemize(string text)
    {
        ThrowIfDisposed();
        return Tokenizer.Phonemize(text, MfaWordPhonemizer, _options.UseNativeEnglishFallback);
    }

    /// <summary>
    /// Creates the complete zero-shot context. Voicebox requires externally forced-aligned phone durations for the
    /// prompt; it does not infer an alignment by evenly spreading frames.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string transcript, int[] durations)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(recording);
        Validation.Guard.NotNull(transcript);
        Validation.Guard.NotNull(durations);
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        var tokens = Tokenizer.Encode(transcript, _options.MaxTextLength, MfaWordPhonemizer, _options.UseNativeEnglishFallback);
        if (durations.Length != tokens.Length)
            throw new ArgumentException($"The forced alignment has {durations.Length} durations for {tokens.Length} phones.", nameof(durations));
        if (durations.Any(duration => duration < 0))
            throw new ArgumentException("Phone durations cannot be negative.", nameof(durations));
        var audio = sampleRate == _options.SampleRate
            ? Engine.Reshape(recording, new[] { recording.Length })
            : AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(recording, new[] { recording.Length }), sampleRate, _options.SampleRate);
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = RawMel(audio);
        int alignedFrames = durations.Sum();
        if (alignedFrames <= 0 || alignedFrames > mel.Shape[0])
            throw new ArgumentException(
                $"The forced alignment covers {alignedFrames} frames but the prompt has {mel.Shape[0]} mel frames.", nameof(durations));
        mel = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { alignedFrames, _options.MelChannels });
        return new TtsVoice<T>
        {
            Reference = mel,
            ReferenceAudio = audio,
            ReferenceTokens = VoiceboxOps<T>.Ids(tokens),
            ReferenceDurations = (int[])durations.Clone(),
        };
    }

    /// <summary>Synthesizes either diverse speech or zero-shot speech when <see cref="TtsModelBase{T}.Voice"/> is set.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var phones = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(phones);
        RequirePipeline();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return SynthesizePhones(VoiceboxOps<T>.ToIds(phones, _options.PhoneVocabularySize), Voice,
                new Random(_options.SamplingSeed));
        }
        finally { SetTrainingMode(wasTraining); }
    }

    /// <summary>
    /// Infills the frames selected by <paramref name="mask"/> while preserving all context frames exactly.
    /// </summary>
    /// <param name="normalizedMel">Normalized mel <c>[frames, 80]</c>.</param>
    /// <param name="framePhoneIds">One phone id per frame.</param>
    /// <param name="mask">True for frames to regenerate.</param>
    public Tensor<T> Infill(Tensor<T> normalizedMel, IReadOnlyList<int> framePhoneIds, IReadOnlyList<bool> mask)
    {
        ThrowIfDisposed();
        RequirePipeline();
        if (normalizedMel.Rank != 2 || normalizedMel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"Voicebox infill expects [frames, {_options.MelChannels}] normalized mels.", nameof(normalizedMel));
        int frames = normalizedMel.Shape[0];
        if (framePhoneIds.Count != frames || mask.Count != frames)
            throw new ArgumentException("The mel, frame-level phones, and infill mask must have the same length.");
        var context = new Tensor<T>(normalizedMel.Shape.ToArray());
        for (int f = 0; f < frames; f++)
            if (!mask[f])
                for (int m = 0; m < _options.MelChannels; m++) context[f, m] = normalizedMel[f, m];
        Tensor<T> generated;
        using (new NoGradScope<T>()) generated = Integrate(context, framePhoneIds, new Random(_options.SamplingSeed));
        var result = new Tensor<T>(normalizedMel.Shape.ToArray());
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < _options.MelChannels; m++)
                result[f, m] = mask[f] ? generated[f, m] : normalizedMel[f, m];
        return result;
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text) =>
        VoiceboxOps<T>.Ids(Tokenizer.Encode(text, _options.MaxTextLength, MfaWordPhonemizer, _options.UseNativeEnglishFallback));

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _tokenizer ??= new VoiceboxPhoneTokenizer(_options.PhoneVocabularySize, _phoneVocabulary);
        _audioModel = new VoiceboxAudioModel<T>(Engine, ComponentLayers, _options);
        _durationModel = new VoiceboxDurationModel<T>(Engine, ComponentLayers, _options);
        _melFeatures = new CenteredLogMel<T>(Engine, _options.SampleRate, _options.FftSize, _options.HopSize,
            _options.WindowLength, _options.MelChannels, _options.MelMinFrequency, _options.MelMaxFrequency,
            floor: 1e-5, naturalLog: true);
        _vocoder = new HiFiGanGenerator<T>(Engine, _options.MelChannels, _options.VocoderInitialChannels,
            _options.VocoderUpsampleRates, _options.VocoderUpsampleKernels, _options.VocoderResBlockKernels,
            _options.VocoderResBlockDilations, resblockType1: true,
            initialization: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 17));
        ComponentLayers.AddRange(_vocoder.Layers);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (HasPaperPipeline)
        {
            using var _ = new NoGradScope<T>();
            return SynthesizePhones(VoiceboxOps<T>.ToIds(input, _options.PhoneVocabularySize), Voice,
                new Random(_options.SamplingSeed));
        }
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_optimizer is null) throw new NotSupportedException("The native Voicebox optimizer is unavailable.");
        if (sample.Mel is null && sample.Audio is null)
            throw new ArgumentException("Voicebox training requires a log-mel target or its recording.", nameof(sample));
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens,
            (_, _) => BuildObjective(sample, _trainingRandom), _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (sample.Durations is null)
            throw new ArgumentException("Voicebox training requires forced-alignment durations.", nameof(sample));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample, new Random(_options.SamplingSeed + 1))[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Voicebox-Native" : "Voicebox-ONNX",
            Description = "Voicebox duration-context Transformer, OT conditional flow, and adapted HiFi-GAN V1",
            FeatureCount = _options.AudioModelDim,
        };
        metadata.AdditionalInfo["Architecture"] =
            "Regression duration Transformer -> masked OT-CFM audio Transformer -> 16 kHz HiFi-GAN V1";
        metadata.AdditionalInfo["AudioLayers"] = _options.AudioLayers;
        metadata.AdditionalInfo["DurationLayers"] = _options.DurationLayers;
        metadata.AdditionalInfo["MelFrameRate"] = _options.SampleRate / _options.HopSize;
        metadata.AdditionalInfo["NFE"] = NumberOfFunctionEvaluations;
        metadata.AdditionalInfo["GuidanceStrength"] = _options.GuidanceStrength;
        metadata.AdditionalInfo["SymmetricBidirectionalAlibi"] = true;
        metadata.AdditionalInfo["UnetSkipConnections"] = true;
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = false;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private VoiceboxPhoneTokenizer Tokenizer => _tokenizer
        ?? throw new InvalidOperationException("Voicebox's native phone front end is unavailable in ONNX/custom-layer mode.");

    private void ConfigureBaseProperties()
    {
        SampleRate = _options.SampleRate;
        MelChannels = _options.MelChannels;
        HopSize = _options.HopSize;
        HiddenDim = _options.AudioModelDim;
    }

    private Tensor<T> SynthesizePhones(int[] targetPhones, TtsVoice<T>? voice, Random random)
    {
        var allPhones = new List<int>();
        var contextDurations = new List<int>();
        Tensor<T>? reference = null;
        int referenceFrames = 0;
        if (voice is not null)
        {
            if (voice.ReferenceTokens is null || voice.ReferenceDurations is null)
                throw new InvalidOperationException(
                    "Voicebox zero-shot synthesis needs the prompt's phone ids and forced-alignment durations.");
            var promptPhones = VoiceboxOps<T>.ToIds(voice.ReferenceTokens, _options.PhoneVocabularySize);
            if (promptPhones.Length != voice.ReferenceDurations.Length)
                throw new InvalidOperationException("The Voicebox prompt has a different number of phones and durations.");
            reference = voice.Reference is not null ? EnsureFrameMajorMel(voice.Reference)
                : voice.ReferenceAudio is not null ? RawMel(voice.ReferenceAudio)
                : throw new InvalidOperationException("Voicebox zero-shot synthesis needs a reference mel or recording.");
            referenceFrames = voice.ReferenceDurations.Sum();
            if (referenceFrames <= 0 || referenceFrames > reference.Shape[0])
                throw new InvalidOperationException("The Voicebox prompt alignment does not fit its mel spectrogram.");
            reference = Engine.TensorSlice(reference, new[] { 0, 0 }, new[] { referenceFrames, _options.MelChannels });
            allPhones.AddRange(promptPhones);
            contextDurations.AddRange(voice.ReferenceDurations);
        }
        int targetStartPhone = allPhones.Count;
        allPhones.AddRange(targetPhones);
        contextDurations.AddRange(Enumerable.Repeat(0, targetPhones.Length));
        var durations = InferDurations(allPhones, contextDurations, targetStartPhone);
        int targetFrames = durations.Skip(targetStartPhone).Sum();
        if (targetFrames <= 0) throw new InvalidOperationException("Voicebox predicted an empty target utterance.");
        int totalFrames = referenceFrames + targetFrames;
        if (totalFrames > _options.MaximumAudioFrames)
            throw new ArgumentException(
                $"Voicebox synthesis needs {totalFrames} frames, above the configured {_options.MaximumAudioFrames} frame limit.", nameof(targetPhones));
        var framePhones = RepeatPhones(allPhones, durations);
        var context = new Tensor<T>(new[] { totalFrames, _options.MelChannels });
        if (reference is not null)
        {
            var normalized = Normalize(reference);
            for (int f = 0; f < referenceFrames; f++)
                for (int m = 0; m < _options.MelChannels; m++) context[f, m] = normalized[f, m];
        }
        var generated = Integrate(context, framePhones, random);
        var target = Engine.TensorSlice(generated,
            new[] { referenceFrames, 0 }, new[] { targetFrames, _options.MelChannels });
        return Vocoder(DeNormalize(target));
    }

    private int[] InferDurations(IReadOnlyList<int> phones, IReadOnlyList<int> contextDurations, int targetStart)
    {
        var transformed = new Tensor<T>(new[] { phones.Count, 1 });
        for (int i = 0; i < phones.Count; i++)
            transformed[i, 0] = NumOps.FromDouble(Math.Log(1.0 + contextDurations[i]));
        var prediction = _durationModel!.Forward(phones, transformed);
        var durations = contextDurations.ToArray();
        double maximum = Math.Log(1.0 + _options.MaximumPhoneDuration);
        for (int i = targetStart; i < durations.Length; i++)
        {
            double value = NumOps.ToDouble(prediction[i, 0]);
            if (double.IsNaN(value) || double.IsInfinity(value)) value = 0;
            int duration = (int)Math.Round(Math.Exp(MathPolyfill.Clamp(value, 0, maximum)) - 1.0);
            durations[i] = phones[i] == Tokenizer.SilenceId ? Math.Max(0, duration) : Math.Max(1, duration);
        }
        return durations;
    }

    private Tensor<T> Integrate(Tensor<T> context, IReadOnlyList<int> framePhones, Random random)
    {
        int frames = context.Shape[0];
        var state = Gaussian(new[] { frames, _options.MelChannels }, random);
        double dt = 1.0 / _options.OdeSteps;
        for (int step = 0; step < _options.OdeSteps; step++)
        {
            double t = step * dt;
            var first = GuidedVelocity(state, context, framePhones, t);
            var midpoint = Engine.TensorAdd(state,
                Engine.TensorMultiplyScalar(first, NumOps.FromDouble(0.5 * dt)));
            var second = GuidedVelocity(midpoint, context, framePhones, t + 0.5 * dt);
            state = Engine.TensorAdd(state, Engine.TensorMultiplyScalar(second, NumOps.FromDouble(dt)));
        }
        return state;
    }

    private Tensor<T> GuidedVelocity(Tensor<T> state, Tensor<T> context, IReadOnlyList<int> phones, double time)
    {
        var conditional = _audioModel!.Forward(state, context, phones, time);
        double guidance = _options.GuidanceStrength;
        if (guidance == 0) return conditional;
        var unconditional = _audioModel.Forward(state, context, phones, time, conditioned: false);
        return Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(conditional, NumOps.FromDouble(1.0 + guidance)),
            Engine.TensorMultiplyScalar(unconditional, NumOps.FromDouble(guidance)));
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample, Random random)
    {
        var durations = sample.Durations
            ?? throw new ArgumentException("Voicebox training requires forced-alignment durations.", nameof(sample));
        int[] phones = VoiceboxOps<T>.ToIds(sample.Tokens, _options.PhoneVocabularySize);
        if (phones.Length != durations.Length)
            throw new ArgumentException($"Voicebox received {phones.Length} phones and {durations.Length} durations.", nameof(sample));
        Tensor<T> rawMel = sample.Mel is not null ? EnsureFrameMajorMel(sample.Mel)
            : RawMel(sample.Audio ?? throw new ArgumentException("Voicebox training requires audio or a mel target.", nameof(sample)));
        if (durations.Any(value => value < 0) || durations.Sum() != rawMel.Shape[0])
            throw new ArgumentException("Voicebox durations must be non-negative and sum to the mel frame count.", nameof(sample));
        (phones, durations, rawMel) = TrainingChunk(phones, durations, rawMel, random);
        var target = Normalize(rawMel);
        int frames = target.Shape[0];

        bool[] audioMask = AudioMask(durations, random);
        var context = new Tensor<T>(target.Shape.ToArray());
        for (int f = 0; f < frames; f++)
            if (!audioMask[f])
                for (int m = 0; m < _options.MelChannels; m++) context[f, m] = target[f, m];
        var prior = Gaussian(target.Shape.ToArray(), random);
        double time = random.NextDouble();
        var (path, velocityTarget) = VoiceboxObjectives<T>.Flow(Engine, target, prior, time, _options.SigmaMinimum);
        var framePhones = RepeatPhones(phones, durations);
        bool audioConditioned = random.NextDouble() >= _options.ConditionalDropout;
        var velocity = _audioModel!.Forward(path, context, framePhones, time, audioConditioned);
        var audioLoss = MaskedMeanSquare(Engine.TensorSubtract(velocityTarget, velocity), audioMask);

        bool[] durationMask = DurationMask(phones.Length, random);
        var durationContext = new Tensor<T>(new[] { phones.Length, 1 });
        var durationTarget = new Tensor<T>(new[] { phones.Length, 1 });
        for (int i = 0; i < phones.Length; i++)
        {
            durationTarget[i, 0] = NumOps.FromDouble(VoiceboxObjectives<T>.TransformDuration(durations[i], random.NextDouble()));
            if (!durationMask[i]) durationContext[i, 0] = durationTarget[i, 0];
        }
        bool durationConditioned = random.NextDouble() >= _options.ConditionalDropout;
        var durationPrediction = _durationModel!.Forward(phones, durationContext, durationConditioned);
        var durationLoss = MaskedMeanAbsolute(Engine.TensorSubtract(durationTarget, durationPrediction), durationMask);
        return Engine.TensorAdd(audioLoss, durationLoss);
    }

    private (int[] Phones, int[] Durations, Tensor<T> Mel) TrainingChunk(
        int[] phones, int[] durations, Tensor<T> mel, Random random)
    {
        if (mel.Shape[0] <= _options.MaximumAudioFrames) return (phones, (int[])durations.Clone(), mel);
        var validStarts = Enumerable.Range(0, durations.Length)
            .Where(start => durations[start] <= _options.MaximumAudioFrames).ToArray();
        if (validStarts.Length == 0)
            throw new ArgumentException("No single phone fits in Voicebox's maximum training chunk.");
        int first = validStarts[random.Next(validStarts.Length)], count = 0, frames = 0;
        while (first + count < durations.Length && frames + durations[first + count] <= _options.MaximumAudioFrames)
        {
            frames += durations[first + count];
            count++;
        }
        if (frames == 0) throw new ArgumentException("Voicebox selected an empty training chunk.");
        int offset = durations.Take(first).Sum();
        return (
            phones.Skip(first).Take(count).ToArray(),
            durations.Skip(first).Take(count).ToArray(),
            Engine.TensorSlice(mel, new[] { offset, 0 }, new[] { frames, _options.MelChannels }));
    }

    private bool[] AudioMask(IReadOnlyList<int> durations, Random random)
    {
        int frames = durations.Sum();
        if (random.NextDouble() < _options.SpectrogramDropProbability)
            return Enumerable.Repeat(true, frames).ToArray();
        double fraction = _options.SpectrogramMaskMinimum
            + random.NextDouble() * (_options.SpectrogramMaskMaximum - _options.SpectrogramMaskMinimum);
        int wanted = MathPolyfill.Clamp((int)Math.Ceiling(frames * fraction), 1, frames);
        int start = random.Next(frames - wanted + 1), end = start + wanted;
        var mask = new bool[frames];
        int cursor = 0;
        foreach (int duration in durations)
        {
            int next = cursor + duration;
            if (cursor < end && next > start)
                for (int f = cursor; f < next; f++) mask[f] = true;
            cursor = next;
        }
        return mask;
    }

    private bool[] DurationMask(int count, Random random)
    {
        if (random.NextDouble() < _options.DurationDropProbability)
            return Enumerable.Repeat(true, count).ToArray();
        double fraction = _options.DurationMaskMinimum
            + random.NextDouble() * (_options.DurationMaskMaximum - _options.DurationMaskMinimum);
        int length = MathPolyfill.Clamp((int)Math.Ceiling(count * fraction), 1, count);
        int start = random.Next(count - length + 1);
        var mask = new bool[count];
        for (int i = start; i < start + length; i++) mask[i] = true;
        return mask;
    }

    private Tensor<T> MaskedMeanSquare(Tensor<T> difference, IReadOnlyList<bool> frameMask)
        => VoiceboxObjectives<T>.MaskedMse(Engine, difference, frameMask);

    private Tensor<T> MaskedMeanAbsolute(Tensor<T> difference, IReadOnlyList<bool> maskValues)
    {
        var mask = new Tensor<T>(new[] { maskValues.Count, 1 });
        int count = 0;
        for (int i = 0; i < maskValues.Count; i++)
            if (maskValues[i]) { mask[i, 0] = NumOps.One; count++; }
        return Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorAbs(difference), mask), new[] { 0, 1 }, false),
            NumOps.FromDouble(1.0 / Math.Max(1, count)));
    }

    private Tensor<T> RawMel(Tensor<T> audio)
    {
        var channelsFirst = _melFeatures!.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        return Engine.TensorTranspose(Engine.Reshape(channelsFirst,
            new[] { _options.MelChannels, channelsFirst.Shape[2] }));
    }

    private Tensor<T> EnsureFrameMajorMel(Tensor<T> mel)
    {
        if (mel.Rank == 2 && mel.Shape[1] == _options.MelChannels) return mel;
        if (mel.Rank == 2 && mel.Shape[0] == _options.MelChannels) return Engine.TensorTranspose(mel);
        if (mel.Rank == 3 && mel.Shape[0] == 1 && mel.Shape[1] == _options.MelChannels)
            return Engine.TensorTranspose(Engine.Reshape(mel, new[] { _options.MelChannels, mel.Shape[2] }));
        throw new ArgumentException($"Voicebox mels are [frames, {_options.MelChannels}] or [1, {_options.MelChannels}, frames].", nameof(mel));
    }

    private Tensor<T> Normalize(Tensor<T> mel) => Engine.TensorMultiplyScalar(
        Engine.TensorAddScalar(mel, NumOps.FromDouble(-_options.MelMean)),
        NumOps.FromDouble(1.0 / _options.MelStandardDeviation));

    private Tensor<T> DeNormalize(Tensor<T> normalized) => Engine.TensorAddScalar(
        Engine.TensorMultiplyScalar(normalized, NumOps.FromDouble(_options.MelStandardDeviation)),
        NumOps.FromDouble(_options.MelMean));

    private Tensor<T> Vocoder(Tensor<T> frameMajorMel)
    {
        int frames = frameMajorMel.Shape[0];
        var channelsFirst = Engine.Reshape(Engine.TensorTranspose(frameMajorMel),
            new[] { 1, _options.MelChannels, frames });
        var waveform = _vocoder!.Forward(channelsFirst);
        return Engine.Reshape(waveform, new[] { waveform.Length });
    }

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i += 2)
        {
            double u1 = Math.Max(random.NextDouble(), 1e-12), u2 = random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            tensor[i] = NumOps.FromDouble(radius * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < tensor.Length) tensor[i + 1] = NumOps.FromDouble(radius * Math.Sin(2.0 * Math.PI * u2));
        }
        return tensor;
    }

    private static int[] RepeatPhones(IReadOnlyList<int> phones, IReadOnlyList<int> durations)
    {
        var result = new int[durations.Sum()];
        int cursor = 0;
        for (int i = 0; i < phones.Count; i++)
            for (int j = 0; j < durations[i]; j++) result[cursor++] = phones[i];
        return result;
    }

    private void RequirePipeline()
    {
        if (!HasPaperPipeline)
            throw new InvalidOperationException("Voicebox's paper-native graph is unavailable in ONNX/custom-layer mode.");
    }

    private static void ValidateOptions(VoiceboxOptions options)
    {
        if (options.SampleRate != 16000 || options.HopSize != 160 || options.FftSize != 1024
            || options.WindowLength != 640 || options.MelChannels != 80)
            throw new ArgumentException("Voicebox requires 16 kHz, hop 160, FFT 1024, window 640, and 80 mel bins.", nameof(options));
        if (options.AudioLayers <= 0 || options.AudioLayers % 2 != 0
            || options.DurationLayers <= 0 || options.DurationLayers % 2 != 0)
            throw new ArgumentException("Voicebox's symmetric U-Net skips require positive even layer counts.", nameof(options));
        if (options.AudioModelDim <= 0 || options.AudioModelDim % options.AudioHeads != 0
            || options.DurationModelDim <= 0 || options.DurationModelDim % options.DurationHeads != 0)
            throw new ArgumentException("Each Voicebox model width must divide evenly among its attention heads.", nameof(options));
        if (options.AudioModelDim % options.AudioConvPositionGroups != 0
            || options.DurationModelDim % options.DurationConvPositionGroups != 0)
            throw new ArgumentException("Voicebox convolutional-position groups must divide their model widths.", nameof(options));
        if (options.AudioConvPositionWidth % 2 == 0 || options.DurationConvPositionWidth % 2 == 0)
            throw new ArgumentException("Voicebox convolutional-position widths must be odd.", nameof(options));
        if (options.SigmaMinimum <= 0 || options.SigmaMinimum >= 1 || options.OdeSteps <= 0)
            throw new ArgumentException("Voicebox needs sigma_min in (0,1) and at least one ODE step.", nameof(options));
        if (options.MelStandardDeviation <= 0 || options.MaximumAudioFrames <= 0 || options.MaximumPhoneDuration <= 0)
            throw new ArgumentException("Voicebox scale and length limits must be positive.", nameof(options));
        if (options.VocoderUpsampleRates.Length != options.VocoderUpsampleKernels.Length
            || options.VocoderUpsampleRates.Aggregate(1, (a, b) => a * b) != options.HopSize)
            throw new ArgumentException("Voicebox's vocoder stages must match and multiply to the 160-sample hop.", nameof(options));
        if (options.VocoderResBlockKernels.Length != options.VocoderResBlockDilations.Length)
            throw new ArgumentException("Every Voicebox vocoder residual kernel needs its dilation list.", nameof(options));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Voicebox<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
