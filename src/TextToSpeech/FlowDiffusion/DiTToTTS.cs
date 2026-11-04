using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// DiTTo-TTS: scalable zero-shot text-to-speech with a diffusion Transformer in CLaM-TTS Mel-VAE space.
/// </summary>
/// <remarks>
/// This is the paper pipeline rather than the former generic layer stack: frozen SpeechT5 text features, a separately
/// trained causal length predictor, 9-way global AdaLN DiT with v-prediction and classifier-free guidance, the frozen
/// CLaM-TTS Mel-VAE/RVQ, and BigVGAN. The authors did not release code or checkpoints; native checkpoint loading therefore
/// uses the explicit AiDotNet conversion names documented by <see cref="LoadConvertedWeights"/>.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiTTo-TTS: Diffusion Transformers for Scalable Text-to-Speech without Domain-Specific Factors", "https://arxiv.org/abs/2406.11427", Year = 2024, Authors = "Lee et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999,
    WeightDecay = 0, WarmupSteps = 1000, MinLearningRate = 0, Schedule = LearningRateSchedulerType.CosineAnnealing,
    Source = "Lee et al. 2024 §4: AdamW, 1e-4, betas (0.9, 0.999), zero weight decay, cosine schedule and 1K warmup.")]
public partial class DiTToTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly DiTToTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly HashSet<LayerBase<T>> _frozen = new();
    private readonly HashSet<LayerBase<T>> _codecLayers = new();
    private bool _useNativeMode;
    private bool _disposed;
    private SentencePieceModel? _tokenizer;
    private CenteredLogMel<T>? _mel;
    private Random _trainingRandom;

    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, DiTToTTSOptions? options = null) : base(architecture)
    {
        _options = options is null ? new DiTToTTSOptions() : new DiTToTTSOptions(options); Options = _options;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 1);
        _useNativeMode = false; ApplyBaseOptions();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, DiTToTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options is null ? new DiTToTTSOptions() : new DiTToTTSOptions(options); Options = _options;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 1);
        _useNativeMode = true; ApplyBaseOptions(); InitializePaperModel();
        if (IsCodecStage) EnsureCodecTraining();
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    private void ApplyBaseOptions()
    {
        base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize; base.HiddenDim = _options.DiTHiddenDim;
    }

    private void InitializePaperModel()
    {
        if (Architecture.Layers is { Count: > 0 }) { Layers.AddRange(Architecture.Layers); return; }
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);

        var textLayers = new List<LayerBase<T>>();
        TextEncoder = new DiTToTextEncoder<T>(Engine, textLayers, _options, random); Layers.AddRange(textLayers); Freeze(textLayers);
        var codecLayers = new List<LayerBase<T>>();
        MelVae = new DiTToMelVae<T>(Engine, codecLayers, _options, random); Layers.AddRange(codecLayers); Freeze(codecLayers);
        foreach (var layer in codecLayers) _codecLayers.Add(layer);
        var diffusionLayers = new List<LayerBase<T>>();
        DiffusionTransformer = new DiTToDiffusionTransformer<T>(Engine, diffusionLayers, _options, random); Layers.AddRange(diffusionLayers);
        var lengthLayers = new List<LayerBase<T>>();
        LengthPredictor = new DiTToLengthPredictor<T>(Engine, lengthLayers, _options, random); Layers.AddRange(lengthLayers);
        BigVgan = new BigVganGenerator<T>(Engine, random, _options.MelChannels, _options.BigVganInitialChannels,
            _options.BigVganUpsampleRates, _options.BigVganUpsampleKernels, _options.BigVganResidualKernels,
            _options.BigVganResidualKernels.Select(_ => (int[])_options.BigVganResidualDilations.Clone()).ToArray(), snakeBeta: true);
        Layers.AddRange(BigVgan.Layers); Freeze(BigVgan.Layers);
    }

    private void Freeze(IEnumerable<LayerBase<T>> layers) { foreach (var layer in layers) _frozen.Add(layer); }

    internal DiTToTextEncoder<T>? TextEncoder { get; private set; }
    internal DiTToMelVae<T>? MelVae { get; private set; }
    internal DiTToDiffusionTransformer<T>? DiffusionTransformer { get; private set; }
    internal DiTToLengthPredictor<T>? LengthPredictor { get; private set; }
    internal BigVganGenerator<T>? BigVgan { get; private set; }
    internal bool HasPaperModel => TextEncoder is not null && MelVae is not null && DiffusionTransformer is not null && LengthPredictor is not null && BigVgan is not null;

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.DiTHiddenDim;
    public int NumFlowSteps => _options.InferenceSteps;
    public DiTToTrainingStage CurrentStage
    {
        get => _options.TrainingStage;
        set
        {
            if (!Enum.IsDefined(typeof(DiTToTrainingStage), value)) throw new ArgumentOutOfRangeException(nameof(value));
            _options.TrainingStage = value;
            if (IsCodecStage && HasPaperModel) EnsureCodecTraining();
        }
    }
    public override bool TrainsOnSampleObjective => true;
    protected override TtsSupervision RequiredSupervision => CurrentStage switch
    {
        // CodecObjective checks recording-or-mel itself; pretraining is unsupervised.
        DiTToTrainingStage.MelVaePretraining => TtsSupervision.None,
        DiTToTrainingStage.MelVaeSemanticFinetuning => TtsSupervision.Text,
        _ => TtsSupervision.Text | TtsSupervision.Recording
    };
    protected override int TargetFftSize => _options.FftSize;
    protected override bool SupportsParameterMutation => _useNativeMode;

    protected override ParameterSlotRole ResolveLayerParameterRole(int layerIndex, ParameterSlotRole declared)
    {
        if (IsCodecStage && layerIndex >= 0 && layerIndex < Layers.Count)
        {
            var layer = Layers[layerIndex];
            bool selected = _codecDiscriminatorTurn
                ? _codecTrainer?.Critics.Layers.Contains(layer) == true
                : _codecLayers.Contains(layer) || (CurrentStage == DiTToTrainingStage.MelVaeSemanticFinetuning
                    && _codecTrainer?.ProjectionLayers.Contains(layer) == true);
            return selected ? declared : ParameterSlotRole.Frozen;
        }
        if (_useNativeMode && layerIndex >= 0 && layerIndex < Layers.Count && _frozen.Contains(Layers[layerIndex])) return ParameterSlotRole.Frozen;
        if (HasPaperModel)
        {
            bool lengthLayer = LengthPredictor is not null && IsOwnedBy(LengthPredictor, Layers[layerIndex]);
            if (CurrentStage == DiTToTrainingStage.Diffusion && lengthLayer) return ParameterSlotRole.Frozen;
            if (CurrentStage == DiTToTrainingStage.LengthPredictor && !lengthLayer && !_frozen.Contains(Layers[layerIndex])) return ParameterSlotRole.Frozen;
        }
        return declared;
    }

    private static bool IsOwnedBy(DiTToLengthPredictor<T> predictor, object layer) =>
        ReferenceEquals(layer, predictor.AudioInput) || ReferenceEquals(layer, predictor.TextInput) || ReferenceEquals(layer, predictor.Bos)
        || ReferenceEquals(layer, predictor.Classifier) || predictor.Blocks.Any(b =>
            ReferenceEquals(layer, b.SelfQuery) || ReferenceEquals(layer, b.SelfKey) || ReferenceEquals(layer, b.SelfValue)
            || ReferenceEquals(layer, b.SelfOutput) || ReferenceEquals(layer, b.CrossQuery) || ReferenceEquals(layer, b.CrossKey)
            || ReferenceEquals(layer, b.CrossValue) || ReferenceEquals(layer, b.CrossOutput) || ReferenceEquals(layer, b.SelfNorm)
            || ReferenceEquals(layer, b.CrossNorm) || ReferenceEquals(layer, b.FeedForwardNorm)
            || ReferenceEquals(layer, b.FeedForward1) || ReferenceEquals(layer, b.FeedForward2));

    private static readonly Lazy<SentencePieceModel> CharacterModel = new(() =>
    {
        const string resource = "AiDotNet.TextToSpeech.MultiModal.speecht5_spm_char.model.gz";
        using var stream = typeof(DiTToTTS<T>).Assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidDataException($"The embedded resource {resource} is missing.");
        return SentencePieceModel.FromGzip(stream);
    });

    internal SentencePieceModel Tokenizer => _tokenizer ??= _options.SentencePieceModelPath is { } path
        ? new SentencePieceModel(File.ReadAllBytes(path)) : CharacterModel.Value;

    protected override Tensor<T> PreprocessText(string text) => IndexTtsOps.Ids<T>(TokenizeText(text));

    public int[] TokenizeText(string text)
    {
        Validation.Guard.NotNull(text);
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        return Tokenizer.Encode(text).Take(Math.Max(0, _options.MaxTextLength - 1)).Concat(new[] { _options.TextEosId }).ToArray();
    }

    public Tensor<T> ComputeMel(Tensor<T> audio)
    {
        _mel ??= new CenteredLogMel<T>(Engine, _options.SampleRate, _options.FftSize, _options.HopSize, _options.FftSize,
            _options.MelChannels, 0, 8000, 1e-5, htkScale: true, naturalLog: true);
        var mel = _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        return Engine.TensorTranspose(Engine.Reshape(mel, new[] { _options.MelChannels, mel.Shape[2] }));
    }

    /// <summary>Builds an optional zero-shot acoustic prompt. When its transcript is supplied, synthesis prepends it to the requested new text before text encoding.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? transcript = null)
    {
        Validation.Guard.NotNull(recording);
        var audio = sampleRate == _options.SampleRate ? Engine.Reshape(recording, new[] { recording.Length })
            : AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(recording, new[] { recording.Length }), sampleRate, _options.SampleRate);
        return new TtsVoice<T> { ReferenceAudio = audio, Reference = ComputeMel(audio),
            ReferenceTokens = transcript is null ? null : IndexTtsOps.Ids<T>(TokenizeText(transcript)) };
    }

    internal int[] TextWithPrompt(string text)
    {
        var target = TokenizeText(text);
        if (Voice?.Reference is null || Voice.ReferenceTokens is not { } reference) return target;
        var prompt = FrontEnd.TtsTokenIds.Read(reference, _options.TextVocabulary, _options.MaxTextLength,
            "DiTTo prompt transcript");
        int count = prompt.Length;
        while (count > 0 && (prompt[count - 1] == _options.TextEosId || prompt[count - 1] == _options.TextPadId)) count--;
        return prompt.Take(count).Concat(target.Take(target.Length - 1))
            .Take(Math.Max(0, _options.MaxTextLength - 1)).Concat(new[] { _options.TextEosId }).ToArray();
    }

    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(IndexTtsOps.Ids<T>(TokenizeText(text)));
        if (!HasPaperModel) return PostprocessAudio(Predict(PreprocessText(text)));
        RequirePaperModel();
        Tensor<T> encoded;
        using (new NoGradScope<T>()) encoded = TextEncoder!.Forward(TextWithPrompt(text));
        return SynthesizeFromEncoding(encoded, Voice?.Reference is { } promptMel ? MelVae!.Encode(promptMel) : null);
    }

    internal Tensor<T> SynthesizeFromEncoding(Tensor<T> text, Tensor<T>? prompt)
    {
        var latent = GenerateLatentsFromEncoding(text, prompt);
        if (latent.Shape[0] == 0) return new Tensor<T>(new[] { 0 });
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = MelVae!.Decode(latent);
        var channels = Engine.TensorPermute(Engine.Reshape(mel, new[] { 1, mel.Shape[0], mel.Shape[1] }), new[] { 0, 2, 1 });
        var waveform = BigVgan!.Forward(channels);
        return Engine.Reshape(waveform, new[] { waveform.Length });
    }

    internal Tensor<T> GenerateLatentsFromEncoding(Tensor<T> text, Tensor<T>? prompt, Random? random = null)
    {
        int promptLength = prompt?.Shape[0] ?? 0;
        int remaining = LengthPredictor!.ExpectedRemaining(text, prompt);
        int total = checked(promptLength + remaining);
        if (total == 0) return new Tensor<T>(new[] { 0, _options.LatentDim });
        var latent = Gaussian(total, _options.LatentDim, random ?? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        var mask = new Tensor<T>(new[] { total, 1 });
        for (int i = promptLength; i < total; i++) mask[i, 0] = Ops.One;
        if (prompt is not null) latent = RestorePrompt(latent, prompt);
        for (int step = _options.InferenceSteps; step > 0; step--)
        {
            double time = step / (double)_options.InferenceSteps, next = (step - 1) / (double)_options.InferenceSteps;
            var conditional = DiffusionTransformer!.Forward(latent, mask, text, time, false);
            var unconditional = DiffusionTransformer.Forward(latent, mask, text, time, true);
            var velocity = Engine.TensorAdd(unconditional, Engine.TensorMultiplyScalar(Engine.TensorSubtract(conditional, unconditional), Ops.FromDouble(_options.GuidanceScale)));
            var (signal, noise) = NoiseLevel(time); var (nextSignal, nextNoise) = NoiseLevel(next);
            var x0 = Engine.TensorSubtract(Engine.TensorMultiplyScalar(latent, Ops.FromDouble(signal)), Engine.TensorMultiplyScalar(velocity, Ops.FromDouble(noise)));
            var epsilon = Engine.TensorAdd(Engine.TensorMultiplyScalar(latent, Ops.FromDouble(noise)), Engine.TensorMultiplyScalar(velocity, Ops.FromDouble(signal)));
            latent = Engine.TensorAdd(Engine.TensorMultiplyScalar(x0, Ops.FromDouble(nextSignal)), Engine.TensorMultiplyScalar(epsilon, Ops.FromDouble(nextNoise)));
            if (prompt is not null) latent = RestorePrompt(latent, prompt);
        }
        return latent;
    }

    private Tensor<T> RestorePrompt(Tensor<T> latent, Tensor<T> prompt)
    {
        if (prompt.Shape[0] == 0) return latent;
        int rest = latent.Shape[0] - prompt.Shape[0];
        return rest == 0 ? prompt : Engine.TensorConcatenate(new[] { prompt,
            Engine.TensorSlice(latent, new[] { prompt.Shape[0], 0 }, new[] { rest, latent.Shape[1] }) }, 0);
    }

    internal (double Signal, double Noise) NoiseLevel(double time)
    {
        double cosinePower = Math.Pow(Math.Cos(MathPolyfill.Clamp(time, 0, 1) * Math.PI / 2), 2);
        cosinePower = MathPolyfill.Clamp(cosinePower, 1e-9, 1 - 1e-9);
        double logSnr = Math.Log(cosinePower) - Math.Log(1 - cosinePower) + 2 * Math.Log(_options.NoiseScheduleScale);
        double signalPower = 1 / (1 + Math.Exp(-logSnr));
        return (Math.Sqrt(signalPower), Math.Sqrt(1 - signalPower));
    }

    private Tensor<T> Gaussian(int rows, int columns, Random random)
    {
        var result = new Tensor<T>(new[] { rows, columns });
        for (int i = 0; i < result.Length; i++)
        {
            double u1 = 1 - random.NextDouble(), u2 = random.NextDouble();
            result.SetFlat(i, Ops.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)));
        }
        return result;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperModel)
        {
            SetTrainingMode(false);
            var output = input;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }
        var ids = FrontEnd.TtsTokenIds.Read(input, _options.TextVocabulary, _options.MaxTextLength, "DiTTo SpeechT5");
        Tensor<T> encoded; using (new NoGradScope<T>()) encoded = TextEncoder!.Forward(ids);
        return SynthesizeFromEncoding(encoded, null);
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is { Count: > 0 } && Layers.Count == 0) Layers.AddRange(Architecture.Layers);
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "DiTTo-TTS-Native" : "DiTTo-TTS-ONNX",
        Description = "DiTTo-TTS SpeechT5 + CLaM Mel-VAE + length predictor + v-prediction DiT + BigVGAN",
        FeatureCount = _options.DiTHiddenDim,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["ModelType"] = "DiTTo-TTS", ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            ["LatentDim"] = _options.LatentDim, ["LatentRateHz"] = _options.SampleRate / (double)(_options.HopSize * _options.MelVaeDownsampleFactor),
            ["DiTHiddenDim"] = _options.DiTHiddenDim, ["DiTLayers"] = _options.DiTLayers, ["DiTHeads"] = _options.DiTHeads,
            ["AdaLNParametersPerBlock"] = 9 * _options.DiTHiddenDim, ["DiffusionSteps"] = _options.InferenceSteps,
            ["GuidanceScale"] = _options.GuidanceScale, ["SampleRate"] = _options.SampleRate
        }
    };

    private void RequirePaperModel()
    {
        if (!HasPaperModel) throw new NotSupportedException("This operation needs DiTTo's native paper modules, not caller-supplied layers.");
    }

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DiTToTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
