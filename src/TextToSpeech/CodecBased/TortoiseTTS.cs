using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>TorToiSe: Better Speech Synthesis Through Scaling.</summary>
/// <remarks>
/// The native graph is the complete released pipeline: an 8,192-code VQ-VAE speech tokenizer, a separately positioned
/// GPT-2 speech prior, CLVP candidate ranking, latent-conditioned classifier-free diffusion, and UnivNet-c32. The
/// components are deliberately not collapsed into one generic transformer because they are trained independently and
/// perform different jobs in the paper.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Better Speech Synthesis Through Scaling",
    "https://arxiv.org/abs/2305.07243",
    Year = 2023,
    Authors = "Betker")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.96, WeightDecay = 0.01,
    WarmupSteps = 500, Schedule = LearningRateSchedulerType.LinearWarmup, ReferenceBatchSize = 1024,
    Source = "Betker 2023, App. B.2, Table 2: autoregressive prior AdamW, learning rate 1e-4, betas 0.9/0.96, "
        + "weight decay 0.01, 500 warmup steps and batch size 1024.")]
public partial class TortoiseTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly TortoiseTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly TortoiseMelFrontend<T>? _melFrontend;
    private TortoiseTextTokenizer? _textTokenizer;
    private TortoiseSpeechTokenizer<T>? _tokenizer;
    private TortoiseAutoregressive<T>? _autoregressive;
    private TortoiseClvp<T>? _clvp;
    internal TortoiseClvp<T>? ClvpModel => _clvp;
    private TortoiseDiffusion<T>? _diffusion;
    private TortoiseUnivNet<T>? _univNet;
    private TortoiseUnivNet<T> UnivNetVocoder => _univNet ?? throw new InvalidOperationException("The UnivNet vocoder is not initialized.");
    private UnivNetTrainingLoss<T>? _vocoderTraining;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _vocoderDiscriminatorOptimizer;
    private bool _vocoderDiscriminatorTurn;
    private long _vocoderTrainingSteps;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public TortoiseTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, TortoiseTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TortoiseTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Creates the native paper/release graph.</summary>
    public TortoiseTTS(
        NeuralNetworkArchitecture<T> architecture,
        TortoiseTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TortoiseTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? (_options.TrainingComponent == TortoiseTrainingComponent.UnivNet
            ? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                { InitialLearningRate = _options.VocoderTraining.LearningRate, Beta1 = .5, Beta2 = .9 })
            : null)
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _melFrontend = new TortoiseMelFrontend<T>(Engine, _options);
        ConfigureBaseProperties();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => 1;
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => 1;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override int OutputFeatureWidth => _options.CodebookSize + 2;
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Text | TtsSupervision.Recording;
    public override bool TrainsOnSampleObjective => true;

    /// <summary>Whether this instance owns all five native TorToiSe inference components.</summary>
    public bool HasPaperPipeline => _tokenizer is not null && _autoregressive is not null && _clvp is not null
        && _diffusion is not null && _univNet is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <summary>Creates a zero-shot voice from a recording and resamples it to TorToiSe's 24 kHz output rate.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate = 24000)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(recording);
        var audio = Resample(recording, sampleRate, _options.SampleRate);
        return new TtsVoice<T> { ReferenceAudio = audio, Reference = DiffusionMel(audio) };
    }

    /// <summary>Synthesizes with a recording without changing <see cref="TtsModelBase{T}.Voice"/>.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> recording, int sampleRate = 24000)
    {
        var previous = Voice;
        try
        {
            Voice = CreateVoice(recording, sampleRate);
            return Synthesize(text);
        }
        finally { Voice = previous; }
    }

    /// <summary>Runs autoregressive candidate generation, CLVP ranking, DDIM and UnivNet.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        RequirePipeline();
        var reference = RequireVoice().ReferenceAudio!;
        using var _ = new NoGradScope<T>();
        return SynthesizeIds(TextIds(text), reference);
    }

    /// <summary>Encodes 24 kHz audio to the paper's one-stream VQ-VAE speech codes.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio) => EncodeToTokens(audio, _options.SampleRate);

    /// <summary>Encodes audio at an explicitly supplied rate to VQ-VAE speech codes.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio, int sampleRate)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        var mel = AutoregressiveMel(Resample(audio, sampleRate, _options.TokenizerSampleRate));
        return TortoiseOps<T>.Ids(_tokenizer!.Encode(mel));
    }

    /// <summary>Decodes VQ-VAE codes through its mel decoder and the released UnivNet vocoder.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        var mel80 = _tokenizer!.Decode(TortoiseOps<T>.IdsOf(tokens));
        return UnivNetVocoder.Generate(ToVocoderMel(mel80), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
    }

    /// <summary>Returns CLVP's scalar compatibility score for a candidate speech-code sequence.</summary>
    public double ScoreCandidate(Tensor<T> textTokens, Tensor<T> speechTokens)
    {
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return _clvp!.Score(TextIds(textTokens), TortoiseOps<T>.IdsOf(speechTokens));
    }

    /// <summary>Trains CLVP with the released symmetric in-batch contrastive objective.</summary>
    public T TrainClvpBatch(IReadOnlyList<TtsTrainingSample<T>> samples)
    {
        RequirePipeline();
        if (_options.TrainingComponent != TortoiseTrainingComponent.Clvp)
            throw new InvalidOperationException("Select the CLVP training component before training a contrastive batch.");
        if (samples is null || samples.Count == 0) throw new ArgumentException("A nonempty CLVP batch is required.", nameof(samples));
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        var texts = samples.Select(s => (IReadOnlyList<int>)TextIds(s.Tokens)).ToArray();
        var speech = samples.Select(s => (IReadOnlyList<int>)TortoiseOps<T>.IdsOf(s.CodecTokens is { } storedCodecTokens ? storedCodecTokens
            : EncodeToTokens(s.Audio ?? throw new ArgumentException("Every CLVP example requires CodecTokens or Audio.")))).ToArray();
        return TrainWithCustomObjective(samples[0].Tokens, samples[0].Tokens,
            (_, _) => (_clvp ?? throw new InvalidOperationException("CLVP is not initialized.")).ContrastiveBatchLoss(texts, speech, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 17)), _optimizer);
    }

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        if (_options.TrainingComponent == TortoiseTrainingComponent.UnivNet) return TrainVocoder(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample), _optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample)[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _tokenizer = new TortoiseSpeechTokenizer<T>(Engine, ComponentLayers, _options);
        _autoregressive = new TortoiseAutoregressive<T>(Engine, ComponentLayers, _options);
        _clvp = new TortoiseClvp<T>(Engine, ComponentLayers, _options);
        _diffusion = new TortoiseDiffusion<T>(Engine, ComponentLayers, _options);
        _univNet = new TortoiseUnivNet<T>(Engine, ComponentLayers, _options);
        if (_options.TrainingComponent == TortoiseTrainingComponent.UnivNet) EnsureVocoderTraining();
    }

    private void EnsureVocoderTraining()
    {
        if (_vocoderTraining is not null) return;
        _vocoderTraining = new UnivNetTrainingLoss<T>(Engine, _options.VocoderTraining);
        ComponentLayers.AddRange(_vocoderTraining.Layers);
        _vocoderDiscriminatorOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                { InitialLearningRate = _options.VocoderTraining.LearningRate, Beta1 = .5, Beta2 = .9 });
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_options.TrainingComponent != TortoiseTrainingComponent.UnivNet || _univNet is null) return parameters;
        EnsureVocoderTraining();
        var layers = _vocoderDiscriminatorTurn ? _vocoderTraining!.Layers : _univNet.Layers;
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList()),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private T TrainVocoder(TtsTrainingSample<T> sample)
    {
        EnsureVocoderTraining();
        var audio = sample.Audio ?? throw new ArgumentException("UnivNet training requires the recording.", nameof(sample));
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = DiffusionMel(audio);
        bool adversarial = _vocoderTrainingSteps >= _options.VocoderTraining.PretrainingSteps;
        Tensor<T>? savedGeneration = null;
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
        {
            var generated = UnivNetVocoder.Generate(mel, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + (int)(_vocoderTrainingSteps % int.MaxValue)));
            savedGeneration = new Tensor<T>(generated.Shape.ToArray(), new Vector<T>(generated.ToArray()));
            return _vocoderTraining!.Generator(audio, generated, adversarial);
        }, _optimizer!);
        if (adversarial)
        {
            _vocoderDiscriminatorTurn = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens,
                    (_, _) => _vocoderTraining!.Discriminator(audio, savedGeneration!), _vocoderDiscriminatorOptimizer!);
            }
            finally { _vocoderDiscriminatorTurn = false; }
        }
        _vocoderTrainingSteps++;
        return loss;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (HasPaperPipeline)
        {
            var reference = RequireVoice().ReferenceAudio!;
            using var _ = new NoGradScope<T>();
            return SynthesizeIds(TextIds(input), reference);
        }
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "TorToiSe-Native" : "TorToiSe-ONNX",
            Description = "TorToiSe GPT-2 prior, CLVP ranking, diffusion decoder and UnivNet-c32",
            FeatureCount = _options.LLMDim,
        };
        metadata.AdditionalInfo["Architecture"] = "VQ-VAE -> GPT-2 candidates -> CLVP -> spaced DDPM -> UnivNet-c32";
        metadata.AdditionalInfo["AutoregressiveLayers"] = _options.NumLLMLayers;
        metadata.AdditionalInfo["ClvpLayersPerTower"] = _options.ClvpLayers;
        metadata.AdditionalInfo["DiffusionLayers"] = _options.DiffusionLayers;
        metadata.AdditionalInfo["DiffusionSteps"] = _options.DiffusionInferenceSteps;
        metadata.AdditionalInfo["CodebookSize"] = _options.CodebookSize;
        metadata.AdditionalInfo["TrainingComponent"] = _options.TrainingComponent.ToString();
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = true;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private Tensor<T> SynthesizeIds(int[] text, Tensor<T> referenceAt24Khz)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var reference22 = Resample(referenceAt24Khz, _options.SampleRate, _options.TokenizerSampleRate);
        var arReference = PrefixOrPad(reference22, _options.ConditioningSamples);
        var condition = _autoregressive!.Conditioning(AutoregressiveMel(arReference), false, random);
        int count = Math.Max(1, _options.NumAutoregressiveSamples);
        int[]? best = null;
        double bestScore = double.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            var candidate = TortoiseCandidateTail.Prepare(_autoregressive.Generate(text, condition, random),
                Math.Min(_options.MaxCodecFrames, _options.MaximumAutoregressiveTokens),
                _options.SpeechStopToken, _options.CodebookSize);
            double score = _clvp!.Score(text, candidate);
            if (best is null || score > bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }
        var latents = _autoregressive.Latents(text, best!, condition);
        int latentLength = TortoiseCandidateTail.LatentLength(best!, _options.CodebookSize);
        latents = Engine.TensorSlice(latents, new[] { 0, 0 }, new[] { Math.Max(1, latentLength), _options.LLMDim });
        var diffusionReference = PrefixOrPad(referenceAt24Khz, _options.DiffusionConditioningSamples);
        var mel = _diffusion!.Sample(latents, DiffusionMel(diffusionReference), random);
        return _univNet!.Generate(mel, random);
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample)
    {
        var audio = sample.Audio ?? throw new ArgumentException("TorToiSe training requires Audio.", nameof(sample));
        if (_options.TrainingComponent == TortoiseTrainingComponent.UnivNet)
        {
            EnsureVocoderTraining();
            var generated = UnivNetVocoder.Generate(DiffusionMel(audio), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 17));
            return _vocoderTraining!.Generator(audio, generated, _vocoderTrainingSteps >= _options.VocoderTraining.PretrainingSteps);
        }
        int[] text = TextIds(sample.Tokens);
        int[] speech = sample.CodecTokens is not null
            ? TortoiseOps<T>.IdsOf(sample.CodecTokens)
            : TortoiseOps<T>.IdsOf(EncodeToTokens(audio));
        var reference = sample.SpeakerReference ?? audio;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 17);

        if (_options.TrainingComponent == TortoiseTrainingComponent.Clvp)
            return _clvp!.ContrastiveLoss(text, speech, random);

        var mel22 = AutoregressiveMel(Resample(audio, _options.SampleRate, _options.TokenizerSampleRate));
        if (_options.TrainingComponent == TortoiseTrainingComponent.SpeechTokenizer)
        {
            var reconstruction = _tokenizer!.Reconstruct(mel22);
            int frames = Math.Min(mel22.Shape[0], reconstruction.Shape[0]);
            var target = Engine.TensorSlice(mel22, new[] { 0, 0 }, new[] { frames, mel22.Shape[1] });
            var predicted = Engine.TensorSlice(reconstruction, new[] { 0, 0 }, new[] { frames, reconstruction.Shape[1] });
            var loss = TortoiseOps<T>.Mse(Engine, predicted, target);
            if (_tokenizer.Quantizer.CommitmentLoss is not null)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(_tokenizer.Quantizer.CommitmentLoss,
                    TortoiseOps<T>.NumOps.FromDouble(_options.TokenizerCommitmentWeight)));
            return loss;
        }

        Tensor<T> condition;
        using (new NoGradScope<T>())
        {
            var reference22 = Resample(reference, _options.SampleRate, _options.TokenizerSampleRate);
            condition = _autoregressive!.Conditioning(AutoregressiveMel(
                PrefixOrPad(reference22, _options.ConditioningSamples)), false, random);
        }
        if (_options.TrainingComponent == TortoiseTrainingComponent.Autoregressive)
            return _autoregressive!.Loss(text, speech, condition, true, random);

        if (_options.TrainingComponent == TortoiseTrainingComponent.Diffusion)
        {
            Tensor<T> latents;
            using (new NoGradScope<T>()) latents = _autoregressive!.Latents(text, speech, condition);
            var target = DiffusionMel(audio);
            var speaker = DiffusionMel(PrefixOrPad(reference, _options.DiffusionConditioningSamples));
            return _diffusion!.TrainingLoss(target, latents, speaker, random);
        }

        throw new InvalidOperationException("Unknown TorToiSe training component.");
    }

    private Tensor<T> AutoregressiveMel(Tensor<T> audio)
    {
        if (_melFrontend is null) throw new InvalidOperationException("The native mel front end is unavailable.");
        return _melFrontend.Autoregressive(EnsureLength(audio, _options.AutoregressiveFftSize));
    }

    private Tensor<T> DiffusionMel(Tensor<T> audio)
    {
        if (_melFrontend is null) throw new InvalidOperationException("The native mel front end is unavailable.");
        return _melFrontend.Diffusion(EnsureLength(audio, _options.FftSize));
    }

    private Tensor<T> ToVocoderMel(Tensor<T> mel80)
    {
        if (mel80.Shape[1] == _options.MelChannels) return mel80;
        int frames = mel80.Shape[0], bins = mel80.Shape[1], channels = _options.MelChannels;
        var kept = Engine.TensorSlice(mel80, new[] { 0, 0 }, new[] { frames, Math.Min(bins, channels) });
        if (bins >= channels) return kept.Clone();
        // Bins past the front end's are log(1e-5), the floor of an empty mel bin.
        var floor = Tensor<T>.CreateDefault(new[] { frames, channels - bins }, TortoiseOps<T>.NumOps.FromDouble(Math.Log(1e-5)));
        return Engine.TensorConcatenate(new[] { kept, floor }, 1);
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : AudioHelper<T>.ResampleBandLimited(Engine, flat, from, to);
    }

    private Tensor<T> PrefixOrPad(Tensor<T> audio, int samples)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        if (flat.Length >= samples) return Engine.TensorSlice(flat, new[] { 0 }, new[] { samples });
        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { samples - flat.Length }) }, 0);
    }

    private Tensor<T> EnsureLength(Tensor<T> audio, int samples)
        => audio.Length >= samples ? Engine.Reshape(audio, new[] { audio.Length }) : PrefixOrPad(audio, samples);

    /// <summary>Loads the original Whitespace/character-BPE tokenizer for a checkpoint.</summary>
    public void LoadTextTokenizer(string path)
    {
        ThrowIfDisposed();
        var tokenizer = new TortoiseTextTokenizer(File.ReadAllText(path));
        if (tokenizer.VocabularySize != _options.VocabSize - 1)
            throw new InvalidDataException("The tokenizer vocabulary must match the text embedding rows excluding the start token.");
        _textTokenizer = tokenizer;
        _options.TextTokenizerPath = path;
    }

    /// <summary>Runs the released English cleaner and exact BPE vocabulary.</summary>
    public Tensor<T> TokenizeText(string text)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        var ids = (_textTokenizer ?? TortoiseTextTokenizer.Default.Value).Encode(text);
        return TortoiseOps<T>.Ids(TextIds(TortoiseOps<T>.Ids(ids)));
    }

    protected override Tensor<T> PreprocessText(string text) => TokenizeText(text);

    private int[] TextIds(string text) => TextIds(PreprocessText(text));

    private int[] TextIds(Tensor<T> tokens)
    {
        var ids = TortoiseOps<T>.IdsOf(tokens);
        if (ids.Length == 0 || ids.Length > _options.MaxTextLength || ids.Any(id => id < 0 || id >= _options.VocabSize))
            throw new ArgumentException("Text tokens must fit the configured vocabulary and maximum text length.", nameof(tokens));
        return ids;
    }

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (!string.IsNullOrWhiteSpace(_options.TextTokenizerPath)) LoadTextTokenizer(_options.TextTokenizerPath!);
    }

    private static void ValidateOptions(TortoiseTTSOptions options)
    {
        if (options.NumCodebooks != 1 || options.CodebookSize <= 0)
            throw new ArgumentException("TorToiSe uses one positive-sized VQ-VAE codebook.");
        if (options.SpeechStartToken != options.CodebookSize || options.SpeechStopToken != options.CodebookSize + 1)
            throw new ArgumentException("TorToiSe speech start/stop tokens must immediately follow the VQ codebook.");
        if (options.LLMDim <= 0 || options.NumHeads <= 0 || options.LLMDim % options.NumHeads != 0)
            throw new ArgumentException("The autoregressive width must be divisible by its attention heads.");
        if (options.ClvpDim <= 0 || options.ClvpHeads <= 0 || options.ClvpDim % options.ClvpHeads != 0)
            throw new ArgumentException("The CLVP width must be divisible by its attention heads.");
        if (options.DiffusionDim <= 0 || options.DiffusionHeads <= 0 || options.DiffusionDim % options.DiffusionHeads != 0)
            throw new ArgumentException("The diffusion width must be divisible by its attention heads.");
        if (options.TokenizerCompression != 4)
            throw new ArgumentException("The TorToiSe VQ-VAE has fixed four-times temporal compression.");
        if (options.UnivNetUpsampleRates.Aggregate(1, (product, rate) => product * rate) != options.HopSize)
            throw new ArgumentException("UnivNet's upsampling product must equal the output mel hop.");
        if (options.NumAutoregressiveSamples <= 0 || options.DiffusionInferenceSteps < 2 || options.DiffusionTrainingSteps <= 0)
            throw new ArgumentException("Candidate and diffusion step counts must be positive.");
        if (options.DiffusionTrainingSteps <= 20 || options.DiffusionInferenceSteps > options.DiffusionTrainingSteps)
            throw new ArgumentException("The rescaled linear beta schedule needs more than 20 training steps and no more sampling than training steps.");
        if (options.TopP <= 0 || options.TopP > 1 || options.Temperature <= 0 || options.RepetitionPenalty <= 0)
            throw new ArgumentException("TopP must be a probability and sampling temperature/penalty must be positive.");
        if (options.MaximumAutoregressiveTokens <= 0 || options.MinimumAutoregressiveTokens < 0
            || options.MinimumAutoregressiveTokens > Math.Min(options.MaxCodecFrames, options.MaximumAutoregressiveTokens))
            throw new ArgumentException("Autoregressive generation limits must satisfy 0 <= minimum <= maximum and fit the checkpoint capacity.");
    }

    private void RequirePipeline()
    {
        if (!HasPaperPipeline)
            throw new NotSupportedException("This operation requires the native TorToiSe paper pipeline.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TortoiseTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
