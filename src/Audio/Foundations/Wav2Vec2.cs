using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// wav2vec 2.0 self-supervised speech representation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
/// learning over quantized latent speech units. With just 10 minutes of labeled data, it
/// achieves WER 1.8% on LibriSpeech test-clean when fine-tuned for ASR. It pioneered the
/// self-supervised approach later extended by HuBERT and WavLM.
/// </para>
/// <para>
/// <b>For Beginners:</b> wav2vec 2.0 was the breakthrough that showed AI could learn to
/// understand speech with very little labeled data. It works by:
/// 1. Converting raw audio into features with a CNN
/// 2. Masking some features (hiding them)
/// 3. Learning to predict the masked parts from context
/// This is similar to how GPT predicts the next word, but for audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 768);
/// var model = new Wav2Vec2&lt;float&gt;(arch, "wav2vec2_base.onnx");
/// var embeddings = model.ExtractEmbeddings(audioWaveform);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "https://arxiv.org/abs/2006.11477", Year = 2020, Authors = "Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-4, Variant = "base",
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.08,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Baevski et al. 2020, pre-training setup: Adam, warmup over the first 8% of updates to a peak of 5e-4 for BASE and 3e-4 for LARGE, then linear decay. This row is the BASE peak rate.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-4, Variant = "large",
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.08,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Baevski et al. 2020, pre-training setup: Adam, warmup over the first 8% of updates to a peak of 5e-4 for BASE and 3e-4 for LARGE, then linear decay. This row is the LARGE peak rate.")]
public partial class Wav2Vec2<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>, IPaperOptimizerVariant,
    ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>Predict returns the encoder's contextual representations, <c>[frames, HiddenDim]</c>.</remarks>
    protected override int OutputFeatureWidth => _options.HiddenDim;

    #region Fields

    private readonly Wav2Vec2Options _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Wav2Vec2Network<T>? _network;
    private Wav2Vec2Pretraining<T>? _pretraining;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    /// <summary>The loss terms of the last pre-training step (contrastive, diversity, feature penalty).</summary>
    internal (double Contrastive, double Diversity, double FeaturePenalty) LastPretrainingLosses { get; private set; }

    #endregion

    #region IAudioFoundationModel Properties

    /// <inheritdoc />
    public int EmbeddingDimension => _options.HiddenDim;

    /// <inheritdoc />
    public int NumLayers => _options.NumLayers;

    #endregion

    #region Constructors

    public Wav2Vec2(NeuralNetworkArchitecture<T> architecture, string modelPath, Wav2Vec2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2Options();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public Wav2Vec2(NeuralNetworkArchitecture<T> architecture, Wav2Vec2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2Options();
        Options = _options;
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        { InitialLearningRate = _options.LearningRate });
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    /// <summary>The variant this instance was built at, keying the paper recipe for it.</summary>
    /// <remarks>
    /// The paper states a different peak rate for BASE and LARGE, so one recipe for the class
    /// would be wrong for one of them. Read during construction, which is safe because the
    /// options are assigned before the optimizer is built.
    /// </remarks>
    public string PaperOptimizerVariant => _options.Variant;

    internal static async Task<Wav2Vec2<T>> CreateAsync(Wav2Vec2Options? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new Wav2Vec2Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("wav2vec2", $"wav2vec2_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.HiddenDim);
        return new Wav2Vec2<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFoundationModel

    /// <inheritdoc />
    /// <remarks>The last encoder layer's representations, <c>[frames, HiddenDim]</c> for a waveform (batched input
    /// keeps its batch axis).</remarks>
    public Tensor<T> ExtractEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    public Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => ExtractEmbeddings(audio), cancellationToken);

    /// <inheritdoc />
    /// <remarks>Layer <c>i</c>'s output (negative indices count from the last).</remarks>
    public Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        int target = layerIndex < 0 ? _options.NumLayers + layerIndex : layerIndex;
        if (target < 0 || target >= _options.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index {layerIndex} is out of range. Valid range: [{-_options.NumLayers}, {_options.NumLayers - 1}].");
        return Unbatch(audio, Encode(audio).LayerStates[target + 1]);
    }

    /// <inheritdoc />
    /// <remarks>Σᵢ wᵢ·(layer i's output) over the encoder layers, equal weights when none are given.</remarks>
    public Tensor<T> ExtractWeightedFeatures(Tensor<T> audio, T[]? layerWeights = null)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        var states = Encode(audio).LayerStates;
        int count = states.Count - 1;
        if (layerWeights is not null && layerWeights.Length != count)
            throw new ArgumentException($"Give one weight per encoder layer ({count}).", nameof(layerWeights));
        Tensor<T>? sum = null;
        for (int i = 0; i < count; i++)
        {
            var term = Engine.TensorMultiplyScalar(states[i + 1], layerWeights is null ? NumOps.FromDouble(1.0 / count) : layerWeights[i]);
            sum = sum is null ? term : Engine.TensorAdd(sum, term);
        }
        return Unbatch(audio, sum!);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException("Wav2Vec2 builds the wav2vec 2.0 network; custom layers in the architecture are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new Wav2Vec2Network<T>(Engine, layers, _options.ToConfiguration());
        _pretraining = new Wav2Vec2Pretraining<T>(Engine, layers, _options.ToPretrainingConfiguration(), _options.FeatureEncoderDim, _options.HiddenDim);
        var seed = Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42);
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _network.InitializeLikeReference(seed);
        _pretraining.InitializeLikeReference(seed);
        Layers.AddRange(layers);
    }

    /// <summary>Loads a transformers <c>Wav2Vec2ForPreTraining</c> state dict (or a <c>Wav2Vec2Model</c>'s, whose keys
    /// have no <c>wav2vec2.</c> prefix and no quantizer).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, bool pretrainingCheckpoint = true)
    {
        if (_network is null || _pretraining is null) throw new InvalidOperationException("Weights load into the native network.");
        _network.LoadTorchWeights((name, shape) => read(pretrainingCheckpoint ? $"wav2vec2.{name}" : name, shape));
        if (pretrainingCheckpoint) _pretraining.LoadTorchWeights(read);
    }

    private Wav2Vec2Network<T>.Output Encode(Tensor<T> audio)
    {
        if (_network is null) throw new InvalidOperationException("The native network is not initialized.");
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return _network.Forward(audio, training: false, _random);
    }

    private Tensor<T> Unbatch(Tensor<T> audio, Tensor<T> x)
        => audio.Rank == 1 ? Engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }) : x;

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) throw new InvalidOperationException("The native network is not initialized.");
        return Unbatch(input, _network.Forward(input, training: false, _random).Hidden);
    }

    /// <inheritdoc />
    /// <remarks>The forward is not a fold of <c>Layers</c>; the activations are those the encoder computes.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => _useNativeMode ? ObservedLayerActivations(input) : new Dictionary<string, Tensor<T>>();

    /// <inheritdoc />
    /// <remarks>The encoder output of the training forward (masking, dropout, LayerDrop).</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_network is null) throw new InvalidOperationException("The native network is not initialized.");
        return Unbatch(input, _network.Forward(input, training: true, _random).Hidden);
    }

    /// <summary>
    /// One self-supervised pre-training step on waveforms <c>[B, S]</c> (or <c>[S]</c>): spans of the latent features
    /// are masked, and the model learns to identify each masked step's quantized latent among distractors (the
    /// contrastive loss), with the codebook diversity loss and the feature-encoder activation penalty (§3.2, §4.2).
    /// </summary>
    /// <param name="input">The waveforms.</param>
    /// <param name="expected">Unused: the objective is self-supervised.</param>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        if (_network is null || _pretraining is null) throw new InvalidOperationException("The native network is not initialized.");
        var network = _network;
        var pretraining = _pretraining;
        TrainWithCustomLoss(input, _ =>
        {
            var output = network.Forward(input, training: true, _random, maskProbability: pretraining.Configuration.MaskProbability,
                featureGradientScale: pretraining.Configuration.FeatureGradientScale);
            var negatives = pretraining.SampleNegatives(output.Mask!, _random);
            var losses = pretraining.Loss(output, negatives, training: true, _random);
            LastPretrainingLosses = (NumOps.ToDouble(losses.Contrastive[0]), NumOps.ToDouble(losses.Diversity[0]), NumOps.ToDouble(losses.FeaturePenalty[0]));
            return losses.Total;
        }, _optimizer);
        pretraining.AfterUpdate();
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.SelfSupervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The pre-training loss of the input (the target is not used), evaluated in inference mode with a mask
    /// and distractors drawn from a fixed seed, so it is a function of the input and the weights.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (_network is null || _pretraining is null) throw new InvalidOperationException("The native network is not initialized.");
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 2);
        var waveform = input.Rank == 1 ? Engine.Reshape(input, new[] { 1, input.Length })
            : input.Rank == 2 ? input : Engine.Reshape(input, new[] { input.Shape[0], input.Length / input.Shape[0] });
        var mask = _network.DrawMask(waveform.Shape[0], _network.Frames(waveform.Shape[1]), _pretraining.Configuration.MaskProbability, random);
        var output = _network.Forward(waveform, training: false, random, mask: mask);
        return _pretraining.Loss(output, _pretraining.SampleNegatives(mask, random), training: false, random).Total[0];
    }

    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "wav2vec2-Native" : "wav2vec2-ONNX",
            Description = $"wav2vec 2.0 {_options.Variant} self-supervised speech model (Baevski et al., 2020)",
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Wav2Vec2<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
