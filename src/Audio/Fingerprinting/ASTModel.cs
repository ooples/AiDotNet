using AiDotNet.Attributes;
using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// AST (Audio Spectrogram Transformer, Gong et al. 2021) as an audio fingerprinter: the clip's AST representation is
/// its fingerprint, compared by cosine similarity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This runs the same network as <see cref="AST{T}"/> (<see cref="AstNetwork{T}"/>, the official code's model): a
/// 128-bin Kaldi filterbank of the waveform, 16×16 patches with an overlap of 6, DeiT's class and distillation tokens,
/// 12 pre-norm Transformer blocks, and the average of the two tokens as the clip representation. The fingerprint is
/// that representation; <see cref="Classify"/> scores the labels with the paper's sigmoid head, and training is the
/// paper's AudioSet recipe.
/// </para>
/// <para>
/// <b>Reference:</b> Gong, Y., Chung, Y.-A. &amp; Glass, J. (2021),
/// "AST: Audio Spectrogram Transformer", Interspeech.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "AST: Audio Spectrogram Transformer",
    "https://arxiv.org/abs/2104.01778",
    Year = 2021,
    Authors = "Yuan Gong, Yu-An Chung, James Glass")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-5, Beta1 = 0.95, Beta2 = 0.999, WeightDecay = 5e-7, ReferenceBatchSize = 12,
                WarmupSteps = 1000, Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.5, MilestoneFractions = [0.4, 0.6, 0.8],
                Source = "Gong et al. 2021, Sec. 3.1.1 (full AudioSet): batch 12, Adam, binary cross-entropy, initial rate 1e-5 for 5 epochs, halved every epoch after the 2nd (milestones at 2/5, 3/5 and 4/5 of the run). The official recipe (YuanGongND/ast egs/audioset/run.sh, traintest.py) fills in betas (0.95, 0.999), weight decay 5e-7 and a linear warmup over the first 1000 steps.")]
[TaskVariantOf(typeof(AST<>), "audio fingerprinting with the clip representation")]
public partial class ASTModel<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    /// <inheritdoc />
    /// <remarks>Predict returns one sigmoid probability per label.</remarks>
    protected override int OutputFeatureWidth => _labels.Length;

    private readonly ASTOptions _options;
    private readonly string[] _labels;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private AstPipeline<T>? _pipeline;
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    /// <summary>
    /// Path to the loaded ONNX model file when constructed in ONNX mode;
    /// <c>null</c> in native mode. Captured so a clone can reconstruct an ONNX model without losing its execution-mode
    /// configuration.
    /// </summary>
    private readonly string? _modelPath;

    /// <inheritdoc/>
    public string Name => _useNativeMode ? "AST-Native" : "AST-ONNX";

    /// <inheritdoc/>
    public int FingerprintLength => _options.EmbeddingDim;

    /// <summary>The labels <see cref="Classify"/> names (<see cref="ASTOptions.CustomLabels"/>, else AudioSet's 527).</summary>
    public IReadOnlyList<string> Labels => _labels;

    #region Constructors

    /// <summary>Initializes AST in ONNX inference mode, for a graph returning the clip representation
    /// <c>[batch, EmbeddingDim]</c> from the official input <c>[batch, TargetLength, NumMels]</c>.</summary>
    public ASTModel(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ASTOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path is required for ONNX mode.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new ASTOptions();
        Options = _options;
        _labels = _options.CustomLabels ?? AST<T>.AudioSetLabels;
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;
        _useNativeMode = false;
        _modelPath = modelPath;
        _pipeline = new AstPipeline<T>(Engine, _options, _labels.Length, layers: null);
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Initializes AST in native training / inference mode.</summary>
    public ASTModel(
        NeuralNetworkArchitecture<T> architecture,
        ASTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new ASTOptions();
        Options = _options;
        _labels = _options.CustomLabels ?? AST<T>.AudioSetLabels;
        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;
        _useNativeMode = true;
        InitializeLayers();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("ASTModel declares its paper recipe.");
    }

    #endregion

    /// <summary>The network, the same one <see cref="AST{T}"/> runs.</summary>
    internal AstNetwork<T> Network => Pipeline.Network;

    private AstPipeline<T> Pipeline => _pipeline ?? throw new InvalidOperationException("The AST front end is not initialized.");

    /// <summary>Loads an official AST checkpoint's state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode || _options is null || _pipeline is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("ASTModel builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _pipeline = new AstPipeline<T>(Engine, _options, _labels.Length, layers);
        int seed = _options.Seed ?? 42;
        _pipeline.Network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Layers.AddRange(layers);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    #endregion

    #region Preprocessing

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => Pipeline.Features(rawAudio);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => Engine.Sigmoid(modelOutput);

    #endregion

    #region Public API

    /// <summary>The per-label probabilities <c>[batch, labels]</c> of waveforms <c>[samples]</c> or <c>[batch, samples]</c>.</summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (!_useNativeMode)
            throw new NotSupportedException("The ONNX graph returns the representation; use Fingerprint.");
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return PostprocessOutput(Network.Forward(PreprocessAudio(input)));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <inheritdoc />
    /// <remarks>The label logits of the waveforms.</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Network.Forward(PreprocessAudio(input));

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    /// <summary>The <paramref name="topK"/> most probable labels of the first clip, by the paper's independent sigmoid
    /// scores (AudioSet is multi-label).</summary>
    public List<(string Label, double Probability)> Classify(Tensor<T> audio, int topK = 5)
    {
        var probabilities = Predict(audio);
        int count = _labels.Length;
        var row = probabilities.ToArray().Take(count).Select((p, i) => (Label: _labels[i], Probability: NumOps.ToDouble(p)));
        return row.OrderByDescending(r => r.Probability).Take(Math.Min(topK, count)).ToList();
    }

    /// <inheritdoc/>
    /// <remarks>The fingerprint is the clip's AST representation (the averaged class and distillation tokens).</remarks>
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> representation;
        if (!_useNativeMode && OnnxEncoder is not null)
        {
            representation = OnnxEncoder.Run(features);
        }
        else
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                representation = Network.Encode(features);
        }
        return new AudioFingerprint<T>
        {
            Data = representation.ToArray().Take(_options.EmbeddingDim).ToArray(),
            SampleRate = SampleRate,
            Duration = audio.Length / (double)SampleRate
        };
    }

    /// <inheritdoc/>
    public AudioFingerprint<T> Fingerprint(Vector<T> audio) =>
        Fingerprint(Tensor<T>.FromVector(audio));

    /// <inheritdoc/>
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        if (fp1.Data.Length != fp2.Data.Length)
            throw new ArgumentException("Fingerprint dimensions do not match.");
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int i = 0; i < fp1.Data.Length; i++)
        {
            double a = NumOps.ToDouble(fp1.Data[i]);
            double b = NumOps.ToDouble(fp2.Data[i]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        return dot / (Math.Sqrt(n1) * Math.Sqrt(n2) + 1e-12);
    }

    /// <inheritdoc/>
    public IReadOnlyList<FingerprintMatch> FindMatches(
        AudioFingerprint<T> query, AudioFingerprint<T> reference, int minMatchLength = 10)
    {
        if (minMatchLength <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(minMatchLength), minMatchLength, "minMatchLength must be > 0.");

        // Honor the caller's minimum-match guard: if the embedding overlap
        // is shorter than what they asked for, return no matches rather
        // than silently misreport a global-embedding similarity as a
        // valid match.
        int overlap = Math.Min(query.Data.Length, reference.Data.Length);
        if (overlap < minMatchLength)
            return Array.Empty<FingerprintMatch>();

        // AST produces global embeddings; whole-clip similarity only.
        double similarity = ComputeSimilarity(query, reference);
        if (similarity < 0.5) return Array.Empty<FingerprintMatch>();
        return new[]
        {
            new FingerprintMatch
            {
                QueryStartTime = 0.0,
                ReferenceStartTime = 0.0,
                Duration = Math.Max(query.Duration, reference.Duration),
                Confidence = similarity,
                MatchCount = Math.Min(query.Data.Length, reference.Data.Length)
            }
        };
    }

    /// <inheritdoc />
    protected override string? TrainingUnavailableReason => _useNativeMode ? null : OnnxTrainingUnavailableReason;

    /// <summary>
    /// One training step on waveforms <c>[batch, samples]</c> and multi-hot labels <c>[batch, labels]</c>, as the
    /// paper trains on AudioSet: mixup, SpecAugment and binary cross-entropy.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        var (features, targets) = Pipeline.TrainingBatch(input, expected, _trainingRandom);
        TrainWithCustomObjective(features, targets, (x, y) => Pipeline.Loss(Network.Forward(x), y), _optimizer);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    #endregion

    #region Helpers

    private bool _disposed;
    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) _disposed = true;
        base.Dispose(disposing);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() =>
        new ModelMetadata<T>
        {
            Name = Name,
            Description = "AST — Audio Spectrogram Transformer (Gong et al. 2021).",
            Complexity = _options.NumEncoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EmbeddingDim"] = _options.EmbeddingDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumAttentionHeads"] = _options.NumAttentionHeads,
                ["PatchSize"] = _options.PatchSize,
                ["FrequencyStride"] = _options.FrequencyStride,
                ["TimeStride"] = _options.TimeStride,
                ["NumMels"] = _options.NumMels,
                ["TargetLength"] = _options.TargetLength,
                ["NumClasses"] = _labels.Length,
                ["SampleRate"] = _options.SampleRate,
            }
        };

    #endregion
}
