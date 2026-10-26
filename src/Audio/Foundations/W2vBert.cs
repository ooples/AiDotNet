using AiDotNet.Attributes;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// w2v-BERT: self-supervised speech representations from contrastive learning and masked prediction together
/// (Chung et al. 2021), and the semantic tokens AudioLM, SPEAR-TTS and SoundStorm build on.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A feature encoder turns 80-bin log-mel features into 25 frames a second. Spans of them are masked (replaced by random
/// vectors) and passed through the contrastive module, whose context vectors must pick each masked step's quantized
/// feature among distractors (wav2vec 2.0's task); the quantizer's choice for each step is its token id. The
/// masked-prediction module reads the context vectors and predicts the masked steps' token ids. Both modules train
/// together on L_c + L_m (§3.2). See <see cref="W2vBertOptions"/> for the sizes and the choices the paper leaves open.
/// </para>
/// <para>
/// <b>Semantic tokens</b> (AudioLM, Borsos et al. 2022, §4): the output of the masked-prediction module's 7th block,
/// standardized per dimension, clustered by k-means into 1024 tokens (<see cref="FitSemanticTokenizer"/>,
/// <see cref="SemanticTokens"/>).
/// </para>
/// <para><b>For Beginners:</b> w2v-BERT learns what speech sounds like from recordings alone. Its middle layers capture
/// what is being said, which is why speech generators use them as "semantic" tokens.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 1024);
/// var model = new W2vBert&lt;float&gt;(architecture, new W2vBertOptions());
/// foreach (var batch in unlabeledSpeech) model.Train(batch, batch);     // self-supervised pre-training
/// model.FitSemanticTokenizer(sampleRecordings);
/// int[] tokens = model.SemanticTokens(recording);                       // 25 tokens a second
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training",
    "https://arxiv.org/abs/2108.06209", Year = 2021, Authors = "Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, Yonghui Wu")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-3, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9, WarmupSteps = 25_000,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Chung et al. 2021, Section 4.2: w2v-BERT XL uses Adam with the Transformer learning-rate schedule of "
             + "Vaswani et al. 2017 Section 5.3 (beta 0.9 / 0.98, epsilon 1e-9, lr proportional to "
             + "min(step / warmup, sqrt(warmup / step))) at a peak of 2e-3 after 25k warm-up steps.")]
[PreprocessesInput("W2vBertNetwork.Features converts the waveform to log-mel filterbanks, and Encode feeds them to "
    + "the first convolution as a [B, 1, frames, bins] image, so Layers[0] never sees the raw samples.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)]
public partial class W2vBert<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly W2vBertOptions _options;
    private readonly bool _useNativeMode;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private W2vBertNetwork<T>? _network;
    private Wav2Vec2Pretraining<T>? _pretraining;
    private Random _random;

    // The fitted semantic tokenizer: per-dimension mean and standard deviation, and the k-means model.
    private double[]? _semanticMean;
    private double[]? _semanticScale;
    private KMeans<T>? _semanticClusters;

    /// <summary>The loss terms of the last pre-training step.</summary>
    internal (double Contrastive, double Diversity, double MaskedPrediction) LastPretrainingLosses { get; private set; }


    /// <inheritdoc />
    protected override int OutputFeatureWidth => _options.ModelDim;

    /// <inheritdoc />
    public int EmbeddingDimension => _options.ModelDim;

    /// <inheritdoc />
    public int NumLayers => _options.ContrastiveLayers + _options.MaskedLayers;

    /// <summary>Creates an ONNX-backed w2v-BERT for inference.</summary>
    public W2vBert(NeuralNetworkArchitecture<T> architecture, string modelPath, W2vBertOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new W2vBertOptions(); Options = _options;
        _useNativeMode = false;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable w2v-BERT.</summary>
    public W2vBert(NeuralNetworkArchitecture<T> architecture, W2vBertOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new W2vBertOptions(); Options = _options;
        _useNativeMode = true;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        _optimizer = optimizer;
        InitializeLayers();
    }

    internal W2vBertNetwork<T>? Network => _network;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException("W2vBert builds the w2v-BERT network; custom layers in the architecture are not supported.");
        var o = _options;
        var layers = new List<LayerBase<T>>();
        _network = new W2vBertNetwork<T>(Engine, layers, o);
        _pretraining = new Wav2Vec2Pretraining<T>(Engine, layers, new Wav2Vec2PretrainingConfiguration
        {
            Groups = 1,
            EntriesPerGroup = o.CodebookSize,
            CodevectorDim = o.CodeDim,
            ProjectedDim = o.CodeDim,
            Negatives = o.Negatives,
            DiversityWeight = o.DiversityWeight,
            MaskProbability = o.MaskProbability,
            // w2v-BERT has no feature penalty; its feature encoder is trained at the full rate.
            FeaturePenaltyWeight = 0.0,
            FeatureGradientScale = 1.0,
        }, o.ModelDim, o.ModelDim);
        _pretraining.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    private W2vBertNetwork<T> RequireNetwork() =>
        _network ?? throw new InvalidOperationException("The native network is not initialized.");

    private Wav2Vec2Pretraining<T> RequirePretraining() =>
        _pretraining ?? throw new InvalidOperationException("The native network is not initialized.");

    private W2vBertNetwork<T>.Output Encode(Tensor<T> audio)
    {
        var network = RequireNetwork();
        using var _ = new NoGradScope<T>();
        return network.Forward(network.Features(audio), mask: null, training: false, _random);
    }

    private Tensor<T> Unbatch(Tensor<T> audio, Tensor<T> x) =>
        audio.Rank == 1 ? Engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }) : x;

    // ---------------------------------------------------------------- representations

    /// <inheritdoc />
    /// <remarks>The masked-prediction module's output, <c>[frames, ModelDim]</c> at 25 frames a second.</remarks>
    public Tensor<T> ExtractEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractEmbeddings(audio), cancellationToken);

    /// <inheritdoc />
    /// <remarks>Block <c>i</c>'s output, the contrastive module's blocks first (negative indices count from the
    /// last).</remarks>
    public Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        int target = layerIndex < 0 ? NumLayers + layerIndex : layerIndex;
        if (target < 0 || target >= NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), $"Valid range: [{-NumLayers}, {NumLayers - 1}].");
        return Unbatch(audio, Encode(audio).BlockStates[target]);
    }

    /// <inheritdoc />
    /// <remarks>Σᵢ wᵢ·(block i's output), equal weights when none are given.</remarks>
    public Tensor<T> ExtractWeightedFeatures(Tensor<T> audio, T[]? layerWeights = null)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        var states = Encode(audio).BlockStates;
        if (layerWeights is not null && layerWeights.Length != states.Count)
            throw new ArgumentException($"Give one weight per block ({states.Count}).", nameof(layerWeights));
        Tensor<T>? sum = null;
        for (int i = 0; i < states.Count; i++)
        {
            var term = Engine.TensorMultiplyScalar(states[i], layerWeights is null ? NumOps.FromDouble(1.0 / states.Count) : layerWeights[i]);
            sum = sum is null ? term : Engine.TensorAdd(sum, term);
        }
        return Unbatch(audio, sum ?? throw new InvalidOperationException("The network has no blocks."));
    }

    // ---------------------------------------------------------------- semantic tokens (AudioLM §4)

    private double[,] SemanticEmbeddings(Tensor<T> recording)
    {
        if (_options.SemanticLayer < 1 || _options.SemanticLayer > _options.MaskedLayers)
            throw new InvalidOperationException($"SemanticLayer must lie in [1, {_options.MaskedLayers}].");
        var states = Encode(recording.Rank == 1 ? recording : Engine.Reshape(recording, new[] { recording.Length })).BlockStates;
        var x = states[_options.ContrastiveLayers + _options.SemanticLayer - 1];                                 // [1, L, D]
        int length = x.Shape[1], d = x.Shape[2];
        var rows = new double[length, d];
        for (int t = 0; t < length; t++)
            for (int i = 0; i < d; i++) rows[t, i] = NumOps.ToDouble(x[0, t, i]);
        return rows;
    }

    /// <summary>
    /// Fits AudioLM's semantic tokenizer on <paramref name="recordings"/>: the masked-prediction module's
    /// <see cref="W2vBertOptions.SemanticLayer"/>-th block output, standardized per dimension (zero mean, unit
    /// variance over the frames), clustered by k-means into <see cref="W2vBertOptions.SemanticClusters"/> tokens.
    /// </summary>
    public void FitSemanticTokenizer(IEnumerable<Tensor<T>> recordings)
    {
        ThrowIfDisposed();
        if (recordings is null) throw new ArgumentNullException(nameof(recordings));
        var all = recordings.Select(SemanticEmbeddings).ToList();
        int d = _options.ModelDim, frames = all.Sum(r => r.GetLength(0));
        if (frames < _options.SemanticClusters)
            throw new ArgumentException($"{frames} frames cannot form {_options.SemanticClusters} clusters.", nameof(recordings));
        var mean = new double[d];
        var scale = new double[d];
        foreach (var r in all)
            for (int t = 0; t < r.GetLength(0); t++)
                for (int i = 0; i < d; i++) mean[i] += r[t, i] / frames;
        foreach (var r in all)
            for (int t = 0; t < r.GetLength(0); t++)
                for (int i = 0; i < d; i++) scale[i] += Math.Pow(r[t, i] - mean[i], 2) / frames;
        for (int i = 0; i < d; i++) scale[i] = Math.Sqrt(scale[i]) is var s && s > 0 ? s : 1.0;
        _semanticMean = mean;
        _semanticScale = scale;
        var data = Standardize(all);
        var clusters = new KMeans<T>(new KMeansOptions<T> { NumClusters = _options.SemanticClusters, Seed = _options.Seed ?? 42 });
        clusters.Train(data, new Vector<T>(data.Rows));
        _semanticClusters = clusters;
    }

    private Matrix<T> Standardize(IReadOnlyList<double[,]> recordings)
    {
        var mean = _semanticMean ?? throw new InvalidOperationException("Fit the semantic tokenizer first.");
        var scale = _semanticScale ?? throw new InvalidOperationException("Fit the semantic tokenizer first.");
        int d = mean.Length, frames = recordings.Sum(r => r.GetLength(0)), row = 0;
        var data = new Matrix<T>(frames, d);
        foreach (var r in recordings)
            for (int t = 0; t < r.GetLength(0); t++, row++)
                for (int i = 0; i < d; i++) data[row, i] = NumOps.FromDouble((r[t, i] - mean[i]) / scale[i]);
        return data;
    }

    /// <summary>The semantic tokens of a recording, one per frame at 25 Hz (<see cref="FitSemanticTokenizer"/>
    /// first).</summary>
    public int[] SemanticTokens(Tensor<T> recording)
    {
        ThrowIfDisposed();
        var clusters = _semanticClusters ?? throw new InvalidOperationException("Fit the semantic tokenizer first.");
        var ids = clusters.Predict(Standardize(new[] { SemanticEmbeddings(recording) }));
        var tokens = new int[ids.Length];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (int)Math.Round(NumOps.ToDouble(ids[i]));
        return tokens;
    }

    // ---------------------------------------------------------------- inference and training

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var network = RequireNetwork();
        return Unbatch(input, network.Forward(network.Features(input), mask: null, training: false, _random).Final);
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => _useNativeMode ? ObservedLayerActivations(input) : new Dictionary<string, Tensor<T>>();

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var network = RequireNetwork();
        return Unbatch(input, network.Forward(network.Features(input), mask: null, training: true, _random).Final);
    }

    private Tensor<T> PretrainingLoss(Tensor<T> input, bool training, Random random)
    {
        var network = RequireNetwork();
        var pretraining = RequirePretraining();
        var mel = network.Features(input);
        int batch = mel.Shape[0], length = network.EncodedLength(mel.Shape[1]);
        var mask = network.DrawMask(batch, length, _options.MaskProbability, random);
        var output = network.Forward(mel, mask, training, random);
        // wav2vec 2.0's contrastive task on the context vectors, quantizing the unmasked encoded features.
        var asWav2Vec2 = new Wav2Vec2Network<T>.Output(output.Context, output.Features, output.BlockStates, output.Features, mask);
        var losses = pretraining.Loss(asWav2Vec2, pretraining.SampleNegatives(mask, random), training, random);
        var maskedPrediction = network.MaskedPredictionLoss(output, losses.CodeIds);
        LastPretrainingLosses = (NumOps.ToDouble(losses.Contrastive[0]), NumOps.ToDouble(losses.Diversity[0]),
            NumOps.ToDouble(maskedPrediction[0]));
        return Engine.TensorAdd(losses.Total, maskedPrediction);                                                  // β = γ = 1
    }

    /// <summary>
    /// One self-supervised pre-training step on waveforms <c>[B, S]</c> (or <c>[S]</c>): L_c + L_m (§3.2).
    /// </summary>
    /// <param name="input">The waveforms.</param>
    /// <param name="expected">Unused: the objective is self-supervised.</param>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        var pretraining = RequirePretraining();
        TrainWithCustomLoss(input, _ => PretrainingLoss(input, training: true, _random), Optimizer());
        pretraining.AfterUpdate();
        return true;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        double peak = _options.LearningRate;
        int warmup = Math.Max(1, _options.WarmupSteps);
        // Vaswani et al. 2017, §5.3: lr ∝ min(step^-0.5, step · warmup^-1.5), scaled to the stated peak at the warm-up's end.
        var schedule = new LambdaLRScheduler(peak, step =>
        {
            double s = step + 1;
            return Math.Min(s / warmup, Math.Sqrt(warmup / s));
        });
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = peak,
                Beta1 = 0.9,
                Beta2 = 0.98,
                Epsilon = 1e-9,
                LearningRateScheduler = schedule,
            }), "w2v-BERT");
        return _optimizer;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.SelfSupervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>Mask, noise and distractor draws averaged by <see cref="ITrainingObjectiveProvider{T}.EvaluateTrainingObjective"/>.</summary>
    private const int ObjectiveDraws = 8;

    /// <inheritdoc />
    /// <remarks>
    /// The pre-training loss of the input in inference mode, averaged over <see cref="ObjectiveDraws"/> draws of the
    /// mask, noise and distractors from fixed seeds, so it is a function of the input and the weights. The objective is
    /// an expectation over those draws, and one draw is a poor estimate of it: when every sampled distractor quantizes
    /// to its positive's code, fairseq's rule excludes them all and the contrastive term is exactly 0. Measured on a
    /// 16-wide model: one input's contrastive term was 0 for one draw in eight and 1.3 to 15.6 for the others.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        double total = 0, contrastive = 0, diversity = 0, masked = 0;
        for (int draw = 0; draw < ObjectiveDraws; draw++)
        {
            var random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 2 + draw);
            total += NumOps.ToDouble(PretrainingLoss(input, training: false, random)[0]);
            contrastive += LastPretrainingLosses.Contrastive;
            diversity += LastPretrainingLosses.Diversity;
            masked += LastPretrainingLosses.MaskedPrediction;
        }
        LastPretrainingLosses = (contrastive / ObjectiveDraws, diversity / ObjectiveDraws, masked / ObjectiveDraws);
        return NumOps.FromDouble(total / ObjectiveDraws);
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "w2v-BERT-Native" : "w2v-BERT-ONNX",
            Description = "w2v-BERT self-supervised speech model (Chung et al., 2021)",
        };
        m.AdditionalInfo["ModelDim"] = _options.ModelDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["Layers"] = NumLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(W2vBert<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}
