using AiDotNet.LearningRateSchedulers;
using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// CAM++ (Context-Aware Masking Plus Plus) speaker verification model (Wang et al., 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CAM++ is a fast speaker verification model using context-aware masking with a densely
/// connected TDNN (D-TDNN). It learns to mask uninformative frames (silence, noise) and
/// focus on speech-rich segments. Achieves competitive EER while being significantly faster
/// than Transformer-based approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> CAM++ is like a smart listener who can automatically tune out
/// background noise and silence, focusing only on the parts of audio where someone is actually
/// speaking. It creates a unique "voiceprint" from just the speech portions, making it both
/// fast and accurate for identifying people by their voice.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 192);
/// var model = new CAMPlusPlus&lt;float&gt;(arch, "cam_plus_plus.onnx");
/// var embedding = ((ISpeakerEmbeddingExtractor&lt;float&gt;)model).ExtractEmbedding(audio);
/// var result = model.Verify(testAudio, enrolledEmbedding);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking", "https://arxiv.org/abs/2303.00332", Year = 2023, Authors = "Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, Qian Chen")]
[PaperOptimizer(OptimizerKind.Sgd, LearningRate = 0.1, MinLearningRate = 1e-4,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Wang et al. 2023, Sec. 3: SGD with a cosine annealing scheduler and a linear warm-up scheduler, the learning rate varying between 0.1 and 1e-4. The warm-up length is not stated, so none is declared.")]
public partial class CAMPlusPlus<T> : SpeakerRecognitionBase<T>, ISpeakerVerifier<T>, ISpeakerEmbeddingExtractor<T>
{
    #region Fields

    private readonly CAMPlusPlusOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // The native default topology; null in ONNX mode and when the caller supplied layers.
    private CamPlusPlusEncoder<T>? _encoder;

    // The Kaldi filterbank front end, built on first use.
    private KaldiFbank<T>? _fbank;

    #endregion

    #region Properties

    public T DefaultThreshold { get; }
    public ISpeakerEmbeddingExtractor<T> EmbeddingExtractor => this;
    public new bool IsOnnxMode => !_useNativeMode && OnnxEncoder is not null;
    public double MinimumDurationSeconds => _options.MinDurationSeconds;
    public double VerificationThreshold => NumOps.ToDouble(DefaultThreshold);

    #endregion

    #region Constructors

    public CAMPlusPlus(NeuralNetworkArchitecture<T> architecture, string modelPath, CAMPlusPlusOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CAMPlusPlusOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        DefaultThreshold = NumOps.FromDouble(_options.DefaultThreshold);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public CAMPlusPlus(NeuralNetworkArchitecture<T> architecture, CAMPlusPlusOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CAMPlusPlusOptions();
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
        EmbeddingDimension = _options.EmbeddingDim;
        DefaultThreshold = NumOps.FromDouble(_options.DefaultThreshold);
        InitializeLayers();
    }

    internal static async Task<CAMPlusPlus<T>> CreateAsync(CAMPlusPlusOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CAMPlusPlusOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("cam_plus_plus", "cam_plus_plus.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new CAMPlusPlus<T>(arch, mp, options);
    }

    #endregion

    #region ISpeakerVerifier

    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding)
        => Verify(audio, referenceEmbedding, DefaultThreshold);

    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding, T threshold)
    {
        ThrowIfDisposed();
        var testEmbedding = ExtractEmbedding(audio);
        var score = ComputeCosineSimilarity(testEmbedding, referenceEmbedding);
        bool isAccepted = NumOps.GreaterThanOrEquals(score, threshold);
        var confidence = NumOps.Abs(NumOps.Subtract(score, threshold));
        return new SpeakerVerificationResult<T> { IsAccepted = isAccepted, Score = score, Threshold = threshold, Confidence = confidence };
    }

    public SpeakerVerificationResult<T> VerifyWithReferenceAudio(Tensor<T> audio, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        var refEmb = ExtractEmbedding(referenceAudio);
        return Verify(audio, refEmb);
    }

    public Task<SpeakerVerificationResult<T>> VerifyAsync(Tensor<T> audio, Tensor<T> referenceEmbedding, CancellationToken cancellationToken = default)
        => Task.Run(() => Verify(audio, referenceEmbedding), cancellationToken);

    public SpeakerProfile<T> Enroll(IReadOnlyList<Tensor<T>> enrollmentAudio)
    {
        ThrowIfDisposed();
        if (enrollmentAudio.Count == 0) throw new ArgumentException("At least one audio sample required for enrollment.");
        var embeddings = enrollmentAudio.Select(a => ExtractEmbedding(a)).ToList();
        var aggregated = AggregateEmbeddings(embeddings);
        double totalDuration = enrollmentAudio.Sum(a => (double)a.Length / SampleRate);
        return new SpeakerProfile<T>
        {
            SpeakerId = Guid.NewGuid().ToString(), Embedding = aggregated,
            NumEnrollmentSamples = enrollmentAudio.Count, TotalEnrollmentDuration = totalDuration,
            CreatedAt = DateTime.UtcNow, UpdatedAt = DateTime.UtcNow
        };
    }

    public SpeakerProfile<T> Enroll(Tensor<T> enrollmentAudio) => Enroll([enrollmentAudio]);

    public SpeakerProfile<T> UpdateProfile(SpeakerProfile<T> existingProfile, Tensor<T> newAudio)
    {
        ThrowIfDisposed();
        var newEmb = ExtractEmbedding(newAudio);
        var aggregated = AggregateEmbeddings([existingProfile.Embedding, newEmb]);
        return new SpeakerProfile<T>
        {
            SpeakerId = existingProfile.SpeakerId, Embedding = aggregated,
            NumEnrollmentSamples = existingProfile.NumEnrollmentSamples + 1,
            TotalEnrollmentDuration = existingProfile.TotalEnrollmentDuration + ((double)newAudio.Length / SampleRate),
            CreatedAt = existingProfile.CreatedAt, UpdatedAt = DateTime.UtcNow
        };
    }

    public T ComputeScore(Tensor<T> audio, Tensor<T> referenceEmbedding)
    {
        ThrowIfDisposed();
        var testEmb = ExtractEmbedding(audio);
        return ComputeCosineSimilarity(testEmb, referenceEmbedding);
    }

    public T GetThresholdForFAR(double targetFAR)
    {
        double threshold = 0.85 - Math.Log10(1.0 / targetFAR) * 0.1;
        threshold = Math.Max(0.3, Math.Min(0.95, threshold));
        return NumOps.FromDouble(threshold);
    }

    #endregion

    #region ISpeakerEmbeddingExtractor

    public Tensor<T> ExtractEmbedding(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> raw = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return NormalizeEmbedding(raw);
    }

    public Task<Tensor<T>> ExtractEmbeddingAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => ExtractEmbedding(audio), cancellationToken);

    public IReadOnlyList<Tensor<T>> ExtractEmbeddings(IReadOnlyList<Tensor<T>> audioSegments)
        => audioSegments.Select(a => ExtractEmbedding(a)).ToList();

    public T ComputeSimilarity(Tensor<T> embedding1, Tensor<T> embedding2)
        => ComputeCosineSimilarity(embedding1, embedding2);

    Tensor<T> ISpeakerEmbeddingExtractor<T>.AggregateEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
        => AggregateEmbeddings(embeddings);

    Tensor<T> ISpeakerEmbeddingExtractor<T>.NormalizeEmbedding(Tensor<T> embedding)
        => NormalizeEmbedding(embedding);

    #endregion

    #region NeuralNetworkBase

    /// <remarks>
    /// The default topology is 3D-Speaker's <c>CAMPPlus</c> (<see cref="CamPlusPlusEncoder{T}"/>): the FCM head, the
    /// TDNN layer, three context-aware-masked D-TDNN blocks, statistics pooling and the embedding with its non-affine
    /// BatchNorm. Its layers are published through <c>Layers</c>, so training, serialization and cloning walk the
    /// instances the forward runs. A caller-supplied stack runs as the sequential chain it was written as.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        var configuration = new CamPlusPlusConfiguration(_options.NumMels, _options.EmbeddingDim, _options.GrowthRate,
            _options.BottleneckMultiplier, _options.InitialChannels);
        var layers = new List<LayerBase<T>>();
        _encoder = new CamPlusPlusEncoder<T>(Engine, layers, configuration,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? _options.Seed ?? 0));
        Layers.AddRange(layers);
        _encoder.SetTrainingMode(IsTrainingMode);
    }

    /// <inheritdoc />
    /// <remarks>The embedding BatchNorm is computed by the encoder, so it follows the mode too.</remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _encoder?.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    /// <remarks>The embedding's BatchNorm has no scale or shift in CAM++ (<c>affine=False</c>); the layer's are held at 1
    /// and 0 by leaving them out of every training step.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_encoder is null) return parameters;
        var frozen = new HashSet<Tensor<T>>(_encoder.FrozenParameters, Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(p => !frozen.Contains(p)).ToList();
    }

    /// <summary>
    /// Loads a 3D-Speaker <c>CAMPPlus</c> state dict (as <c>speech_campplus_sv_zh-cn_16k-common</c> releases it), each
    /// tensor read by name and shape.
    /// </summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        Encoder.LoadTorchWeights(read);
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    private CamPlusPlusEncoder<T> Encoder =>
        _encoder ?? throw new InvalidOperationException("The CAM++ encoder exists only for the native default topology.");

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        return ForwardNative(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardNative(input);

    /// <inheritdoc/>
    /// <remarks>Every layer of the default topology is built at its final size, so there is nothing to resolve; the
    /// base walk would instead run them as a chain, which they are not.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_encoder is null) base.ResolveLazyLayerShapes();
    }

    /// <summary>
    /// Runs filterbank frames <c>[T, NumMels]</c> (or <c>[B, T, NumMels]</c>) through the encoder to an embedding
    /// <c>[EmbeddingDim]</c> (or <c>[B, EmbeddingDim]</c>).
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        if (_encoder is null)
        {
            var c = input;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }

        // The encoder holds its layers directly; a restore that swapped one out would leave it running stale weights.
        if (!_encoder.Runs(Layers))
            throw new InvalidOperationException("CAM++'s layers are no longer the instances its encoder runs.");
        if (input.Shape.Length == 2)
        {
            var embedding = _encoder.Forward(Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }));
            return Engine.Reshape(embedding, new[] { _options.EmbeddingDim });
        }
        if (input.Shape.Length != 3)
            throw new ArgumentException($"CAM++ takes filterbank frames [T, {_options.NumMels}] or [B, T, {_options.NumMels}].", nameof(input));
        return _encoder.Forward(input);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>
    /// CosyVoice's and 3D-Speaker's front end for a waveform <c>[samples]</c>: Kaldi filterbank energies
    /// (<c>kaldi.fbank(num_mel_bins=80, dither=0)</c>, the Povey window) with each band's mean over the utterance
    /// subtracted. Features (rank 2 or more) pass through.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (rawAudio.Shape.Length != 1) return rawAudio;
        _fbank ??= new KaldiFbank<T>(Engine, _options.SampleRate, _options.NumMels, WindowType.Povey,
            _options.FrameLengthMilliseconds, _options.FrameShiftMilliseconds);
        var features = _fbank.Forward(rawAudio);
        var mean = Engine.ReduceMean(features, new[] { 0 }, true);
        return Engine.TensorSubtract(features, Engine.TensorBroadcastTo(mean, features.Shape.ToArray()));
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => NormalizeEmbedding(o);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CAM++-Native" : "CAM++-ONNX",
            Description = "CAM++ Context-Aware Masking Speaker Model (Wang et al., 2023)",
            Complexity = CamPlusPlusConfiguration.BlockShapes.Length
        };
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["InitialChannels"] = _options.InitialChannels.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CAMPlusPlus<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
