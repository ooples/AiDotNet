using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// TitaNet speaker verification and embedding extraction model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TitaNet (Koluguri et al., ICASSP 2022) is NVIDIA's speaker embedding model based on
/// 1D depth-wise separable convolutions with Squeeze-Excitation and global context.
/// TitaNet-L achieves 0.68% EER on VoxCeleb1-O, outperforming ECAPA-TDNN.
/// </para>
/// <para>
/// <b>For Beginners:</b> TitaNet is NVIDIA's advanced voice fingerprinting model. It listens
/// to someone speaking and creates a compact vector that uniquely identifies their voice.
/// You can use it to verify a speaker's identity or find who's speaking among known voices.
/// It comes in three sizes: Small (6M params), Medium (13M params), and Large (25M params).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 192);
/// var model = new TitaNet&lt;float&gt;(arch, "titanet_large.onnx");
/// var embedding = ((ISpeakerEmbeddingExtractor&lt;float&gt;)model).ExtractEmbedding(audio);
/// var result = model.Verify(testAudio, enrolledEmbedding);
/// </code>
/// </para>
/// </remarks>
public class TitaNet<T> : SpeakerRecognitionBase<T>, ISpeakerVerifier<T>, ISpeakerEmbeddingExtractor<T>
{
    #region Fields

    private readonly TitaNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ConcurrentDictionary<string, SpeakerProfile<T>> _enrolledSpeakers;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Properties

    public T DefaultThreshold { get; }
    public ISpeakerEmbeddingExtractor<T> EmbeddingExtractor => this;
    public new bool IsOnnxMode => !_useNativeMode && OnnxEncoder is not null;
    public double MinimumDurationSeconds => _options.MinDurationSeconds;
    public double VerificationThreshold => NumOps.ToDouble(DefaultThreshold);

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TitaNet speaker model in ONNX inference mode.
    /// </summary>
    public TitaNet(NeuralNetworkArchitecture<T> architecture, string modelPath, TitaNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TitaNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        DefaultThreshold = NumOps.FromDouble(_options.DefaultThreshold);
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _enrolledSpeakers = new ConcurrentDictionary<string, SpeakerProfile<T>>();
        InitializeLayers();
    }

    /// <summary>
    /// Creates a TitaNet speaker model in native training mode.
    /// </summary>
    public TitaNet(NeuralNetworkArchitecture<T> architecture, TitaNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TitaNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        DefaultThreshold = NumOps.FromDouble(_options.DefaultThreshold);
        _enrolledSpeakers = new ConcurrentDictionary<string, SpeakerProfile<T>>();
        InitializeLayers();
    }

    internal static async Task<TitaNet<T>> CreateAsync(TitaNetOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new TitaNetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("titanet", "titanet_large.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new TitaNet<T>(arch, mp, options);
    }

    #endregion

    #region ISpeakerVerifier

    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding)
        => Verify(audio, referenceEmbedding, DefaultThreshold);

    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding, T threshold)
    {
        ThrowIfDisposed();
        var testEmb = ExtractEmbedding(audio);
        var score = ComputeCosineSimilarity(testEmb, referenceEmbedding);
        bool isAccepted = NumOps.ToDouble(score) >= NumOps.ToDouble(threshold);
        var confidence = NumOps.Abs(NumOps.Subtract(score, threshold));
        return new SpeakerVerificationResult<T> { IsAccepted = isAccepted, Score = score, Threshold = threshold, Confidence = confidence };
    }

    public SpeakerVerificationResult<T> VerifyWithReferenceAudio(Tensor<T> audio, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        return Verify(audio, ExtractEmbedding(referenceAudio));
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
        return ComputeCosineSimilarity(ExtractEmbedding(audio), referenceEmbedding);
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

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultTitaNetLayers(
            numMels: _options.NumMels, encoderDim: _options.EncoderDim,
            embeddingDim: _options.EmbeddingDim, numBlocks: _options.NumEncoderBlocks,
            attentivePoolingDim: _options.AttentivePoolingDim, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MfccExtractor is not null) return MfccExtractor.Extract(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => NormalizeEmbedding(o);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "TitaNet-Native" : "TitaNet-ONNX",
            Description = $"TitaNet-{_options.Variant} Speaker Embedding (Koluguri et al., ICASSP 2022)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumEncoderBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.EmbeddingDim);
        w.Write(_options.EncoderDim); w.Write(_options.NumEncoderBlocks); w.Write(_options.ConvKernelSize);
        w.Write(_options.AttentivePoolingDim); w.Write(_options.DropoutRate); w.Write(_options.DefaultThreshold);
        w.Write(_options.Variant);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.EmbeddingDim = r.ReadInt32();
        _options.EncoderDim = r.ReadInt32(); _options.NumEncoderBlocks = r.ReadInt32(); _options.ConvKernelSize = r.ReadInt32();
        _options.AttentivePoolingDim = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.DefaultThreshold = r.ReadDouble();
        _options.Variant = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new TitaNet<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TitaNet<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
