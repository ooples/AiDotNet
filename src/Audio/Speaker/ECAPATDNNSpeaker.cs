using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// ECAPA-TDNN speaker verification and embedding extraction model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ECAPA-TDNN (Desplanques et al., Interspeech 2020) is a state-of-the-art speaker embedding
/// architecture that extends x-vectors with Squeeze-Excitation blocks, multi-layer feature
/// aggregation, and channel- and context-dependent statistics pooling. Achieves 0.87% EER
/// on VoxCeleb1 test set.
/// </para>
/// <para>
/// <b>For Beginners:</b> ECAPA-TDNN creates a unique "voiceprint" for any speaker. Feed it
/// audio of someone speaking, and it returns a compact vector (embedding) that captures the
/// speaker's unique voice characteristics. You can compare two embeddings to check if they
/// are the same person (verification) or find the closest match among enrolled speakers
/// (identification).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 192);
/// var model = new ECAPATDNNSpeaker&lt;float&gt;(arch, "ecapa_tdnn.onnx");
/// var embedding = ((ISpeakerEmbeddingExtractor&lt;float&gt;)model).ExtractEmbedding(audio);
/// var result = model.Verify(testAudio, enrolledEmbedding);
/// </code>
/// </para>
/// </remarks>
public class ECAPATDNNSpeaker<T> : SpeakerRecognitionBase<T>, ISpeakerVerifier<T>, ISpeakerEmbeddingExtractor<T>
{
    #region Fields

    private readonly ECAPATDNNSpeakerOptions _options;
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
    /// Creates an ECAPA-TDNN speaker model in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Model options. If null, defaults are used.</param>
    public ECAPATDNNSpeaker(NeuralNetworkArchitecture<T> architecture, string modelPath, ECAPATDNNSpeakerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ECAPATDNNSpeakerOptions();
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
    /// Creates an ECAPA-TDNN speaker model in native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Model options. If null, defaults are used.</param>
    /// <param name="optimizer">Optimizer for training. If null, AdamW is used.</param>
    public ECAPATDNNSpeaker(NeuralNetworkArchitecture<T> architecture, ECAPATDNNSpeakerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ECAPATDNNSpeakerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        DefaultThreshold = NumOps.FromDouble(_options.DefaultThreshold);
        _enrolledSpeakers = new ConcurrentDictionary<string, SpeakerProfile<T>>();
        InitializeLayers();
    }

    internal static async Task<ECAPATDNNSpeaker<T>> CreateAsync(ECAPATDNNSpeakerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new ECAPATDNNSpeakerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("ecapa_tdnn", "ecapa_tdnn.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new ECAPATDNNSpeaker<T>(arch, mp, options);
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
        bool isAccepted = NumOps.ToDouble(score) >= NumOps.ToDouble(threshold);
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

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultECAPATDNNSpeakerLayers(
            numMels: _options.NumMels, channels: _options.Channels[0],
            embeddingDim: _options.EmbeddingDim, numBlocks: _options.Channels.Length - 2,
            poolingDim: _options.PoolingDim, seBottleneckDim: _options.SEBottleneckDim,
            dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "ECAPA-TDNN-Native" : "ECAPA-TDNN-ONNX",
            Description = "ECAPA-TDNN Speaker Embedding (Desplanques et al., Interspeech 2020)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.Channels.Length
        };
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["Res2NetScale"] = _options.Res2NetScale.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.EmbeddingDim);
        w.Write(_options.PoolingDim); w.Write(_options.SEBottleneckDim); w.Write(_options.Res2NetScale);
        w.Write(_options.DropoutRate); w.Write(_options.DefaultThreshold);
        w.Write(_options.Channels.Length); foreach (var c in _options.Channels) w.Write(c);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.EmbeddingDim = r.ReadInt32();
        _options.PoolingDim = r.ReadInt32(); _options.SEBottleneckDim = r.ReadInt32(); _options.Res2NetScale = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble(); _options.DefaultThreshold = r.ReadDouble();
        int n = r.ReadInt32(); _options.Channels = new int[n]; for (int i = 0; i < n; i++) _options.Channels[i] = r.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new ECAPATDNNSpeaker<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ECAPATDNNSpeaker<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
