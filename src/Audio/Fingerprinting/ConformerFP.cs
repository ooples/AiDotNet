using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Conformer-based audio fingerprinting model combining self-attention with convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ConformerFP applies the Conformer architecture (convolution-augmented Transformer) to audio
/// fingerprinting. It combines self-attention for global context with convolutions for local
/// feature extraction, producing highly robust fingerprints for large-scale audio retrieval.
/// </para>
/// <para>
/// <b>For Beginners:</b> ConformerFP uses a powerful architecture that combines two ways of
/// understanding audio: attention (looking at the big picture) and convolutions (looking at
/// local details). This combination makes it very good at creating fingerprints that can
/// identify songs even from noisy or distorted recordings.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 256, outputSize: 128);
/// var model = new ConformerFP&lt;float&gt;(arch, "conformer_fp.onnx");
/// var fp = model.Fingerprint(audioClip);
/// double similarity = model.ComputeSimilarity(fp, referenceFp);
/// </code>
/// </para>
/// </remarks>
public class ConformerFP<T> : AudioNeuralNetworkBase<T>, IAudioFingerprinter<T>
{
    #region Fields

    private readonly ConformerFPOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private MelSpectrogram<T>? _melSpectrogram;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioFingerprinter Properties

    /// <inheritdoc />
    public string Name => "ConformerFP";

    /// <inheritdoc />
    public int FingerprintLength => _options.EmbeddingDim;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a ConformerFP model in ONNX inference mode.
    /// </summary>
    public ConformerFP(NeuralNetworkArchitecture<T> architecture, string modelPath, ConformerFPOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ConformerFPOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a ConformerFP model in native training mode.
    /// </summary>
    public ConformerFP(NeuralNetworkArchitecture<T> architecture, ConformerFPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ConformerFPOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength);
        InitializeLayers();
    }

    internal static async Task<ConformerFP<T>> CreateAsync(ConformerFPOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new ConformerFPOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("conformer_fp", "conformer_fp.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new ConformerFP<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFingerprinter

    /// <inheritdoc />
    public AudioFingerprint<T> Fingerprint(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> embedding = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        double norm = 0;
        for (int i = 0; i < embedding.Length; i++) { double v = NumOps.ToDouble(embedding[i]); norm += v * v; }
        norm = Math.Sqrt(norm);

        var data = new T[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
            data[i] = norm > 0 ? NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm) : embedding[i];

        return new AudioFingerprint<T>
        {
            Data = data,
            Duration = audio.Length / (double)_options.SampleRate,
            SampleRate = _options.SampleRate,
            Algorithm = "ConformerFP",
            FrameCount = 1
        };
    }

    /// <inheritdoc />
    public AudioFingerprint<T> Fingerprint(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++) tensor[i] = audio[i];
        return Fingerprint(tensor);
    }

    /// <inheritdoc />
    public double ComputeSimilarity(AudioFingerprint<T> fp1, AudioFingerprint<T> fp2)
    {
        ThrowIfDisposed();
        double dot = 0;
        int len = Math.Min(fp1.Data.Length, fp2.Data.Length);
        for (int i = 0; i < len; i++)
            dot += NumOps.ToDouble(fp1.Data[i]) * NumOps.ToDouble(fp2.Data[i]);
        return Math.Max(0, Math.Min(1, (dot + 1.0) / 2.0));
    }

    /// <inheritdoc />
    public IReadOnlyList<FingerprintMatch> FindMatches(AudioFingerprint<T> query, AudioFingerprint<T> reference, int minMatchLength = 10)
    {
        ThrowIfDisposed();
        var matches = new List<FingerprintMatch>();
        int embDim = _options.EmbeddingDim;
        int queryFrames = query.Data.Length / Math.Max(1, embDim);
        int refFrames = reference.Data.Length / Math.Max(1, embDim);
        if (queryFrames <= 0 || refFrames <= 0) return matches;

        double threshold = _options.MatchThreshold;
        for (int rStart = 0; rStart <= refFrames - Math.Min(queryFrames, minMatchLength); rStart++)
        {
            int matchLen = Math.Min(queryFrames, refFrames - rStart);
            double sim = 0, normQ = 0, normR = 0;
            for (int f = 0; f < matchLen; f++)
                for (int d = 0; d < embDim && (f * embDim + d) < query.Data.Length && ((rStart + f) * embDim + d) < reference.Data.Length; d++)
                {
                    double q = NumOps.ToDouble(query.Data[f * embDim + d]);
                    double r = NumOps.ToDouble(reference.Data[(rStart + f) * embDim + d]);
                    sim += q * r; normQ += q * q; normR += r * r;
                }
            double denom = Math.Sqrt(normQ) * Math.Sqrt(normR);
            double cosSim = denom > 1e-8 ? sim / denom : 0;
            if (cosSim >= threshold && matchLen >= minMatchLength)
            {
                double timePerFrame = query.Duration / Math.Max(1, queryFrames);
                matches.Add(new FingerprintMatch { QueryStartTime = 0, ReferenceStartTime = rStart * timePerFrame,
                    Duration = matchLen * timePerFrame, Confidence = cosSim, MatchCount = matchLen });
                rStart += matchLen - 1;
            }
        }
        return matches;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultConformerFPLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, embeddingDim: _options.EmbeddingDim,
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
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is not null) return _melSpectrogram.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ConformerFP-Native" : "ConformerFP-ONNX",
            Description = "Conformer-based audio fingerprinting (attention + convolution)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.SegmentDurationSec);
        w.Write(_options.EmbeddingDim); w.Write(_options.HiddenDim);
        w.Write(_options.NumLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.FeedForwardDim); w.Write(_options.ConvKernelSize);
        w.Write(_options.Temperature); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.SegmentDurationSec = r.ReadDouble();
        _options.EmbeddingDim = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.FeedForwardDim = r.ReadInt32(); _options.ConvKernelSize = r.ReadInt32();
        _options.Temperature = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new ConformerFP<T>(Architecture, mp, _options);
        return new ConformerFP<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ConformerFP<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
