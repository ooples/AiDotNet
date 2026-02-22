using AiDotNet.Diffusion;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// SCNet (Sparse Compression Network) for music source separation (Chen et al., 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SCNet uses sparse compression to reduce the frequency dimension before processing with
/// attention layers, achieving competitive separation quality with significantly fewer parameters.
/// It compresses frequency features into compact cluster representations, processes them, then
/// decompresses back to the full frequency resolution for mask estimation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of processing every individual frequency separately (which is slow),
/// SCNet groups similar frequencies into clusters, processes the clusters efficiently, then expands
/// back to full detail. Think of it like editing a compressed photo—you can make changes much faster,
/// and the results still look great when decompressed.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 1025);
/// var model = new SCNet&lt;float&gt;(arch, "scnet.onnx");
/// var result = model.Separate(mixedAudio);
/// var vocals = result.GetSource("vocals");
/// </code>
/// </para>
/// </remarks>
public class SCNet<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    #region Fields

    private readonly SCNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    public SCNet(NeuralNetworkArchitecture<T> architecture, string modelPath, SCNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SCNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    public SCNet(NeuralNetworkArchitecture<T> architecture, SCNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SCNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    internal static async Task<SCNet<T>> CreateAsync(SCNetOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SCNetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("scnet", "scnet.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins * options.NumStems);
        return new SCNet<T>(arch, mp, options);
    }

    #endregion

    #region IMusicSourceSeparator Properties

    public IReadOnlyList<string> SupportedSources => _options.Sources;
    public int NumStems => _options.NumStems;

    #endregion

    #region IMusicSourceSeparator Methods

    public SourceSeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var stft = ComputeSTFT(audio);
        Tensor<T> masks;
        if (IsOnnxMode && OnnxEncoder is not null) masks = OnnxEncoder.Run(stft);
        else masks = Predict(stft);
        return BuildSeparationResult(audio, stft, masks);
    }

    public Task<SourceSeparationResult<T>> SeparateAsync(Tensor<T> audio, CancellationToken ct = default)
        => Task.Run(() => Separate(audio), ct);

    public Tensor<T> ExtractSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        return result.GetSource(source);
    }

    public Tensor<T> RemoveSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        var output = new Tensor<T>([audio.Length]);
        foreach (var kvp in result.Sources)
        {
            if (!string.Equals(kvp.Key, source, StringComparison.OrdinalIgnoreCase))
                for (int i = 0; i < Math.Min(output.Length, kvp.Value.Length); i++)
                    output[i] = NumOps.Add(output[i], kvp.Value[i]);
        }
        return output;
    }

    public Tensor<T> GetSourceMask(Tensor<T> audio, string source)
    {
        var stft = ComputeSTFT(audio);
        Tensor<T> masks;
        if (IsOnnxMode && OnnxEncoder is not null) masks = OnnxEncoder.Run(stft);
        else masks = Predict(stft);
        int idx = Array.IndexOf(_options.Sources, source);
        if (idx < 0) throw new ArgumentException($"Unknown source: {source}");
        int frameCount = stft.Shape[0];
        var mask = new Tensor<T>([frameCount, _options.NumFreqBins]);
        for (int f = 0; f < frameCount; f++)
            for (int b = 0; b < _options.NumFreqBins && idx * _options.NumFreqBins + b < masks.Length; b++)
                mask[f, b] = masks[f * _options.NumFreqBins * _options.NumStems + idx * _options.NumFreqBins + b];
        return mask;
    }

    public Tensor<T> Remix(SourceSeparationResult<T> separationResult, IReadOnlyDictionary<string, double> sourceVolumes)
    {
        int len = 0;
        foreach (var s in separationResult.Sources.Values) { if (s.Length > len) len = s.Length; }
        var output = new Tensor<T>([len]);
        foreach (var kvp in separationResult.Sources)
        {
            double vol = sourceVolumes.TryGetValue(kvp.Key, out var v) ? v : 1.0;
            for (int i = 0; i < kvp.Value.Length; i++)
                output[i] = NumOps.Add(output[i], NumOps.FromDouble(NumOps.ToDouble(kvp.Value[i]) * vol));
        }
        return output;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultSCNetLayers(
            numClusters: _options.NumClusters, compressionDim: _options.CompressionDim,
            numEncoderBlocks: _options.NumEncoderBlocks, numDecoderBlocks: _options.NumDecoderBlocks,
            attentionDim: _options.AttentionDim, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, numStems: _options.NumStems,
            numFreqBins: _options.NumFreqBins, dropoutRate: _options.DropoutRate));
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
        SetTrainingMode(true); var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers); SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SCNet-Native" : "SCNet-ONNX",
            Description = "SCNet Sparse Compression Network (Chen et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumEncoderBlocks + _options.NumDecoderBlocks
        };
        m.AdditionalInfo["NumClusters"] = _options.NumClusters.ToString();
        m.AdditionalInfo["NumStems"] = _options.NumStems.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength); w.Write(_options.NumFreqBins);
        w.Write(_options.NumClusters); w.Write(_options.CompressionDim);
        w.Write(_options.NumEncoderBlocks); w.Write(_options.NumDecoderBlocks);
        w.Write(_options.AttentionDim); w.Write(_options.NumAttentionHeads);
        w.Write(_options.NumStems); w.Write(_options.DropoutRate);
        w.Write(_options.Sources.Length); foreach (var s in _options.Sources) w.Write(s);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32(); _options.NumFreqBins = r.ReadInt32();
        _options.NumClusters = r.ReadInt32(); _options.CompressionDim = r.ReadInt32();
        _options.NumEncoderBlocks = r.ReadInt32(); _options.NumDecoderBlocks = r.ReadInt32();
        _options.AttentionDim = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.NumStems = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32(); _options.Sources = new string[n]; for (int i = 0; i < n; i++) _options.Sources[i] = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new SCNet<T>(Architecture, _options);

    #endregion

    #region Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
    }

    private SourceSeparationResult<T> BuildSeparationResult(Tensor<T> audio, Tensor<T> magnitude, Tensor<T> masks)
    {
        int nf = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        var sources = new Dictionary<string, Tensor<T>>();
        for (int si = 0; si < _options.NumStems && si < _options.Sources.Length; si++)
        {
            var maskedMag = new Tensor<T>(magnitude.Shape);
            for (int f = 0; f < nf; f++)
                for (int b = 0; b < numBins; b++)
                {
                    int mi = f * numBins * _options.NumStems + si * numBins + b;
                    double mask = mi < masks.Length ? Math.Max(0, Math.Min(1, NumOps.ToDouble(masks[mi]))) : 0;
                    maskedMag[f, b] = NumOps.FromDouble(NumOps.ToDouble(magnitude[f, b]) * mask);
                }
            if (_lastPhase is not null)
                sources[_options.Sources[si]] = _stft.InverseFromMagnitudeAndPhase(maskedMag, _lastPhase, audio.Length);
            else
                sources[_options.Sources[si]] = new Tensor<T>([audio.Length]);
        }
        return new SourceSeparationResult<T>
        {
            Sources = sources, OriginalMix = audio,
            SampleRate = _options.SampleRate, Duration = audio.Length / (double)_options.SampleRate
        };
    }

    private static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SCNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
