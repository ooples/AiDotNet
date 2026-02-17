using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// MelBand-RoFormer for mel-band music source separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MelBand-RoFormer (2024) extends BS-RoFormer by using mel-scale frequency bands instead of
/// linear bands, better matching human perception. Achieves 13.2 dB SDR on vocals (MUSDB18-HQ).
/// </para>
/// <para>
/// <b>For Beginners:</b> This model separates instruments in a song using mel-scale bands
/// that match how humans hear. It's like having a smart equalizer that knows which parts
/// belong to vocals, drums, bass, and other instruments.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 1025);
/// var model = new MelBandRoFormer&lt;float&gt;(arch, "melband_roformer.onnx");
/// var result = model.Separate(mixedAudio);
/// </code>
/// </para>
/// </remarks>
public class MelBandRoFormer<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    #region Fields

    private readonly MelBandRoFormerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    public MelBandRoFormer(NeuralNetworkArchitecture<T> architecture, string modelPath, MelBandRoFormerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MelBandRoFormerOptions(); _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    public MelBandRoFormer(NeuralNetworkArchitecture<T> architecture, MelBandRoFormerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MelBandRoFormerOptions(); _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft2 = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft2, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft2 ? _options.FftSize : null);
        InitializeLayers();
    }

    internal static async Task<MelBandRoFormer<T>> CreateAsync(MelBandRoFormerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MelBandRoFormerOptions(); string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp)) { var dl = new OnnxModelDownloader(); mp = await dl.DownloadAsync("melbandroformer", "melband_roformer.onnx", progress: progress, cancellationToken); options.ModelPath = mp; }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins * options.NumStems);
        return new MelBandRoFormer<T>(arch, mp, options);
    }

    #endregion

    #region IMusicSourceSeparator

    public IReadOnlyList<string> SupportedSources => _options.Sources;
    public int NumStems => _options.NumStems;

    public SourceSeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed(); var stft = ComputeSTFT(audio);
        Tensor<T> masks = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(stft) : Predict(stft);
        return BuildResult(audio, stft, masks);
    }

    public Task<SourceSeparationResult<T>> SeparateAsync(Tensor<T> audio, CancellationToken ct = default) => Task.Run(() => Separate(audio), ct);

    public Tensor<T> ExtractSource(Tensor<T> audio, string source) => Separate(audio).GetSource(source);

    public Tensor<T> RemoveSource(Tensor<T> audio, string source)
    {
        var r = Separate(audio); var o = new Tensor<T>([audio.Length]);
        foreach (var kvp in r.Sources) if (!string.Equals(kvp.Key, source, StringComparison.OrdinalIgnoreCase))
            for (int i = 0; i < Math.Min(o.Length, kvp.Value.Length); i++) o[i] = NumOps.Add(o[i], kvp.Value[i]);
        return o;
    }

    public Tensor<T> GetSourceMask(Tensor<T> audio, string source)
    {
        var stft = ComputeSTFT(audio);
        Tensor<T> masks = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(stft) : Predict(stft);
        int idx = Array.IndexOf(_options.Sources, source); if (idx < 0) throw new ArgumentException($"Unknown source: {source}");
        int nf = stft.Shape[0]; var mask = new Tensor<T>([nf, _options.NumFreqBins]);
        for (int f = 0; f < nf; f++) for (int b = 0; b < _options.NumFreqBins; b++)
            { int mi = f * _options.NumFreqBins * _options.NumStems + idx * _options.NumFreqBins + b; if (mi < masks.Length) mask[f, b] = masks[mi]; }
        return mask;
    }

    public Tensor<T> Remix(SourceSeparationResult<T> sep, IReadOnlyDictionary<string, double> vols)
    {
        int len = 0; foreach (var s in sep.Sources.Values) if (s.Length > len) len = s.Length;
        var o = new Tensor<T>([len]);
        foreach (var kvp in sep.Sources) { double v = vols.TryGetValue(kvp.Key, out var vol) ? vol : 1.0; for (int i = 0; i < kvp.Value.Length; i++) o[i] = NumOps.Add(o[i], NumOps.FromDouble(NumOps.ToDouble(kvp.Value[i]) * v)); }
        return o;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMelBandRoFormerLayers(
            numMelBands: _options.NumMelBands, bandEmbeddingDim: _options.BandEmbeddingDim,
            transformerDim: _options.TransformerDim, numTransformerLayers: _options.NumTransformerLayers,
            numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim,
            numStems: _options.NumStems, numFreqBins: _options.NumFreqBins, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true); var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers); SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "MelBand-RoFormer-Native" : "MelBand-RoFormer-ONNX", Description = "MelBand-RoFormer Mel-Band Source Separation (2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFreqBins, Complexity = _options.NumTransformerLayers };
        m.AdditionalInfo["NumMelBands"] = _options.NumMelBands.ToString(); m.AdditionalInfo["NumStems"] = _options.NumStems.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength); w.Write(_options.NumFreqBins);
        w.Write(_options.NumMelBands); w.Write(_options.TransformerDim); w.Write(_options.NumTransformerLayers);
        w.Write(_options.NumStems); w.Write(_options.DropoutRate);
        w.Write(_options.Sources.Length); foreach (var s in _options.Sources) w.Write(s);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32(); _options.NumFreqBins = r.ReadInt32();
        _options.NumMelBands = r.ReadInt32(); _options.TransformerDim = r.ReadInt32(); _options.NumTransformerLayers = r.ReadInt32();
        _options.NumStems = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32(); _options.Sources = new string[n]; for (int i = 0; i < n; i++) _options.Sources[i] = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MelBandRoFormer<T>(Architecture, _options);

    #endregion

    #region Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
    }

    private SourceSeparationResult<T> BuildResult(Tensor<T> audio, Tensor<T> magnitude, Tensor<T> masks)
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
        return new SourceSeparationResult<T> { Sources = sources, OriginalMix = audio, SampleRate = _options.SampleRate, Duration = audio.Length / (double)_options.SampleRate };
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MelBandRoFormer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
