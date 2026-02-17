using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Danna-Sep music source separation model using dual-path attention networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Danna-Sep (2024) uses a dual-path attention network with a novel aggregation strategy
/// for music source separation. It processes audio in both time and frequency dimensions
/// using interleaved attention blocks, achieving competitive SDR scores with efficient
/// computation on the MUSDB18 benchmark.
/// </para>
/// <para>
/// <b>For Beginners:</b> Danna-Sep takes a mixed song and separates it into individual
/// instruments (vocals, drums, bass, other). It works by analyzing the audio from two
/// perspectives - time patterns and frequency patterns - to figure out which parts of
/// the mix belong to which instrument.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 1025);
/// var model = new DannaSep&lt;float&gt;(arch, "danna_sep.onnx");
/// var result = model.Separate(mixedAudio);
/// var vocals = result.GetSource("vocals");
/// </code>
/// </para>
/// </remarks>
public class DannaSep<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    #region Fields

    private readonly DannaSepOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a Danna-Sep model in ONNX inference mode.</summary>
    public DannaSep(NeuralNetworkArchitecture<T> architecture, string modelPath, DannaSepOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DannaSepOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>Creates a Danna-Sep model in native training mode.</summary>
    public DannaSep(NeuralNetworkArchitecture<T> architecture, DannaSepOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DannaSepOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    internal static async Task<DannaSep<T>> CreateAsync(DannaSepOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new DannaSepOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("danna_sep", "danna_sep.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins * options.NumSources);
        return new DannaSep<T>(arch, mp, options);
    }

    #endregion

    #region IMusicSourceSeparator Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedSources => _options.SourceNames;

    /// <inheritdoc />
    public int NumStems => _options.NumSources;

    #endregion

    #region IMusicSourceSeparator Methods

    /// <inheritdoc />
    public SourceSeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var stft = ComputeSTFT(audio);
        Tensor<T> masks;
        if (IsOnnxMode && OnnxEncoder is not null) masks = OnnxEncoder.Run(stft);
        else masks = Predict(stft);
        return BuildSeparationResult(audio, stft, masks);
    }

    /// <inheritdoc />
    public Task<SourceSeparationResult<T>> SeparateAsync(Tensor<T> audio, CancellationToken ct = default)
        => Task.Run(() => Separate(audio), ct);

    /// <inheritdoc />
    public Tensor<T> ExtractSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        return result.GetSource(source);
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
    public Tensor<T> GetSourceMask(Tensor<T> audio, string source)
    {
        var stft = ComputeSTFT(audio);
        Tensor<T> masks;
        if (IsOnnxMode && OnnxEncoder is not null) masks = OnnxEncoder.Run(stft);
        else masks = Predict(stft);
        int idx = Array.IndexOf(_options.SourceNames, source);
        if (idx < 0) throw new ArgumentException($"Unknown source: {source}");
        int frameCount = stft.Shape[0];
        var mask = new Tensor<T>([frameCount, _options.NumFreqBins]);
        for (int f = 0; f < frameCount; f++)
            for (int b = 0; b < _options.NumFreqBins; b++)
            {
                int maskIdx = f * _options.NumFreqBins * _options.NumSources + idx * _options.NumFreqBins + b;
                mask[f, b] = maskIdx < masks.Length ? masks[maskIdx] : NumOps.Zero;
            }
        return mask;
    }

    /// <inheritdoc />
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

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultDannaSepLayers(
            encoderDim: _options.EncoderDim, numDualPathBlocks: _options.NumDualPathBlocks,
            chunkSize: _options.ChunkSize, numHeads: _options.NumHeads,
            numSources: _options.NumSources, numFreqBins: _options.NumFreqBins,
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

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "DannaSep-Native" : "DannaSep-ONNX",
            Description = "Danna-Sep dual-path attention music source separation (2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumDualPathBlocks
        };
        m.AdditionalInfo["NumSources"] = _options.NumSources.ToString();
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.EncoderDim); w.Write(_options.NumDualPathBlocks);
        w.Write(_options.ChunkSize); w.Write(_options.NumHeads); w.Write(_options.NumSources);
        w.Write(_options.SourceNames.Length);
        foreach (var s in _options.SourceNames) w.Write(s);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumDualPathBlocks = r.ReadInt32();
        _options.ChunkSize = r.ReadInt32(); _options.NumHeads = r.ReadInt32(); _options.NumSources = r.ReadInt32();
        int numNames = r.ReadInt32();
        var names = new string[numNames]; for (int i = 0; i < numNames; i++) names[i] = r.ReadString();
        _options.SourceNames = names;
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new DannaSep<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
    }

    private SourceSeparationResult<T> BuildSeparationResult(Tensor<T> audio, Tensor<T> magnitude, Tensor<T> masks)
    {
        var sources = new Dictionary<string, Tensor<T>>();
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        for (int s = 0; s < _options.NumSources; s++)
        {
            var maskedMag = new Tensor<T>(magnitude.Shape);
            for (int f = 0; f < numFrames; f++)
                for (int b = 0; b < numBins; b++)
                {
                    int maskIdx = f * numBins * _options.NumSources + s * numBins + b;
                    T maskVal = maskIdx < masks.Length ? masks[maskIdx] : NumOps.Zero;
                    maskedMag[f, b] = NumOps.Multiply(magnitude[f, b], maskVal);
                }
            string name = s < _options.SourceNames.Length ? _options.SourceNames[s] : $"source_{s}";
            if (_lastPhase is not null)
                sources[name] = _stft.InverseFromMagnitudeAndPhase(maskedMag, _lastPhase, audio.Length);
            else
                sources[name] = new Tensor<T>([audio.Length]);
        }
        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DannaSep<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
