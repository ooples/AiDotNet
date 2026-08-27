using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SourceSeparation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Music Source Separation with Band-Split RNN", "https://doi.org/10.48550/arXiv.2309.02612", Year = 2024, Authors = "Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, Yun-Ning Hung")]
public partial class MelBandRoFormer<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// DERIVED, not stored: PredictCore folds over Layers, and the last layer
    /// CreateDefaultMelBandRoFormerLayers emits is
    /// <c>DenseLayer&lt;T&gt;(numFreqBins * numStems * 2)</c>. The trailing x2 is the COMPLEX mask
    /// (real and imaginary parts per bin). 8200 (1025 x 4 x 2 at the defaults) is stored nowhere, and
    /// note the head is sized from NumFreqBins - NOT NumMelBands, which only sets the band-split
    /// front end.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.NumFreqBins * _options.NumStems * 2;

    #region Fields

    private readonly MelBandRoFormerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    [Scratch]
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
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
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
        foreach (var kvp in sep.Sources)
        {
            double v = vols.TryGetValue(kvp.Key, out var vol) ? vol : 1.0;
            if (kvp.Value.Length == o.Length && kvp.Value.Rank == o.Rank)
            {
                var scaled = Engine.TensorMultiplyScalar(kvp.Value, NumOps.FromDouble(v));
                o = Engine.TensorAdd(o, scaled);
            }
            else
            {
                T vT = NumOps.FromDouble(v);
                for (int i = 0; i < kvp.Value.Length && i < o.Length; i++)
                    o[i] = NumOps.Add(o[i], NumOps.Multiply(kvp.Value[i], vT));
            }
        }
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

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true); try { TrainWithTape(input, expected, _optimizer); } finally { SetTrainingMode(false); }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "MelBand-RoFormer-Native" : "MelBand-RoFormer-ONNX", Description = "MelBand-RoFormer Mel-Band Source Separation (2024)", FeatureCount = _options.NumFreqBins, Complexity = _options.NumTransformerLayers };
        m.AdditionalInfo["NumMelBands"] = _options.NumMelBands.ToString(); m.AdditionalInfo["NumStems"] = _options.NumStems.ToString();
        return m;
    }





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
            var maskedMag = new Tensor<T>(magnitude._shape);
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
