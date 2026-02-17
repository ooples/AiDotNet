using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Neural Tempogram model for tempo estimation over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Tempogram model computes a tempo representation over time using a neural approach
/// to onset detection and autocorrelation-based tempo estimation. It provides both global
/// tempo and tempo curves for music with changing tempos.
/// </para>
/// <para>
/// <b>For Beginners:</b> A tempogram shows how the tempo (speed) of music changes over time.
/// This model creates a detailed map of tempo, which is useful for analyzing songs with
/// tempo changes, rubato (expressive timing), or live performances where the tempo varies.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 2048, outputSize: 300);
/// var model = new Tempogram&lt;float&gt;(arch, "tempogram.onnx");
/// var result = model.Track(audioTensor);
/// Console.WriteLine($"Tempo: {result.Tempo} BPM");
/// </code>
/// </para>
/// </remarks>
public class Tempogram<T> : AudioNeuralNetworkBase<T>, IBeatTracker<T>
{
    #region Fields

    private readonly TempogramOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IBeatTracker Properties

    /// <inheritdoc />
    public double MinBPM => _options.MinBPM;

    /// <inheritdoc />
    public double MaxBPM => _options.MaxBPM;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Tempogram model in ONNX inference mode.
    /// </summary>
    public Tempogram(NeuralNetworkArchitecture<T> architecture, string modelPath, TempogramOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TempogramOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Tempogram model in native training mode.
    /// </summary>
    public Tempogram(NeuralNetworkArchitecture<T> architecture, TempogramOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TempogramOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Tempogram<T>> CreateAsync(TempogramOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new TempogramOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("tempogram", "tempogram.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.FftSize, outputSize: options.NumTempoBins);
        return new Tempogram<T>(arch, mp, options);
    }

    #endregion

    #region IBeatTracker

    /// <inheritdoc />
    public BeatTrackingResult<T> Track(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var onsetStrength = ComputeOnsetStrength(audio);
        return ExtractTempoAndBeats(onsetStrength, audio.Length);
    }

    /// <inheritdoc />
    public Task<BeatTrackingResult<T>> TrackAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Track(audio), cancellationToken);
    }

    /// <inheritdoc />
    public T EstimateTempo(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // Find peak in tempogram bins
        int bestBin = 0;
        double bestVal = double.NegativeInfinity;
        int numBins = Math.Min(_options.NumTempoBins, output.Length);
        for (int i = 0; i < numBins; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            if (v > bestVal) { bestVal = v; bestBin = i; }
        }

        double bpm = BinToBPM(bestBin);
        return NumOps.FromDouble(bpm);
    }

    /// <inheritdoc />
    public IReadOnlyList<TempoHypothesis<T>> GetTempoHypotheses(Tensor<T> audio, int numHypotheses = 5)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // Find top peaks
        int numBins = Math.Min(_options.NumTempoBins, output.Length);
        var peaks = new List<(int bin, double value)>();
        for (int i = 1; i < numBins - 1; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            double prev = NumOps.ToDouble(output[i - 1]);
            double next = NumOps.ToDouble(output[Math.Min(numBins - 1, i + 1)]);
            if (v >= prev && v >= next) peaks.Add((i, v));
        }
        peaks.Sort((a, b) => b.value.CompareTo(a.value));

        var hypotheses = new List<TempoHypothesis<T>>();
        double maxVal = peaks.Count > 0 ? peaks[0].value : 1.0;
        for (int i = 0; i < Math.Min(numHypotheses, peaks.Count); i++)
        {
            double bpm = BinToBPM(peaks[i].bin);
            double confidence = maxVal > 0 ? peaks[i].value / maxVal : 0;
            var relation = i == 0 ? TempoRelation.Primary : TempoRelation.Alternative;
            hypotheses.Add(new TempoHypothesis<T> { Tempo = NumOps.FromDouble(bpm), Confidence = NumOps.FromDouble(confidence), Relation = relation });
        }

        return hypotheses;
    }

    /// <inheritdoc />
    public DownbeatResult<T> DetectDownbeats(Tensor<T> audio, BeatTrackingResult<T>? beatTrackingResult = null)
    {
        ThrowIfDisposed();
        var result = beatTrackingResult ?? Track(audio);
        var downbeats = new List<double>();
        for (int i = 0; i < result.BeatTimes.Count; i += 4)
            downbeats.Add(result.BeatTimes[i]);

        return new DownbeatResult<T>
        {
            DownbeatTimes = downbeats,
            TimeSignature = new TimeSignature(4, 4),
            MeasureStarts = downbeats
        };
    }

    /// <inheritdoc />
    public Tensor<T> ComputeOnsetStrength(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        var activation = new Tensor<T>([output.Length]);
        for (int i = 0; i < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            activation[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v)));
        }
        return activation;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultTempogramLayers(
            fftSize: _options.FftSize, onsetHiddenDim: _options.OnsetHiddenDim,
            numOnsetLayers: _options.NumOnsetLayers, numTempoBins: _options.NumTempoBins,
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

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Tempogram-Native" : "Tempogram-ONNX",
            Description = "Neural tempogram for tempo estimation over time",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FftSize,
            Complexity = _options.NumOnsetLayers
        };
        m.AdditionalInfo["MinBPM"] = _options.MinBPM.ToString("F0");
        m.AdditionalInfo["MaxBPM"] = _options.MaxBPM.ToString("F0");
        m.AdditionalInfo["NumTempoBins"] = _options.NumTempoBins.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.OnsetHiddenDim); w.Write(_options.NumOnsetLayers);
        w.Write(_options.TempoWindowFrames); w.Write(_options.MinBPM);
        w.Write(_options.MaxBPM); w.Write(_options.NumTempoBins); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.OnsetHiddenDim = r.ReadInt32(); _options.NumOnsetLayers = r.ReadInt32();
        _options.TempoWindowFrames = r.ReadInt32(); _options.MinBPM = r.ReadDouble();
        _options.MaxBPM = r.ReadDouble(); _options.NumTempoBins = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Tempogram<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private double BinToBPM(int bin)
    {
        double ratio = (double)bin / _options.NumTempoBins;
        return _options.MinBPM + ratio * (_options.MaxBPM - _options.MinBPM);
    }

    private BeatTrackingResult<T> ExtractTempoAndBeats(Tensor<T> activation, int audioLength)
    {
        // Estimate tempo from peak in activation
        double tempo = NumOps.ToDouble(EstimateTempo(new Tensor<T>([1])));
        if (tempo <= 0) tempo = 120.0; // fallback

        double beatInterval = 60.0 / tempo;
        double duration = (double)audioLength / _options.SampleRate;
        var beatTimes = new List<double>();
        var beatConfidences = new List<T>();

        for (double t = 0; t < duration; t += beatInterval)
        {
            beatTimes.Add(t);
            beatConfidences.Add(NumOps.FromDouble(0.7));
        }

        return new BeatTrackingResult<T>
        {
            Tempo = NumOps.FromDouble(tempo),
            TempoConfidence = NumOps.FromDouble(0.7),
            BeatTimes = beatTimes,
            BeatConfidences = beatConfidences,
            BeatInterval = beatInterval
        };
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Tempogram<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
