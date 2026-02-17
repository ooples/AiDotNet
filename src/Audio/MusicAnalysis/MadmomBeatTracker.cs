using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Madmom-style neural beat tracker using bidirectional RNNs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Madmom beat tracking system (Bock et al., 2016) uses a recurrent neural network to detect
/// beat positions and downbeat positions in audio. It combines spectrogram features with bidirectional
/// RNNs and a dynamic Bayesian network for beat tracking, achieving state-of-the-art results on
/// multiple beat tracking benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to music and finds exactly where each beat falls,
/// like a musician tapping their foot in time. It can tell you the tempo (beats per minute) and
/// mark every beat position, which is essential for music synchronization, DJ software, and
/// automatic remixing.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 81, outputSize: 1);
/// var model = new MadmomBeatTracker&lt;float&gt;(arch, "madmom_beat.onnx");
/// var result = model.Track(audioTensor);
/// Console.WriteLine($"Tempo: {result.Tempo} BPM, {result.NumBeats} beats");
/// </code>
/// </para>
/// </remarks>
public class MadmomBeatTracker<T> : AudioNeuralNetworkBase<T>, IBeatTracker<T>
{
    #region Fields

    private readonly MadmomBeatTrackerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IBeatTracker Properties

    /// <inheritdoc />
    public double MinBPM => 30;

    /// <inheritdoc />
    public double MaxBPM => 300;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Madmom Beat Tracker in ONNX inference mode.
    /// </summary>
    public MadmomBeatTracker(NeuralNetworkArchitecture<T> architecture, string modelPath, MadmomBeatTrackerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MadmomBeatTrackerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Madmom Beat Tracker in native training mode.
    /// </summary>
    public MadmomBeatTracker(NeuralNetworkArchitecture<T> architecture, MadmomBeatTrackerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MadmomBeatTrackerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MadmomBeatTracker<T>> CreateAsync(MadmomBeatTrackerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MadmomBeatTrackerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("madmom_beat", "madmom_beat_tracker.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumBands, outputSize: 1);
        return new MadmomBeatTracker<T>(arch, mp, options);
    }

    #endregion

    #region IBeatTracker

    /// <inheritdoc />
    public BeatTrackingResult<T> Track(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var onsetStrength = ComputeOnsetStrength(audio);
        return ExtractBeatsFromActivation(onsetStrength, audio.Length);
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
        var result = Track(audio);
        return result.Tempo;
    }

    /// <inheritdoc />
    public IReadOnlyList<TempoHypothesis<T>> GetTempoHypotheses(Tensor<T> audio, int numHypotheses = 5)
    {
        ThrowIfDisposed();
        var result = Track(audio);
        double primaryBpm = NumOps.ToDouble(result.Tempo);
        var hypotheses = new List<TempoHypothesis<T>>
        {
            new() { Tempo = result.Tempo, Confidence = result.TempoConfidence, Relation = TempoRelation.Primary }
        };

        if (numHypotheses >= 2 && primaryBpm * 2 <= MaxBPM)
            hypotheses.Add(new TempoHypothesis<T> { Tempo = NumOps.FromDouble(primaryBpm * 2), Confidence = NumOps.FromDouble(0.3), Relation = TempoRelation.DoubleTime });
        if (numHypotheses >= 3 && primaryBpm / 2 >= MinBPM)
            hypotheses.Add(new TempoHypothesis<T> { Tempo = NumOps.FromDouble(primaryBpm / 2), Confidence = NumOps.FromDouble(0.3), Relation = TempoRelation.HalfTime });

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

        // Apply sigmoid to get beat activation
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMadmomBeatTrackerLayers(
            numBands: _options.NumBands, rnnHiddenSize: _options.RnnHiddenSize,
            numRnnLayers: _options.NumRnnLayers, dropoutRate: _options.DropoutRate));
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

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MadmomBeatTracker-Native" : "MadmomBeatTracker-ONNX",
            Description = "Madmom neural beat tracker (Bock et al., 2016)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumBands,
            Complexity = _options.NumRnnLayers
        };
        m.AdditionalInfo["RnnHiddenSize"] = _options.RnnHiddenSize.ToString();
        m.AdditionalInfo["PeakThreshold"] = _options.PeakThreshold.ToString("F2");
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumBands); w.Write(_options.RnnHiddenSize); w.Write(_options.NumRnnLayers);
        w.Write(_options.PeakThreshold); w.Write(_options.MinBeatInterval); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumBands = r.ReadInt32(); _options.RnnHiddenSize = r.ReadInt32(); _options.NumRnnLayers = r.ReadInt32();
        _options.PeakThreshold = r.ReadDouble(); _options.MinBeatInterval = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MadmomBeatTracker<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private BeatTrackingResult<T> ExtractBeatsFromActivation(Tensor<T> activation, int audioLength)
    {
        double threshold = _options.PeakThreshold;
        double minInterval = _options.MinBeatInterval;
        double frameTime = (double)_options.HopLength / _options.SampleRate;
        var beatTimes = new List<double>();
        var beatConfidences = new List<T>();
        double lastBeat = double.NegativeInfinity;

        for (int i = 1; i < activation.Length - 1; i++)
        {
            double val = NumOps.ToDouble(activation[i]);
            double prev = NumOps.ToDouble(activation[Math.Max(0, i - 1)]);
            double next = NumOps.ToDouble(activation[Math.Min(activation.Length - 1, i + 1)]);
            double time = i * frameTime;

            if (val > threshold && val >= prev && val >= next && (time - lastBeat) >= minInterval)
            {
                beatTimes.Add(time);
                beatConfidences.Add(NumOps.FromDouble(val));
                lastBeat = time;
            }
        }

        double bpm = 0;
        if (beatTimes.Count >= 2)
        {
            double totalInterval = beatTimes[beatTimes.Count - 1] - beatTimes[0];
            bpm = (beatTimes.Count - 1) / totalInterval * 60.0;
        }

        return new BeatTrackingResult<T>
        {
            Tempo = NumOps.FromDouble(bpm),
            TempoConfidence = NumOps.FromDouble(beatTimes.Count > 0 ? 0.8 : 0.0),
            BeatTimes = beatTimes,
            BeatConfidences = beatConfidences,
            BeatInterval = bpm > 0 ? 60.0 / bpm : 0
        };
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MadmomBeatTracker<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
