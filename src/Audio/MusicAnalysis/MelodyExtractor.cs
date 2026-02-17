using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Neural Melody Extractor that identifies the primary melodic line from polyphonic audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Melody Extractor identifies the primary melodic line from a polyphonic audio recording
/// using a neural network. Unlike pitch detection (which finds any pitch), melody extraction
/// specifically tracks the dominant melody even when other instruments are playing simultaneously.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you listen to a song, you can usually hum along to the main
/// melody even though many instruments are playing. This model does the same thing - it finds
/// and extracts just the main tune from a full song, ignoring background harmonies and rhythms.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 128, outputSize: 360);
/// var model = new MelodyExtractor&lt;float&gt;(arch, "melody_extractor.onnx");
/// var (hasPitch, pitch) = model.DetectPitch(audioFrame);
/// Console.WriteLine($"Melody pitch: {pitch} Hz");
/// </code>
/// </para>
/// </remarks>
public class MelodyExtractor<T> : AudioNeuralNetworkBase<T>, IPitchDetector<T>
{
    #region Fields

    private readonly MelodyExtractorOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IPitchDetector Properties

    /// <inheritdoc />
    public double MinPitch { get; set; }

    /// <inheritdoc />
    public double MaxPitch { get; set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Melody Extractor in ONNX inference mode.
    /// </summary>
    public MelodyExtractor(NeuralNetworkArchitecture<T> architecture, string modelPath, MelodyExtractorOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MelodyExtractorOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        MinPitch = _options.MinFrequency;
        MaxPitch = _options.MaxFrequency;
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Melody Extractor in native training mode.
    /// </summary>
    public MelodyExtractor(NeuralNetworkArchitecture<T> architecture, MelodyExtractorOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MelodyExtractorOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        MinPitch = _options.MinFrequency;
        MaxPitch = _options.MaxFrequency;
        InitializeLayers();
    }

    internal static async Task<MelodyExtractor<T>> CreateAsync(MelodyExtractorOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MelodyExtractorOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("melody_extractor", "melody_extractor.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumPitchBins);
        return new MelodyExtractor<T>(arch, mp, options);
    }

    #endregion

    #region IPitchDetector

    /// <inheritdoc />
    public (bool HasPitch, T Pitch) DetectPitch(Tensor<T> audioFrame)
    {
        ThrowIfDisposed();
        var result = DetectPitchWithConfidence(audioFrame);
        if (result is null) return (false, NumOps.Zero);
        return (true, result.Value.Pitch);
    }

    /// <inheritdoc />
    public (T Pitch, T Confidence)? DetectPitchWithConfidence(Tensor<T> audioFrame)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audioFrame);
        Tensor<T> logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        int bestBin = 0;
        double bestVal = double.NegativeInfinity;
        for (int i = 0; i < Math.Min(_options.NumPitchBins, logits.Length); i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > bestVal) { bestVal = v; bestBin = i; }
        }

        // Confidence via softmax
        double sumExp = 0;
        for (int i = 0; i < Math.Min(_options.NumPitchBins, logits.Length); i++)
            sumExp += Math.Exp(NumOps.ToDouble(logits[i]) - bestVal);
        double confidence = 1.0 / sumExp;

        if (confidence < _options.VoicingThreshold) return null;

        double pitchHz = BinToFrequency(bestBin);
        if (pitchHz < MinPitch || pitchHz > MaxPitch) return null;

        return (NumOps.FromDouble(pitchHz), NumOps.FromDouble(confidence));
    }

    /// <inheritdoc />
    public T[] ExtractPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        ThrowIfDisposed();
        var detailed = ExtractDetailedPitchContour(audio, hopSizeMs);
        var result = new T[detailed.Count];
        for (int i = 0; i < detailed.Count; i++)
            result[i] = detailed[i].IsVoiced ? detailed[i].Pitch : NumOps.Zero;
        return result;
    }

    /// <inheritdoc />
    public IReadOnlyList<PitchFrame<T>> ExtractDetailedPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        ThrowIfDisposed();
        var results = new List<PitchFrame<T>>();
        int frameSize = _options.FftSize;
        int hopSamples = _options.SampleRate * hopSizeMs / 1000;

        for (int start = 0; start + frameSize <= audio.Length; start += hopSamples)
        {
            var frame = new Tensor<T>([frameSize]);
            for (int i = 0; i < frameSize; i++) frame[i] = audio[start + i];

            var detection = DetectPitchWithConfidence(frame);
            double time = start / (double)_options.SampleRate;

            if (detection is not null)
            {
                results.Add(new PitchFrame<T>
                {
                    Time = time,
                    Pitch = detection.Value.Pitch,
                    Confidence = detection.Value.Confidence,
                    IsVoiced = true
                });
            }
            else
            {
                results.Add(new PitchFrame<T>
                {
                    Time = time,
                    Pitch = NumOps.Zero,
                    Confidence = NumOps.Zero,
                    IsVoiced = false
                });
            }
        }
        return results;
    }

    /// <inheritdoc />
    public double PitchToMidi(T pitchHz)
    {
        double hz = NumOps.ToDouble(pitchHz);
        if (hz <= 0) return 0;
        return 69.0 + 12.0 * (Math.Log(hz / 440.0) / Math.Log(2.0));
    }

    /// <inheritdoc />
    public T MidiToPitch(double midiNote)
    {
        return NumOps.FromDouble(440.0 * Math.Pow(2.0, (midiNote - 69.0) / 12.0));
    }

    /// <inheritdoc />
    public string PitchToNoteName(T pitchHz)
    {
        double midi = PitchToMidi(pitchHz);
        if (midi <= 0) return "N/A";
        int roundedMidi = (int)Math.Round(midi);
        string[] names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        int octave = (roundedMidi / 12) - 1;
        int noteIndex = roundedMidi % 12;
        return $"{names[noteIndex]}{octave}";
    }

    /// <inheritdoc />
    public double GetCentsDeviation(T pitchHz)
    {
        double midi = PitchToMidi(pitchHz);
        double nearestMidi = Math.Round(midi);
        return (midi - nearestMidi) * 100.0;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMelodyExtractorLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numPitchBins: _options.NumPitchBins,
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
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MelodyExtractor-Native" : "MelodyExtractor-ONNX",
            Description = "Neural melody extraction from polyphonic audio",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumPitchBins"] = _options.NumPitchBins.ToString();
        m.AdditionalInfo["MinFrequency"] = _options.MinFrequency.ToString("F1");
        m.AdditionalInfo["MaxFrequency"] = _options.MaxFrequency.ToString("F1");
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumPitchBins); w.Write(_options.MinFrequency);
        w.Write(_options.MaxFrequency); w.Write(_options.VoicingThreshold); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumPitchBins = r.ReadInt32(); _options.MinFrequency = r.ReadDouble();
        _options.MaxFrequency = r.ReadDouble(); _options.VoicingThreshold = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MelodyExtractor<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private double BinToFrequency(int bin)
    {
        // Log-spaced bins from MinFrequency to MaxFrequency
        double logMin = Math.Log(_options.MinFrequency);
        double logMax = Math.Log(_options.MaxFrequency);
        double logFreq = logMin + (double)bin / _options.NumPitchBins * (logMax - logMin);
        return Math.Exp(logFreq);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MelodyExtractor<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
