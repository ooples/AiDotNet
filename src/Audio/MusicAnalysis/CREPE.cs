using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// CREPE (Convolutional Representation for Pitch Estimation) neural pitch detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CREPE (Kim et al., 2018) is a deep convolutional network for monophonic pitch detection.
/// It operates directly on the audio waveform (1024 samples at 16 kHz) and outputs a
/// 360-dimensional vector representing pitch salience across 20-cent bins from C1 to B7.
/// CREPE outperforms traditional methods (YIN, pYIN) especially in noisy conditions.
/// </para>
/// <para>
/// <b>For Beginners:</b> CREPE listens to a single voice or instrument and tells you exactly
/// what note is being played. It's like a very accurate guitar tuner powered by AI. It takes
/// small chunks of audio and outputs which pitch (frequency) is most likely present.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1024, outputSize: 360);
/// var model = new CREPE&lt;float&gt;(arch, "crepe_full.onnx");
/// var (hasPitch, pitch) = model.DetectPitch(audioFrame);
/// Console.WriteLine($"Pitch: {pitch} Hz");
/// </code>
/// </para>
/// </remarks>
public class CREPE<T> : AudioNeuralNetworkBase<T>, IPitchDetector<T>
{
    #region Fields

    private readonly CREPEOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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
    /// Creates a CREPE model in ONNX inference mode.
    /// </summary>
    public CREPE(NeuralNetworkArchitecture<T> architecture, string modelPath, CREPEOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CREPEOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        MinPitch = _options.MinFrequency;
        MaxPitch = _options.MaxFrequency;
        InitializeLayers();
    }

    /// <summary>
    /// Creates a CREPE model in native training mode.
    /// </summary>
    public CREPE(NeuralNetworkArchitecture<T> architecture, CREPEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CREPEOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        MinPitch = _options.MinFrequency;
        MaxPitch = _options.MaxFrequency;
        InitializeLayers();
    }

    internal static async Task<CREPE<T>> CreateAsync(CREPEOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CREPEOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("crepe", $"crepe_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.FrameSize, outputSize: options.NumBins);
        return new CREPE<T>(arch, mp, options);
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
        Tensor<T> logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audioFrame) : Predict(audioFrame);

        // Find the bin with maximum activation
        int bestBin = 0;
        double bestVal = double.NegativeInfinity;
        for (int i = 0; i < Math.Min(_options.NumBins, logits.Length); i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > bestVal) { bestVal = v; bestBin = i; }
        }

        // Compute confidence via softmax normalization
        double sumExp = 0;
        for (int i = 0; i < Math.Min(_options.NumBins, logits.Length); i++)
            sumExp += Math.Exp(NumOps.ToDouble(logits[i]) - bestVal);
        double confidence = 1.0 / sumExp;

        if (confidence < _options.VoicingThreshold) return null;

        // Convert bin to frequency using weighted average around peak
        double pitchHz = BinToFrequency(bestBin, logits);

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
        int hopSamples = _options.SampleRate * hopSizeMs / 1000;

        for (int start = 0; start + _options.FrameSize <= audio.Length; start += hopSamples)
        {
            var frame = new Tensor<T>([_options.FrameSize]);
            for (int i = 0; i < _options.FrameSize; i++) frame[i] = audio[start + i];

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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultCREPELayers(
            frameSize: _options.FrameSize, capacityMultiplier: _options.CapacityMultiplier,
            numBins: _options.NumBins, dropoutRate: _options.DropoutRate));
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
        // CREPE operates on raw waveform, no mel spectrogram needed
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CREPE-Native" : "CREPE-ONNX",
            Description = $"CREPE {_options.Variant} pitch detection (Kim et al., 2018)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FrameSize,
            Complexity = 6
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumBins"] = _options.NumBins.ToString();
        m.AdditionalInfo["CapacityMultiplier"] = _options.CapacityMultiplier.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FrameSize); w.Write(_options.HopLength);
        w.Write(_options.CapacityMultiplier); w.Write(_options.NumBins);
        w.Write(_options.MinFrequency); w.Write(_options.MaxFrequency);
        w.Write(_options.VoicingThreshold); w.Write(_options.DropoutRate);
        w.Write(_options.Variant);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FrameSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.CapacityMultiplier = r.ReadInt32(); _options.NumBins = r.ReadInt32();
        _options.MinFrequency = r.ReadDouble(); _options.MaxFrequency = r.ReadDouble();
        _options.VoicingThreshold = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        _options.Variant = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new CREPE<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private double BinToFrequency(int bin, Tensor<T> logits)
    {
        // Weighted average around the peak for sub-bin accuracy
        double centsBin = bin * _options.CentsPerBin;
        double weightedSum = 0;
        double weightTotal = 0;
        int windowSize = 4;

        for (int i = Math.Max(0, bin - windowSize); i <= Math.Min(_options.NumBins - 1, bin + windowSize); i++)
        {
            double w = Math.Exp(NumOps.ToDouble(logits[i]));
            weightedSum += w * (i * _options.CentsPerBin);
            weightTotal += w;
        }

        double cents = weightTotal > 0 ? weightedSum / weightTotal : centsBin;
        // C1 = MIDI 24 = 32.70 Hz, cents are relative to C1
        return _options.MinFrequency * Math.Pow(2.0, cents / 1200.0);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CREPE<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
