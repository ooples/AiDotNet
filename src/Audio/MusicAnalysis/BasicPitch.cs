using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Basic Pitch polyphonic music transcription model from Spotify.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Basic Pitch (Bittner et al., 2022) is a lightweight CNN for polyphonic music transcription.
/// It produces three outputs: note activations, onset activations, and pitch contour. Combined,
/// these produce MIDI-like note events. The model is fast enough for real-time use and handles
/// multiple simultaneous instruments.
/// </para>
/// <para>
/// <b>For Beginners:</b> Basic Pitch is like a music-to-MIDI converter. You feed it a recording
/// of music (even with multiple instruments playing at once) and it outputs a list of every note
/// that was played, when it started, when it stopped, and how loud it was. This is called
/// "polyphonic transcription" because it handles multiple notes at the same time.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 264, outputSize: 264);
/// var model = new BasicPitch&lt;float&gt;(arch, "basic_pitch.onnx");
/// var notes = model.Transcribe(audioTensor);
/// foreach (var note in notes)
///     Console.WriteLine($"{note.NoteName}: {note.StartTime:F2}s - {note.EndTime:F2}s");
/// </code>
/// </para>
/// </remarks>
public class BasicPitch<T> : AudioNeuralNetworkBase<T>, IMusicTranscriber<T>
{
    #region Fields

    private readonly BasicPitchOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IMusicTranscriber Properties

    /// <inheritdoc />
    public int NumMidiNotes => _options.NumMidiNotes;

    /// <inheritdoc />
    public int MidiOffset => _options.MidiOffset;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Basic Pitch model in ONNX inference mode.
    /// </summary>
    public BasicPitch(NeuralNetworkArchitecture<T> architecture, string modelPath, BasicPitchOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new BasicPitchOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Basic Pitch model in native training mode.
    /// </summary>
    public BasicPitch(NeuralNetworkArchitecture<T> architecture, BasicPitchOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BasicPitchOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<BasicPitch<T>> CreateAsync(BasicPitchOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new BasicPitchOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("basic_pitch", "basic_pitch.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumHarmonicBins, outputSize: options.NumMidiNotes * 3);
        return new BasicPitch<T>(arch, mp, options);
    }

    #endregion

    #region IMusicTranscriber

    /// <inheritdoc />
    public IReadOnlyList<TranscribedNote<T>> Transcribe(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var frames = GetFrameActivations(audio);
        var onsets = GetOnsetActivations(audio);
        return ExtractNotes(frames, onsets, _options.NoteThreshold, _options.OnsetThreshold);
    }

    /// <inheritdoc />
    public Task<IReadOnlyList<TranscribedNote<T>>> TranscribeAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Transcribe(audio), cancellationToken);
    }

    /// <inheritdoc />
    public Tensor<T> GetFrameActivations(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // First NumMidiNotes values are note activations
        int noteCount = _options.NumMidiNotes;
        var frames = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && i < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            frames[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v))); // sigmoid
        }
        return frames;
    }

    /// <inheritdoc />
    public Tensor<T> GetOnsetActivations(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // Second NumMidiNotes values are onset activations
        int noteCount = _options.NumMidiNotes;
        var onsets = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && (i + noteCount) < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i + noteCount]);
            onsets[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v))); // sigmoid
        }
        return onsets;
    }

    /// <inheritdoc />
    public IReadOnlyList<TranscribedNote<T>> ExtractNotes(
        Tensor<T> frameActivations, Tensor<T> onsetActivations,
        double frameThreshold = 0.5, double onsetThreshold = 0.5)
    {
        ThrowIfDisposed();
        var notes = new List<TranscribedNote<T>>();
        double frameTimeSec = (double)_options.HopLength / _options.SampleRate;
        int noteCount = Math.Min(_options.NumMidiNotes, Math.Min(frameActivations.Length, onsetActivations.Length));

        for (int note = 0; note < noteCount; note++)
        {
            double noteProb = NumOps.ToDouble(frameActivations[note]);
            double onsetProb = NumOps.ToDouble(onsetActivations[note]);

            if (noteProb >= frameThreshold && onsetProb >= onsetThreshold)
            {
                int midiNote = note + _options.MidiOffset;
                double startTime = 0;
                double endTime = startTime + Math.Max(_options.MinNoteDurationSec, frameTimeSec);

                notes.Add(new TranscribedNote<T>
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    MidiNote = midiNote,
                    Confidence = NumOps.FromDouble(noteProb)
                });
            }
        }
        return notes;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultBasicPitchLayers(
            numHarmonicBins: _options.NumHarmonicBins, encoderFilters: _options.EncoderFilters,
            numEncoderLayers: _options.NumEncoderLayers, numMidiNotes: _options.NumMidiNotes,
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
            Name = _useNativeMode ? "BasicPitch-Native" : "BasicPitch-ONNX",
            Description = "Basic Pitch polyphonic music transcription (Bittner et al., 2022, Spotify)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumHarmonicBins,
            Complexity = _options.NumEncoderLayers
        };
        m.AdditionalInfo["NumMidiNotes"] = _options.NumMidiNotes.ToString();
        m.AdditionalInfo["EncoderFilters"] = _options.EncoderFilters.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumHarmonicBins); w.Write(_options.BinsPerOctave);
        w.Write(_options.HopLength); w.Write(_options.NumHarmonics);
        w.Write(_options.NumMidiNotes); w.Write(_options.MidiOffset);
        w.Write(_options.EncoderFilters); w.Write(_options.NumEncoderLayers);
        w.Write(_options.OnsetThreshold); w.Write(_options.NoteThreshold);
        w.Write(_options.MinNoteDurationSec); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumHarmonicBins = r.ReadInt32(); _options.BinsPerOctave = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.NumHarmonics = r.ReadInt32();
        _options.NumMidiNotes = r.ReadInt32(); _options.MidiOffset = r.ReadInt32();
        _options.EncoderFilters = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.OnsetThreshold = r.ReadDouble(); _options.NoteThreshold = r.ReadDouble();
        _options.MinNoteDurationSec = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new BasicPitch<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BasicPitch<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
