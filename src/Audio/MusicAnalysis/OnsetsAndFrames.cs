using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Onsets and Frames piano transcription model from Google Magenta.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Onsets and Frames (Hawthorne et al., 2018) jointly predicts note onsets and frame-level
/// activations for automatic piano transcription. The model uses CNN acoustic features with
/// bidirectional LSTMs, trained on the MAESTRO dataset. It achieves ~90% note F1 on piano.
/// </para>
/// <para>
/// <b>For Beginners:</b> Onsets and Frames is a piano-specific music transcriber. It listens
/// to piano music and detects every key press: when each key is pressed (onset), how long it
/// is held (frame), and which key it is (pitch). The output is a list of notes that can be
/// saved as MIDI or displayed as sheet music.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 229, outputSize: 176);
/// var model = new OnsetsAndFrames&lt;float&gt;(arch, "onsets_frames.onnx");
/// var notes = model.Transcribe(pianoAudio);
/// foreach (var note in notes)
///     Console.WriteLine($"{note.NoteName}: {note.StartTime:F2}s - {note.EndTime:F2}s");
/// </code>
/// </para>
/// </remarks>
public class OnsetsAndFrames<T> : AudioNeuralNetworkBase<T>, IMusicTranscriber<T>
{
    #region Fields

    private readonly OnsetsAndFramesOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private MelSpectrogram<T>? _melSpectrogram;
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
    /// Creates an Onsets and Frames model in ONNX inference mode.
    /// </summary>
    public OnsetsAndFrames(NeuralNetworkArchitecture<T> architecture, string modelPath, OnsetsAndFramesOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OnsetsAndFramesOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an Onsets and Frames model in native training mode.
    /// </summary>
    public OnsetsAndFrames(NeuralNetworkArchitecture<T> architecture, OnsetsAndFramesOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new OnsetsAndFramesOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        InitializeLayers();
    }

    internal static async Task<OnsetsAndFrames<T>> CreateAsync(OnsetsAndFramesOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new OnsetsAndFramesOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("onsets_frames", "onsets_frames.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumMidiNotes * 2);
        return new OnsetsAndFrames<T>(arch, mp, options);
    }

    #endregion

    #region IMusicTranscriber

    /// <inheritdoc />
    public IReadOnlyList<TranscribedNote<T>> Transcribe(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var frames = GetFrameActivations(audio);
        var onsets = GetOnsetActivations(audio);
        return ExtractNotes(frames, onsets, _options.FrameThreshold, _options.OnsetThreshold);
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

        // Second half of output is frame activations
        int noteCount = _options.NumMidiNotes;
        var frames = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && (i + noteCount) < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i + noteCount]);
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

        // First half of output is onset activations
        int noteCount = _options.NumMidiNotes;
        var onsets = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && i < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i]);
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
            double onsetProb = NumOps.ToDouble(onsetActivations[note]);
            double frameProb = NumOps.ToDouble(frameActivations[note]);

            if (onsetProb >= onsetThreshold && frameProb >= frameThreshold)
            {
                int midiNote = note + _options.MidiOffset;
                double startTime = 0;
                double endTime = startTime + Math.Max(_options.MinNoteDurationSec, frameTimeSec);
                double confidence = (onsetProb + frameProb) / 2.0;

                notes.Add(new TranscribedNote<T>
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    MidiNote = midiNote,
                    Confidence = NumOps.FromDouble(confidence)
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultOnsetsAndFramesLayers(
            numMels: _options.NumMels, acousticDim: _options.AcousticModelDim,
            lstmHiddenSize: _options.LstmHiddenSize, numLstmLayers: _options.NumLstmLayers,
            numMidiNotes: _options.NumMidiNotes, dropoutRate: _options.DropoutRate));
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
        if (_melSpectrogram is not null) return _melSpectrogram.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "OnsetsAndFrames-Native" : "OnsetsAndFrames-ONNX",
            Description = "Onsets and Frames piano transcription (Hawthorne et al., 2018, Magenta)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLstmLayers
        };
        m.AdditionalInfo["NumMidiNotes"] = _options.NumMidiNotes.ToString();
        m.AdditionalInfo["AcousticModelDim"] = _options.AcousticModelDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.FMin); w.Write(_options.FMax);
        w.Write(_options.NumMidiNotes); w.Write(_options.MidiOffset);
        w.Write(_options.AcousticModelDim); w.Write(_options.LstmHiddenSize); w.Write(_options.NumLstmLayers);
        w.Write(_options.OnsetThreshold); w.Write(_options.FrameThreshold);
        w.Write(_options.MinNoteDurationSec); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.FMin = r.ReadDouble(); _options.FMax = r.ReadDouble();
        _options.NumMidiNotes = r.ReadInt32(); _options.MidiOffset = r.ReadInt32();
        _options.AcousticModelDim = r.ReadInt32(); _options.LstmHiddenSize = r.ReadInt32(); _options.NumLstmLayers = r.ReadInt32();
        _options.OnsetThreshold = r.ReadDouble(); _options.FrameThreshold = r.ReadDouble();
        _options.MinNoteDurationSec = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels,
            _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new OnsetsAndFrames<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OnsetsAndFrames<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
