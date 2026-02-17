using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// MT3 (Multi-Track Music Transcription) model using T5-style encoder-decoder architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MT3 (Gardner et al., 2022, Google) is a Transformer-based model that transcribes polyphonic
/// audio into MIDI across multiple instruments simultaneously. It uses a T5-style encoder-decoder
/// architecture with spectrogram input and tokenized MIDI output, achieving state-of-the-art
/// multi-instrument transcription on the Slakh2100 dataset.
/// </para>
/// <para>
/// <b>For Beginners:</b> MT3 listens to a full song with multiple instruments and writes out
/// the sheet music (as MIDI) for each instrument separately. It can tell which notes the piano
/// is playing while also transcribing the guitar, drums, and bass at the same time.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 512, outputSize: 6000);
/// var model = new MT3&lt;float&gt;(arch, "mt3.onnx");
/// var notes = model.Transcribe(audioTensor);
/// foreach (var note in notes)
///     Console.WriteLine($"{note.NoteName}: {note.StartTime:F2}s - {note.EndTime:F2}s");
/// </code>
/// </para>
/// </remarks>
public class MT3<T> : AudioNeuralNetworkBase<T>, IMusicTranscriber<T>
{
    #region Fields

    private readonly MT3Options _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IMusicTranscriber Properties

    /// <inheritdoc />
    public int NumMidiNotes => 88;

    /// <inheritdoc />
    public int MidiOffset => 21;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an MT3 model in ONNX inference mode.
    /// </summary>
    public MT3(NeuralNetworkArchitecture<T> architecture, string modelPath, MT3Options? options = null)
        : base(architecture)
    {
        _options = options ?? new MT3Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an MT3 model in native training mode.
    /// </summary>
    public MT3(NeuralNetworkArchitecture<T> architecture, MT3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MT3Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MT3<T>> CreateAsync(MT3Options? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MT3Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("mt3", "mt3.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new MT3<T>(arch, mp, options);
    }

    #endregion

    #region IMusicTranscriber

    /// <inheritdoc />
    public IReadOnlyList<TranscribedNote<T>> Transcribe(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var frames = GetFrameActivations(audio);
        var onsets = GetOnsetActivations(audio);
        return ExtractNotes(frames, onsets);
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

        // Decode token logits into note activations
        int noteCount = NumMidiNotes;
        var frames = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && i < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            frames[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v)));
        }
        return frames;
    }

    /// <inheritdoc />
    public Tensor<T> GetOnsetActivations(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        int noteCount = NumMidiNotes;
        var onsets = new Tensor<T>([noteCount]);
        for (int i = 0; i < noteCount && (i + noteCount) < output.Length; i++)
        {
            double v = NumOps.ToDouble(output[i + noteCount]);
            onsets[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v)));
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
        int noteCount = Math.Min(NumMidiNotes, Math.Min(frameActivations.Length, onsetActivations.Length));

        for (int note = 0; note < noteCount; note++)
        {
            double noteProb = NumOps.ToDouble(frameActivations[note]);
            double onsetProb = NumOps.ToDouble(onsetActivations[note]);

            if (noteProb >= frameThreshold && onsetProb >= onsetThreshold)
            {
                int midiNote = note + MidiOffset;
                notes.Add(new TranscribedNote<T>
                {
                    StartTime = 0,
                    EndTime = Math.Max(0.05, frameTimeSec),
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMT3Layers(
            numMels: _options.NumMels, encoderDim: _options.EncoderDim,
            numEncoderLayers: _options.NumEncoderLayers, decoderDim: _options.DecoderDim,
            numDecoderLayers: _options.NumDecoderLayers, numAttentionHeads: _options.NumAttentionHeads,
            vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "MT3-Native" : "MT3-ONNX",
            Description = "MT3 multi-track music transcription (Gardner et al., 2022, Google)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        m.AdditionalInfo["MaxInstruments"] = _options.MaxInstruments.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers);
        w.Write(_options.DecoderDim); w.Write(_options.NumDecoderLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.VocabSize);
        w.Write(_options.MaxInstruments); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.DecoderDim = r.ReadInt32(); _options.NumDecoderLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.VocabSize = r.ReadInt32();
        _options.MaxInstruments = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MT3<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MT3<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
