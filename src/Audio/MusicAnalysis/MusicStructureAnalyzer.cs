using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Music Structure Analyzer that segments songs into structural sections (intro, verse, chorus, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Music Structure Analyzer segments songs into structural sections (intro, verse, chorus,
/// bridge, outro) using a neural network trained on annotated music datasets. It combines
/// self-similarity matrix features with a segmentation network.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to a song and identifies its sections - where the
/// verse begins, where the chorus kicks in, and where the bridge or outro happens. It's like
/// creating an automatic table of contents for a song.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 128, outputSize: 8);
/// var model = new MusicStructureAnalyzer&lt;float&gt;(arch, "music_structure.onnx");
/// var segments = model.AnalyzeStructure(audioTensor);
/// foreach (var seg in segments)
///     Console.WriteLine($"{seg.Label}: {seg.StartTime:F1}s - {seg.EndTime:F1}s");
/// </code>
/// </para>
/// </remarks>
public class MusicStructureAnalyzer<T> : AudioNeuralNetworkBase<T>
{
    #region Fields

    private readonly MusicStructureAnalyzerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Music Structure Analyzer in ONNX inference mode.
    /// </summary>
    public MusicStructureAnalyzer(NeuralNetworkArchitecture<T> architecture, string modelPath, MusicStructureAnalyzerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MusicStructureAnalyzerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Music Structure Analyzer in native training mode.
    /// </summary>
    public MusicStructureAnalyzer(NeuralNetworkArchitecture<T> architecture, MusicStructureAnalyzerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MusicStructureAnalyzerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MusicStructureAnalyzer<T>> CreateAsync(MusicStructureAnalyzerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MusicStructureAnalyzerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("music_structure", "music_structure_analyzer.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumSections);
        return new MusicStructureAnalyzer<T>(arch, mp, options);
    }

    #endregion

    #region Public API

    /// <summary>
    /// Analyzes the structure of a song, returning labeled time segments.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>List of structural segments with labels and timings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in an entire song (as a waveform) and get back a list of
    /// sections like "intro: 0.0-12.5s", "verse: 12.5-45.0s", "chorus: 45.0-70.0s", etc.
    /// </para>
    /// </remarks>
    public IReadOnlyList<MusicSegment<T>> AnalyzeStructure(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return DecodeSegments(output, audio.Length);
    }

    /// <summary>
    /// Analyzes music structure asynchronously.
    /// </summary>
    public Task<IReadOnlyList<MusicSegment<T>>> AnalyzeStructureAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => AnalyzeStructure(audio), cancellationToken);
    }

    /// <summary>
    /// Gets per-frame section probabilities.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of section probabilities per frame [numFrames * numSections].</returns>
    public Tensor<T> GetSectionProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // Apply softmax over section dimension
        var probs = new Tensor<T>([output.Length]);
        int numSections = _options.NumSections;
        int numFrames = output.Length / numSections;
        for (int f = 0; f < numFrames; f++)
        {
            double maxVal = double.NegativeInfinity;
            for (int s = 0; s < numSections && (f * numSections + s) < output.Length; s++)
            {
                double v = NumOps.ToDouble(output[f * numSections + s]);
                if (v > maxVal) maxVal = v;
            }
            double sumExp = 0;
            for (int s = 0; s < numSections && (f * numSections + s) < output.Length; s++)
                sumExp += Math.Exp(NumOps.ToDouble(output[f * numSections + s]) - maxVal);
            for (int s = 0; s < numSections && (f * numSections + s) < output.Length; s++)
            {
                double v = Math.Exp(NumOps.ToDouble(output[f * numSections + s]) - maxVal) / sumExp;
                probs[f * numSections + s] = NumOps.FromDouble(v);
            }
        }
        return probs;
    }

    /// <summary>
    /// Gets the section labels this model can detect.
    /// </summary>
    public string[] SectionLabels => _options.SectionLabels;

    /// <summary>
    /// Gets the number of structural sections.
    /// </summary>
    public int NumSections => _options.NumSections;

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMusicStructureAnalyzerLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            numSections: _options.NumSections, dropoutRate: _options.DropoutRate));
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
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MusicStructureAnalyzer-Native" : "MusicStructureAnalyzer-ONNX",
            Description = "Neural music structure segmentation (intro/verse/chorus/bridge/outro)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumSections"] = _options.NumSections.ToString();
        m.AdditionalInfo["SectionLabels"] = string.Join(",", _options.SectionLabels);
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.NumSections);
        w.Write(_options.SectionLabels.Length);
        foreach (var label in _options.SectionLabels) w.Write(label);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.NumSections = r.ReadInt32();
        int labelCount = r.ReadInt32();
        var labels = new string[labelCount];
        for (int i = 0; i < labelCount; i++) labels[i] = r.ReadString();
        _options.SectionLabels = labels;
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MusicStructureAnalyzer<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private IReadOnlyList<MusicSegment<T>> DecodeSegments(Tensor<T> output, int audioLength)
    {
        var segments = new List<MusicSegment<T>>();
        int numSections = _options.NumSections;
        int numFrames = Math.Max(1, output.Length / numSections);
        double frameTimeSec = (double)_options.HopLength / _options.SampleRate;
        double duration = (double)audioLength / _options.SampleRate;

        int currentLabel = -1;
        double segmentStart = 0;

        for (int f = 0; f < numFrames; f++)
        {
            // Find best section for this frame
            int bestSection = 0;
            double bestVal = double.NegativeInfinity;
            for (int s = 0; s < numSections && (f * numSections + s) < output.Length; s++)
            {
                double v = NumOps.ToDouble(output[f * numSections + s]);
                if (v > bestVal) { bestVal = v; bestSection = s; }
            }

            if (bestSection != currentLabel)
            {
                if (currentLabel >= 0)
                {
                    double endTime = f * frameTimeSec;
                    string label = currentLabel < _options.SectionLabels.Length ? _options.SectionLabels[currentLabel] : "other";
                    segments.Add(new MusicSegment<T>
                    {
                        Label = label,
                        SectionIndex = currentLabel,
                        StartTime = segmentStart,
                        EndTime = endTime,
                        Confidence = NumOps.FromDouble(0.8)
                    });
                }
                currentLabel = bestSection;
                segmentStart = f * frameTimeSec;
            }
        }

        // Add final segment
        if (currentLabel >= 0)
        {
            string label = currentLabel < _options.SectionLabels.Length ? _options.SectionLabels[currentLabel] : "other";
            segments.Add(new MusicSegment<T>
            {
                Label = label,
                SectionIndex = currentLabel,
                StartTime = segmentStart,
                EndTime = duration,
                Confidence = NumOps.FromDouble(0.8)
            });
        }

        return segments;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicStructureAnalyzer<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

/// <summary>
/// Represents a labeled segment of music structure.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class MusicSegment<T>
{
    /// <summary>Gets or sets the section label (e.g., "verse", "chorus").</summary>
    public required string Label { get; init; }

    /// <summary>Gets or sets the section index.</summary>
    public required int SectionIndex { get; init; }

    /// <summary>Gets or sets the start time in seconds.</summary>
    public required double StartTime { get; init; }

    /// <summary>Gets or sets the end time in seconds.</summary>
    public required double EndTime { get; init; }

    /// <summary>Gets or sets the confidence score.</summary>
    public required T Confidence { get; init; }

    /// <summary>Gets the duration in seconds.</summary>
    public double Duration => EndTime - StartTime;
}
