using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Music Structure Analysis: A Survey", "https://doi.org/10.1007/978-3-319-25226-1_12")]
public partial class MusicStructureAnalyzer<T> : AudioNeuralNetworkBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore folds over Layers, and the last layer
    /// CreateDefaultMusicStructureAnalyzerLayers emits is the section-segmentation head
    /// <c>DenseLayer&lt;T&gt;(numSections)</c>, wired from <c>_options.NumSections</c> (8). A
    /// structural class count; the HiddenDim projection just before it is not the output width.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.NumSections;

    #region Fields

    private readonly MusicStructureAnalyzerOptions _options;
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
        Options = _options;
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
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture)
    {
        _options = options ?? new MusicStructureAnalyzerOptions();
        Options = _options;
        // Music-structure segmentation is a per-frame MULTI-CLASS classifier over the NumSections section
        // labels (intro/verse/chorus/bridge/outro). The NeuralNetworkBase default MSE loss is unstable on raw
        // classification logits, so LossStrictlyDecreases / Training_ShouldReduceLoss do not hold. The
        // paper-faithful DEFAULT is softmax cross-entropy (bounded gradient) so training converges — but the
        // caller can override it via the lossFunction parameter for full user customization (never hardcoded).
        // Mirrors the HuBERTSER audio-classifier fix; AudioGenModel is the customizable-loss template. (#1789)
        LossFunction = lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>();
        _useNativeMode = true;
        // Default optimizer honors the model's configured LearningRate (previously the bare AdamWOptimizer(this)
        // ignored it and ran at Adam's 0.001) and enables gradient clipping (MaxGradientNorm 1.0) so the
        // cross-entropy gradients can't overshoot the first step into a high-loss region. Fully user-overridable
        // via the optimizer parameter and MusicStructureAnalyzerOptions.LearningRate. Mirrors HuBERTSER. (#1789)
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate, EnableGradientClipping = true, MaxGradientNorm = 1.0 });
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    /// <summary>
    /// Routes tape training through the configured optimizer. Without this override the base trainer only
    /// consults <see cref="NeuralNetworkBase{T}.GetOrCreateBaseOptimizer"/>, so the <c>_optimizer</c> field above
    /// was stored but never used and training silently ran at the base Adam 1e-3 default instead of the
    /// model's 1e-4 — Training_ShouldReduceLoss saw loss rise 8.86 -> 10.95 on the CI envelope. Same defect
    /// already fixed in litedvdnet, the madmom beat tracker and mog. (#1789)
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

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

        // Apply softmax over section dimension (the max skips NaN); a trailing partial frame stays 0
        int numSections = _options.NumSections;
        int numFrames = output.Length / numSections;
        if (numFrames <= 0)
            return new Tensor<T>([output.Length]);

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int used = numFrames * numSections;
        var shape = new[] { numFrames, numSections };
        var logits = Engine.Reshape(Engine.TensorSlice(AudioEngineOps.Flatten(Engine, output), [0], [used]), shape);
        var ranked = Engine.TensorWhere(
            Engine.TensorIsNan(logits), Tensor<T>.CreateDefault(shape, NumOps.FromDouble(double.NegativeInfinity)), logits);
        var maxVal = Engine.TensorBroadcastTo(Engine.ReduceMax(ranked, [1], keepDims: true), shape);
        var exp = Engine.TensorExp(Engine.TensorSubtract(logits, maxVal));
        var sumExp = Engine.TensorBroadcastTo(Engine.ReduceSum(exp, [1], keepDims: true), shape);
        var probs = Engine.Reshape(Engine.TensorDivide(exp, sumExp), [used]);
        return AudioEngineOps.FitLength(Engine, probs, output.Length);
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

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MusicStructureAnalyzer-Native" : "MusicStructureAnalyzer-ONNX",
            Description = "Neural music structure segmentation (intro/verse/chorus/bridge/outro)",
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumSections"] = _options.NumSections.ToString();
        m.AdditionalInfo["SectionLabels"] = string.Join(",", _options.SectionLabels);
        return m;
    }





    #endregion

    #region Private Helpers

    private IReadOnlyList<MusicSegment<T>> DecodeSegments(Tensor<T> output, int audioLength)
    {
        var segments = new List<MusicSegment<T>>();
        int numSections = _options.NumSections;
        int numFrames = Math.Max(1, output.Length / numSections);
        double frameTimeSec = (double)_options.HopLength / _options.SampleRate;
        double duration = (double)audioLength / _options.SampleRate;
        T confidence = NumOps.FromDouble(0.8);

        // Best section for each frame
        int[] bestSections = FrameBestSections(output, numSections, numFrames);

        int currentLabel = -1;
        double segmentStart = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int bestSection = bestSections[f];

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
                        Confidence = confidence
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
                Confidence = confidence
            });
        }

        return segments;
    }

    /// <summary>
    /// For each frame, the first section holding the frame's largest score (scores
    /// <c>output[f * numSections + s]</c> that exist; NaN never wins; section 0 when none beats -inf).
    /// </summary>
    private int[] FrameBestSections(Tensor<T> output, int numSections, int numFrames)
    {
        if (numSections <= 0) return new int[numFrames];
        int total = numFrames * numSections;
        int available = Math.Min(output.Length, total);
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var scores = available == 0
            ? Tensor<T>.CreateDefault([total], negativeInfinity)
            : Engine.TensorSlice(AudioEngineOps.Flatten(Engine, output), [0], [available]);
        if (available > 0 && available < total)
            scores = Engine.Concat(new[] { scores, Tensor<T>.CreateDefault([total - available], negativeInfinity) }, 0);
        return AudioEngineOps.RowFirstArgMax(Engine, Engine.Reshape(scores, [numFrames, numSections]));
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicStructureAnalyzer<T>)); }

    /// <inheritdoc />
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
