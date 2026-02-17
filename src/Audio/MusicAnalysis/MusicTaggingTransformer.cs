using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Music Tagging Transformer for multi-label music tag prediction (genre, mood, instrument, era).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Music Tagging Transformer (Won et al., 2021) uses a Transformer encoder on mel spectrogram
/// features to predict music tags (genre, mood, instrument, era). It achieves state-of-the-art
/// results on the MagnaTagATune and Million Song Dataset benchmarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to music and automatically tags it with descriptive
/// labels - like "rock", "upbeat", "guitar", "1980s", or "relaxing". It's the technology behind
/// automatic music categorization in streaming services like Spotify's genre detection.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 128, outputSize: 50);
/// var model = new MusicTaggingTransformer&lt;float&gt;(arch, "music_tagging.onnx");
/// var tags = model.PredictTags(audioTensor);
/// foreach (var (tag, confidence) in tags)
///     Console.WriteLine($"{tag}: {confidence:P0}");
/// </code>
/// </para>
/// </remarks>
public class MusicTaggingTransformer<T> : AudioNeuralNetworkBase<T>
{
    #region Fields

    private readonly MusicTaggingTransformerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Music Tagging Transformer in ONNX inference mode.
    /// </summary>
    public MusicTaggingTransformer(NeuralNetworkArchitecture<T> architecture, string modelPath, MusicTaggingTransformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MusicTaggingTransformerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a Music Tagging Transformer in native training mode.
    /// </summary>
    public MusicTaggingTransformer(NeuralNetworkArchitecture<T> architecture, MusicTaggingTransformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MusicTaggingTransformerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MusicTaggingTransformer<T>> CreateAsync(MusicTaggingTransformerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MusicTaggingTransformerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("music_tagging_transformer", "music_tagging_transformer.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.NumTags);
        return new MusicTaggingTransformer<T>(arch, mp, options);
    }

    #endregion

    #region Public API

    /// <summary>
    /// Predicts music tags for the given audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="threshold">Confidence threshold for including a tag (0.0-1.0).</param>
    /// <returns>List of (tag label, confidence) pairs above the threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in audio and get back a list of tags with confidence scores.
    /// For example: [("rock", 0.92), ("guitar", 0.87), ("energetic", 0.75)].
    /// Higher confidence means the model is more sure about that tag.
    /// </para>
    /// </remarks>
    public IReadOnlyList<(string Tag, double Confidence)> PredictTags(Tensor<T> audio, double threshold = 0.5)
    {
        ThrowIfDisposed();
        var probs = GetTagProbabilities(audio);
        var tags = new List<(string Tag, double Confidence)>();

        int numTags = Math.Min(_options.NumTags, Math.Min(probs.Length, _options.TagLabels.Length));
        for (int i = 0; i < numTags; i++)
        {
            double p = NumOps.ToDouble(probs[i]);
            if (p >= threshold)
                tags.Add((_options.TagLabels[i], p));
        }

        tags.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return tags;
    }

    /// <summary>
    /// Predicts music tags asynchronously.
    /// </summary>
    public Task<IReadOnlyList<(string Tag, double Confidence)>> PredictTagsAsync(Tensor<T> audio, double threshold = 0.5, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => PredictTags(audio, threshold), cancellationToken);
    }

    /// <summary>
    /// Gets raw tag probabilities for all tags.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of tag probabilities [numTags].</returns>
    public Tensor<T> GetTagProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // Apply sigmoid (multi-label, not softmax)
        int numTags = Math.Min(_options.NumTags, output.Length);
        var probs = new Tensor<T>([numTags]);
        for (int i = 0; i < numTags; i++)
        {
            double v = NumOps.ToDouble(output[i]);
            probs[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-v)));
        }
        return probs;
    }

    /// <summary>
    /// Gets the top-K tags by confidence.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="k">Number of top tags to return.</param>
    /// <returns>List of (tag label, confidence) pairs, sorted by confidence descending.</returns>
    public IReadOnlyList<(string Tag, double Confidence)> GetTopKTags(Tensor<T> audio, int k = 10)
    {
        ThrowIfDisposed();
        var probs = GetTagProbabilities(audio);
        var allTags = new List<(string Tag, double Confidence)>();

        int numTags = Math.Min(_options.NumTags, Math.Min(probs.Length, _options.TagLabels.Length));
        for (int i = 0; i < numTags; i++)
            allTags.Add((_options.TagLabels[i], NumOps.ToDouble(probs[i])));

        allTags.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return allTags.GetRange(0, Math.Min(k, allTags.Count));
    }

    /// <summary>
    /// Gets the tag labels this model can predict.
    /// </summary>
    public string[] TagLabels => _options.TagLabels;

    /// <summary>
    /// Gets the number of tags.
    /// </summary>
    public int NumTags => _options.NumTags;

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMusicTaggingTransformerLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, numTags: _options.NumTags,
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
            Name = _useNativeMode ? "MusicTaggingTransformer-Native" : "MusicTaggingTransformer-ONNX",
            Description = "Music Tagging Transformer (Won et al., 2021)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["NumTags"] = _options.NumTags.ToString();
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize);
        w.Write(_options.HopLength); w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardDim);
        w.Write(_options.NumTags);
        w.Write(_options.TagLabels.Length);
        foreach (var label in _options.TagLabels) w.Write(label);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardDim = r.ReadInt32();
        _options.NumTags = r.ReadInt32();
        int labelCount = r.ReadInt32();
        var labels = new string[labelCount];
        for (int i = 0; i < labelCount; i++) labels[i] = r.ReadString();
        _options.TagLabels = labels;
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MusicTaggingTransformer<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicTaggingTransformer<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
