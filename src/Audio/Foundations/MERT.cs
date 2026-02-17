using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// MERT self-supervised music understanding foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MERT (Li et al., 2024) uses acoustic and musical tokenizers to learn rich music
/// representations. Unlike speech models, it incorporates music-specific knowledge through
/// CQT-based teacher targets and codebook clustering, enabling strong performance on 14
/// music information retrieval tasks including tagging, genre, instrument, and key detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> MERT is like HuBERT but for music. It deeply understands musical
/// structure - pitch, harmony, rhythm, instrumentation - without needing labeled data. Use it
/// as a feature extractor: feed it music and get embeddings useful for genre classification,
/// instrument detection, mood analysis, and more.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 768);
/// var model = new MERT&lt;float&gt;(arch, "mert_base.onnx");
/// var embeddings = model.ExtractEmbeddings(musicWaveform);
/// </code>
/// </para>
/// </remarks>
public class MERT<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>
{
    #region Fields

    private readonly MERTOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioFoundationModel Properties

    /// <inheritdoc />
    public int EmbeddingDimension => _options.HiddenDim;

    /// <inheritdoc />
    public int NumLayers => _options.NumLayers;

    #endregion

    #region Constructors

    /// <summary>Creates a MERT model in ONNX inference mode.</summary>
    public MERT(NeuralNetworkArchitecture<T> architecture, string modelPath, MERTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MERTOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a MERT model in native training mode.</summary>
    public MERT(NeuralNetworkArchitecture<T> architecture, MERTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MERTOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MERT<T>> CreateAsync(MERTOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MERTOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("mert", $"mert_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.HiddenDim);
        return new MERT<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFoundationModel

    /// <inheritdoc />
    public Tensor<T> ExtractEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ExtractEmbeddings(audio), cancellationToken);
    }

    /// <inheritdoc />
    public Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);

        int targetLayer = layerIndex < 0 ? _options.NumLayers + layerIndex : layerIndex;
        if (targetLayer < 0 || targetLayer >= _options.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index {layerIndex} is out of range. Valid range: [{-_options.NumLayers}, {_options.NumLayers - 1}].");
        var c = audio;
        int currentLayer = 0;
        foreach (var l in Layers)
        {
            c = l.Forward(c);
            if (l is MultiHeadAttentionLayer<T>) { if (currentLayer == targetLayer) return c; currentLayer++; }
        }
        return c;
    }

    /// <inheritdoc />
    public Tensor<T> ExtractWeightedFeatures(Tensor<T> audio, T[]? layerWeights = null)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
        {
            if (layerWeights is not null)
                throw new NotSupportedException("Layer weights are not supported in ONNX mode. Use native mode for weighted feature extraction.");
            return ExtractEmbeddings(audio);
        }

        var layerOutputs = new List<Tensor<T>>();
        var c = audio;
        foreach (var l in Layers)
        {
            c = l.Forward(c);
            if (l is MultiHeadAttentionLayer<T>) layerOutputs.Add(c);
        }

        if (layerOutputs.Count == 0) return c;

        int count = layerOutputs.Count;
        var weights = new double[count];

        if (layerWeights is null)
        {
            for (int li = 0; li < count; li++)
                weights[li] = 1.0 / count;
        }
        else
        {
            if (layerWeights.Length != count)
                throw new ArgumentException(
                    $"layerWeights length ({layerWeights.Length}) must match the number of transformer layers ({count}).",
                    nameof(layerWeights));

            double sum = 0;
            for (int li = 0; li < count; li++)
            {
                weights[li] = NumOps.ToDouble(layerWeights[li]);
                sum += weights[li];
            }

            if (sum > 0)
            {
                for (int li = 0; li < count; li++)
                    weights[li] /= sum;
            }
        }

        var result = new Tensor<T>(layerOutputs[0].Shape);
        for (int li = 0; li < count; li++)
        {
            for (int i = 0; i < result.Length && i < layerOutputs[li].Length; i++)
                result[i] = NumOps.Add(result[i], NumOps.FromDouble(NumOps.ToDouble(layerOutputs[li][i]) * weights[li]));
        }
        return result;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMERTLayers(
            hiddenDim: _options.HiddenDim, numLayers: _options.NumLayers,
            numHeads: _options.NumHeads, feedForwardDim: _options.FeedForwardDim,
            dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized. Cannot train without an optimizer.");
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
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <summary>Returns raw audio unchanged; MERT expects raw waveform input and handles internal feature extraction.</summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <summary>Returns output unchanged; no post-processing needed for foundation model embeddings.</summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MERT-Native" : "MERT-ONNX",
            Description = $"MERT {_options.Variant} music understanding foundation model (Li et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["CQTBins"] = _options.CQTBins.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.HiddenDim);
        w.Write(_options.NumLayers); w.Write(_options.NumHeads);
        w.Write(_options.FeedForwardDim); w.Write(_options.Variant);
        w.Write(_options.CQTBins); w.Write(_options.NumCodebooks);
        w.Write(_options.CodebookSize); w.Write(_options.NumClusters);
        w.Write(_options.MaskProbability); w.Write(_options.MaskSpanLength);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumLayers = r.ReadInt32(); _options.NumHeads = r.ReadInt32();
        _options.FeedForwardDim = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.CQTBins = r.ReadInt32(); _options.NumCodebooks = r.ReadInt32();
        _options.CodebookSize = r.ReadInt32(); _options.NumClusters = r.ReadInt32();
        _options.MaskProbability = r.ReadDouble(); _options.MaskSpanLength = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new MERT<T>(Architecture, mp, _options);
        return new MERT<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MERT<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
