using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// data2vec 2.0 self-supervised audio representation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// data2vec 2.0 (Baevski et al., 2023, Meta) is a self-supervised framework that predicts
/// contextualized latent representations rather than modality-specific targets. Version 2.0 is
/// 16x faster than v1 through efficient data encoding and self-distillation. It achieves strong
/// results on speech, vision, and language tasks with the same method.
/// </para>
/// <para>
/// <b>For Beginners:</b> data2vec 2.0 learns audio features by predicting its own hidden
/// representations - like learning by explaining things to yourself. Unlike HuBERT which
/// predicts discrete labels, data2vec predicts rich continuous features. It can be used as a
/// powerful feature extractor for any downstream audio task.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 768);
/// var model = new Data2Vec2&lt;float&gt;(arch, "data2vec2_base.onnx");
/// var embeddings = model.ExtractEmbeddings(audioWaveform);
/// </code>
/// </para>
/// </remarks>
public class Data2Vec2<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>
{
    #region Fields

    private readonly Data2Vec2Options _options;
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

    /// <summary>Creates a data2vec 2.0 model in ONNX inference mode.</summary>
    public Data2Vec2(NeuralNetworkArchitecture<T> architecture, string modelPath, Data2Vec2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new Data2Vec2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a data2vec 2.0 model in native training mode.</summary>
    public Data2Vec2(NeuralNetworkArchitecture<T> architecture, Data2Vec2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Data2Vec2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Data2Vec2<T>> CreateAsync(Data2Vec2Options? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new Data2Vec2Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("data2vec2", $"data2vec2_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.HiddenDim);
        return new Data2Vec2<T>(arch, mp, options);
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
        if (IsOnnxMode) return ExtractEmbeddings(audio);

        var layerOutputs = new List<Tensor<T>>();
        var c = audio;
        foreach (var l in Layers)
        {
            c = l.Forward(c);
            if (l is MultiHeadAttentionLayer<T>) layerOutputs.Add(c);
        }

        if (layerOutputs.Count == 0) return c;

        var result = new Tensor<T>(layerOutputs[0].Shape);
        int count = layerOutputs.Count;
        for (int li = 0; li < count; li++)
        {
            double w = layerWeights is not null && li < layerWeights.Length
                ? NumOps.ToDouble(layerWeights[li]) : 1.0 / count;
            for (int i = 0; i < result.Length && i < layerOutputs[li].Length; i++)
                result[i] = NumOps.Add(result[i], NumOps.FromDouble(NumOps.ToDouble(layerOutputs[li][i]) * w));
        }
        return result;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultData2Vec2Layers(
            hiddenDim: _options.HiddenDim, numLayers: _options.NumLayers,
            numHeads: _options.NumHeads, feedForwardDim: _options.FeedForwardDim,
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
        _optimizer?.UpdateParameters(Layers);
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
            Name = _useNativeMode ? "data2vec2-Native" : "data2vec2-ONNX",
            Description = $"data2vec 2.0 {_options.Variant} self-supervised audio model (Baevski et al., 2023, Meta)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim,
            Complexity = _options.NumLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["EMADecay"] = _options.EMADecay.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.HiddenDim);
        w.Write(_options.NumLayers); w.Write(_options.NumHeads);
        w.Write(_options.FeedForwardDim); w.Write(_options.Variant);
        w.Write(_options.EMADecay); w.Write(_options.MaskProbability);
        w.Write(_options.MaskSpanLength); w.Write(_options.TopKLayers);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumLayers = r.ReadInt32(); _options.NumHeads = r.ReadInt32();
        _options.FeedForwardDim = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.EMADecay = r.ReadDouble(); _options.MaskProbability = r.ReadDouble();
        _options.MaskSpanLength = r.ReadInt32(); _options.TopKLayers = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Data2Vec2<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Data2Vec2<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
