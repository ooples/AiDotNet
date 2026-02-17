using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// WavLM self-supervised speech representation model from Microsoft.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WavLM (Chen et al., 2022, Microsoft) extends HuBERT with gated relative position bias
/// and denoising pre-training objectives. It achieves state-of-the-art on the SUPERB benchmark
/// and is particularly strong for speaker-related tasks due to its denoising training.
/// </para>
/// <para>
/// <b>For Beginners:</b> WavLM improves on HuBERT by also learning to handle noisy audio.
/// During training, it was given audio with added noise and had to understand it anyway. This
/// makes it especially good at tasks in real-world noisy conditions, like telling speakers
/// apart in a meeting or understanding speech in a crowded room.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 768);
/// var model = new WavLM&lt;float&gt;(arch, "wavlm_base.onnx");
/// var embeddings = model.ExtractEmbeddings(audioWaveform);
/// </code>
/// </para>
/// </remarks>
public class WavLM<T> : AudioNeuralNetworkBase<T>, IAudioFoundationModel<T>
{
    #region Fields

    private readonly WavLMOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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

    public WavLM(NeuralNetworkArchitecture<T> architecture, string modelPath, WavLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WavLMOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public WavLM(NeuralNetworkArchitecture<T> architecture, WavLMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new WavLMOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<WavLM<T>> CreateAsync(WavLMOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new WavLMOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("wavlm", $"wavlm_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.HiddenDim);
        return new WavLM<T>(arch, mp, options);
    }

    #endregion

    #region IAudioFoundationModel

    /// <inheritdoc />
    public Tensor<T> ExtractEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is null)
            throw new InvalidOperationException("ONNX encoder is not loaded. Provide a valid model path.");
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => ExtractEmbeddings(audio), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return ExtractEmbeddings(audio);
        int targetLayer = layerIndex < 0 ? _options.NumLayers + layerIndex : layerIndex;
        if (targetLayer < 0 || targetLayer >= _options.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), $"Layer index must be between -{_options.NumLayers} and {_options.NumLayers - 1}.");
        var c = audio; int currentLayer = 0;
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
        foreach (var l in Layers) { c = l.Forward(c); if (l is MultiHeadAttentionLayer<T>) layerOutputs.Add(c); }
        if (layerOutputs.Count == 0) return c;
        var result = new Tensor<T>(layerOutputs[0].Shape);
        int count = layerOutputs.Count;
        for (int li = 0; li < count; li++)
        {
            double w = layerWeights is not null && li < layerWeights.Length ? NumOps.ToDouble(layerWeights[li]) : 1.0 / count;
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultFoundationModelLayers(
            featureEncoderDim: _options.FeatureEncoderDim, hiddenDim: _options.HiddenDim,
            numLayers: _options.NumLayers, numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "WavLM-Native" : "WavLM-ONNX",
            Description = $"WavLM {_options.Variant} self-supervised speech model (Chen et al., 2022)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1, Complexity = _options.NumLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardDim);
        w.Write(_options.FeatureEncoderDim); w.Write(_options.UseGatedRelativePositionBias);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardDim = r.ReadInt32();
        _options.FeatureEncoderDim = r.ReadInt32(); _options.UseGatedRelativePositionBias = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        base.SampleRate = _options.SampleRate;
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new WavLM<T>(Architecture, mp, _options);
        return new WavLM<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WavLM<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
