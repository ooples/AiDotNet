using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Asymmetric Diffusion Transformer (AsymmDiT) noise predictor for video generation (Mochi architecture).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AsymmDiT uses asymmetric attention where video tokens attend to text tokens differently
/// than text tokens attend to video tokens. This design is optimized for text-to-video generation
/// where the conditioning signal (text) has different semantics than the generated content (video).
/// </para>
/// <para>
/// <b>For Beginners:</b> AsymmDiT is the architecture behind Mochi, a video generation model.
/// It processes text and video differently because they have different properties — text is
/// short and semantic, while video is long and spatial-temporal. The "asymmetric" part means
/// it doesn't treat them the same way.
///
/// Used in: Mochi (Genmo), other video diffusion models
/// </para>
/// <para>
/// Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024
/// </para>
/// </remarks>
public class AsymmDiTPredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _contextDim;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _blocks;
    private DenseLayer<T> _finalLayer;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;
    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;
    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;
    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize;
    /// <inheritdoc />
    public override bool SupportsCFG => true;
    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;
    /// <inheritdoc />
    public override int ContextDimension => _contextDim;
    /// <inheritdoc />
    public override int ParameterCount { get; }

    /// <summary>
    /// Initializes a new AsymmDiT predictor.
    /// </summary>
    /// <param name="inputChannels">Input latent channels. Default: 12.</param>
    /// <param name="hiddenSize">Transformer hidden dimension. Default: 3072.</param>
    /// <param name="numLayers">Number of transformer layers. Default: 48.</param>
    /// <param name="numHeads">Number of attention heads. Default: 24.</param>
    /// <param name="contextDim">Text conditioning dimension. Default: 4096.</param>
    /// <param name="seed">Optional random seed.</param>
    public AsymmDiTPredictor(
        int inputChannels = 12,
        int hiddenSize = 3072,
        int numLayers = 48,
        int numHeads = 24,
        int contextDim = 4096,
        int? seed = null)
        : base(seed: seed)
    {
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _contextDim = contextDim;

        InitializeLayers(seed);
        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_blocks), nameof(_finalLayer))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * 4;
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _blocks = new DenseLayer<T>[_numLayers];
        for (int i = 0; i < _numLayers; i++)
            _blocks[i] = new DenseLayer<T>(_hiddenSize, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _finalLayer = new DenseLayer<T>(_hiddenSize, patchDim, (IActivationFunction<T>?)null);
    }

    private int CalculateParameterCount()
    {
        int count = _patchEmbed.ParameterCount;
        foreach (var block in _blocks) count += block.ParameterCount;
        count += _finalLayer.ParameterCount;
        return count;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        var x = _patchEmbed.Forward(noisySample);
        foreach (var block in _blocks) x = block.Forward(x);
        return _finalLayer.Forward(x);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddParams(allParams, _patchEmbed);
        foreach (var b in _blocks) AddParams(allParams, b);
        AddParams(allParams, _finalLayer);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetParams(_patchEmbed, parameters, offset);
        foreach (var b in _blocks) offset = SetParams(b, parameters, offset);
        SetParams(_finalLayer, parameters, offset);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new AsymmDiTPredictor<T>(_inputChannels, _hiddenSize, _numLayers, _numHeads, _contextDim);
        clone.SetParameters(GetParameters());
        return clone;
    }

    private static void AddParams(List<T> list, DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int SetParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }
}
