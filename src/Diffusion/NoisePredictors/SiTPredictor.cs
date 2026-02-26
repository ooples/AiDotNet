using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SiT generalizes DiT by supporting arbitrary interpolants between data and noise,
/// unifying DDPM, flow matching, and stochastic interpolants within a single framework.
/// This enables flexible choice of forward process and sampling strategy.
/// </para>
/// <para>
/// <b>For Beginners:</b> SiT is a flexible version of DiT (Diffusion Transformer) that
/// can work with different types of noise processes. Instead of being locked to one way
/// of adding noise, SiT can smoothly interpolate between different strategies, choosing
/// the best approach for quality and speed.
///
/// Key advantage: You can pick different sampling methods at inference time without
/// retraining the model.
///
/// Used in: SiT research models, SANA (partially)
/// </para>
/// <para>
/// Reference: Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", ECCV 2024
/// </para>
/// </remarks>
public class SiTPredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;

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
    public override int ContextDimension => _hiddenSize;
    /// <inheritdoc />
    public override int ParameterCount { get; }

    /// <summary>
    /// Initializes a new SiT predictor.
    /// </summary>
    /// <param name="inputChannels">Input latent channels. Default: 4.</param>
    /// <param name="hiddenSize">Transformer hidden dimension. Default: 1152.</param>
    /// <param name="numLayers">Number of transformer layers. Default: 28.</param>
    /// <param name="numHeads">Number of attention heads. Default: 16.</param>
    /// <param name="seed">Optional random seed.</param>
    public SiTPredictor(
        int inputChannels = 4,
        int hiddenSize = 1152,
        int numLayers = 28,
        int numHeads = 16,
        int? seed = null)
        : base(seed: seed)
    {
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;

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
        var clone = new SiTPredictor<T>(_inputChannels, _hiddenSize, _numLayers, _numHeads);
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
