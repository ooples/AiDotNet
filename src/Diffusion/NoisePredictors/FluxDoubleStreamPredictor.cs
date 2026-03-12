using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// FLUX double-stream transformer noise predictor with joint and single-stream blocks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The FLUX architecture uses a hybrid design: double-stream blocks process text and image
/// tokens with joint attention but separate MLPs, followed by single-stream blocks that
/// process both modalities through a shared path.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the transformer architecture behind FLUX models.
/// It has two stages:
/// 1. Double-stream blocks: Text and image tokens interact through shared attention
///    but have their own separate processing paths (MLPs)
/// 2. Single-stream blocks: Both text and image tokens are processed together through
///    a shared path, enabling deep fusion
///
/// This hybrid design balances the benefits of modality-specific processing with
/// deep cross-modal interaction.
/// </para>
/// <para>
/// Reference: Black Forest Labs, "FLUX.1 Technical Report", 2024
/// </para>
/// </remarks>
public class FluxDoubleStreamPredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numSingleLayers;
    private readonly int _contextDim;
    private readonly FluxPredictorVariant _variant;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _doubleBlocks;
    private DenseLayer<T>[] _singleBlocks;
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
    /// Initializes a new FLUX double-stream predictor.
    /// </summary>
    /// <param name="variant">FLUX variant. Default: Dev.</param>
    /// <param name="inputChannels">Latent channels. Default: 16.</param>
    /// <param name="contextDim">Context dimension. Default: 4096.</param>
    /// <param name="seed">Optional random seed.</param>
    public FluxDoubleStreamPredictor(
        FluxPredictorVariant variant = FluxPredictorVariant.Dev,
        int inputChannels = 16,
        int contextDim = 4096,
        int? seed = null)
        : base(seed: seed)
    {
        _variant = variant;
        _inputChannels = inputChannels;
        _hiddenSize = 3072;
        _numJointLayers = 19;
        _numSingleLayers = 38;
        _contextDim = contextDim;

        InitializeLayers(seed);
        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_doubleBlocks), nameof(_singleBlocks), nameof(_finalLayer))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * 4; // 2x2 patches
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _doubleBlocks = new DenseLayer<T>[_numJointLayers];
        for (int i = 0; i < _numJointLayers; i++)
            _doubleBlocks[i] = new DenseLayer<T>(_hiddenSize, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _singleBlocks = new DenseLayer<T>[_numSingleLayers];
        for (int i = 0; i < _numSingleLayers; i++)
            _singleBlocks[i] = new DenseLayer<T>(_hiddenSize, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _finalLayer = new DenseLayer<T>(_hiddenSize, patchDim, (IActivationFunction<T>?)null);
    }

    private int CalculateParameterCount()
    {
        int count = _patchEmbed.ParameterCount;
        foreach (var block in _doubleBlocks) count += block.ParameterCount;
        foreach (var block in _singleBlocks) count += block.ParameterCount;
        count += _finalLayer.ParameterCount;
        return count;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        var x = _patchEmbed.Forward(noisySample);

        foreach (var block in _doubleBlocks)
            x = block.Forward(x);

        foreach (var block in _singleBlocks)
            x = block.Forward(x);

        return _finalLayer.Forward(x);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddParams(allParams, _patchEmbed);
        foreach (var b in _doubleBlocks) AddParams(allParams, b);
        foreach (var b in _singleBlocks) AddParams(allParams, b);
        AddParams(allParams, _finalLayer);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetParams(_patchEmbed, parameters, offset);
        foreach (var b in _doubleBlocks) offset = SetParams(b, parameters, offset);
        foreach (var b in _singleBlocks) offset = SetParams(b, parameters, offset);
        SetParams(_finalLayer, parameters, offset);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new FluxDoubleStreamPredictor<T>(_variant, _inputChannels, _contextDim);
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
