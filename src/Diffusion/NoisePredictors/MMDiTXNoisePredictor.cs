using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Extended Multi-Modal Diffusion Transformer (MMDiT-X) noise predictor for SD3.5 architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMDiT-X extends the original MMDiT architecture with improved QK-normalization,
/// enhanced position encoding, and optimized attention patterns for SD3.5 models.
/// </para>
/// <para>
/// <b>For Beginners:</b> MMDiT-X is the improved brain of Stable Diffusion 3.5.
/// It processes both text and image information together through joint attention blocks,
/// allowing the model to deeply understand how text descriptions relate to image features.
///
/// Key improvements over MMDiT:
/// - QK-normalization prevents attention collapse at higher resolutions
/// - More efficient attention patterns reduce memory usage
/// - Better positional encoding for multi-resolution support
/// - Available in Medium (2B) and Large (8B) configurations
/// </para>
/// <para>
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
public class MMDiTXNoisePredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _contextDim;
    private readonly MMDiTXVariant _variant;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T>[] _jointBlocks;
    private DenseLayer<T> _finalLayer;
    private Vector<T> _posEmbed;

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
    /// Initializes a new MMDiT-X noise predictor for SD3.5.
    /// </summary>
    /// <param name="variant">SD3.5 variant. Default: Medium.</param>
    /// <param name="inputChannels">Latent channels. Default: 16.</param>
    /// <param name="patchSize">Patch size. Default: 2.</param>
    /// <param name="contextDim">Context dimension from text encoder. Default: 4096.</param>
    /// <param name="seed">Optional random seed.</param>
    public MMDiTXNoisePredictor(
        MMDiTXVariant variant = MMDiTXVariant.Medium,
        int inputChannels = 16,
        int patchSize = 2,
        int contextDim = 4096,
        int? seed = null)
        : base(seed: seed)
    {
        _variant = variant;
        _inputChannels = inputChannels;
        _hiddenSize = GetHiddenSize(variant);
        _numJointLayers = GetNumLayers(variant);
        _numHeads = GetNumHeads(variant);
        _patchSize = patchSize;
        _contextDim = contextDim;

        InitializeLayers(seed);

        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_jointBlocks), nameof(_finalLayer), nameof(_posEmbed))]
    private void InitializeLayers(int? seed)
    {
        int patchDim = _inputChannels * _patchSize * _patchSize;
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _jointBlocks = new DenseLayer<T>[_numJointLayers];
        for (int i = 0; i < _numJointLayers; i++)
            _jointBlocks[i] = new DenseLayer<T>(_hiddenSize, _hiddenSize, (IActivationFunction<T>)new GELUActivation<T>());

        _finalLayer = new DenseLayer<T>(_hiddenSize, patchDim, (IActivationFunction<T>?)null);
        _posEmbed = new Vector<T>(1024 * _hiddenSize);
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        for (int i = 0; i < _posEmbed.Length; i++)
            _posEmbed[i] = NumOps.FromDouble(rng.NextDouble() * 0.02 - 0.01);
    }

    private int CalculateParameterCount()
    {
        int count = _patchEmbed.ParameterCount;
        foreach (var block in _jointBlocks) count += block.ParameterCount;
        count += _finalLayer.ParameterCount;
        count += _posEmbed.Length;
        return count;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        var timeEmb = GetTimestepEmbedding(timestep);
        var x = _patchEmbed.Forward(noisySample);

        foreach (var block in _jointBlocks)
            x = block.Forward(x);

        return _finalLayer.Forward(x);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddLayerParams(allParams, _patchEmbed);
        foreach (var block in _jointBlocks) AddLayerParams(allParams, block);
        AddLayerParams(allParams, _finalLayer);
        for (int i = 0; i < _posEmbed.Length; i++) allParams.Add(_posEmbed[i]);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetLayerParams(_patchEmbed, parameters, offset);
        foreach (var block in _jointBlocks) offset = SetLayerParams(block, parameters, offset);
        offset = SetLayerParams(_finalLayer, parameters, offset);
        for (int i = 0; i < _posEmbed.Length; i++) _posEmbed[i] = parameters[offset++];
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new MMDiTXNoisePredictor<T>(_variant, _inputChannels, _patchSize, _contextDim);
        clone.SetParameters(GetParameters());
        return clone;
    }

    private static void AddLayerParams(List<T> list, DenseLayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int SetLayerParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    private static int GetHiddenSize(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 2048,
        MMDiTXVariant.LargeTurbo => 2560,
        _ => 2560
    };

    private static int GetNumLayers(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 24,
        _ => 38
    };

    private static int GetNumHeads(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 16,
        _ => 20
    };
}
