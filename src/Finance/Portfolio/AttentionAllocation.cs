using AiDotNet.Finance.Base;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Portfolio optimizer using Multi-Head Attention to capture asset relationships.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Uses the same "attention" mechanism that powers ChatGPT to look at
/// relationships between different stocks. It learns which assets move together or diverge
/// dynamically based on market context, rather than relying on static historical correlations.
/// </para>
/// </remarks>
public class AttentionAllocation<T> : PortfolioOptimizerBase<T>
{
    #region Shared Fields

    private readonly AttentionAllocationOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _hiddenDimension;
    private readonly int _numHeads;
    private readonly int _sequenceLength;
    private readonly double _dropout;

    #endregion

    /// <summary>
    /// Initializes a new instance of the AttentionAllocation model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the AttentionAllocation model, AttentionAllocation sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public AttentionAllocation(
        NeuralNetworkArchitecture<T> architecture,
        AttentionAllocationOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumAssets ?? 10, architecture.CalculatedInputSize, lossFunction)
    {
        _options = options ?? new AttentionAllocationOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenDimension = _options.HiddenDimension;
        _numHeads = _options.NumHeads;
        _sequenceLength = _options.SequenceLength;
        _dropout = _options.DropoutRate;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance from ONNX.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the AttentionAllocation model, AttentionAllocation sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public AttentionAllocation(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        AttentionAllocationOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, options?.NumAssets ?? 10, architecture.CalculatedInputSize)
    {
        _options = options ?? new AttentionAllocationOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenDimension = _options.HiddenDimension;
        _numHeads = _options.NumHeads;
        _sequenceLength = _options.SequenceLength;
        _dropout = _options.DropoutRate;

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the AttentionAllocation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the AttentionAllocation model, InitializeLayers builds and wires up model components. This sets up the AttentionAllocation architecture before use.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultAttentionAllocationLayers(
                Architecture,
                NumFeatures,
                _hiddenDimension,
                _numHeads,
                _numAssets,
                _sequenceLength,
                _dropout));
        }
    }

    #endregion

    /// <summary>
    /// Optimizes portfolio weights using attention mechanism.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the AttentionAllocation model, OptimizePortfolio computes portfolio weights. This is the decision output the AttentionAllocation architecture is designed to learn.
    /// </para>
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        var prediction = Predict(marketData);
        return prediction.ToVector();
    }

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Updates the model parameters from a flat parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you load or apply a full set of weights at once,
    /// which is useful for cloning models or applying externally optimized parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            layer.SetParameters(parameters.Slice(offset, layerParams.Length));
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Creates a new instance of the AttentionAllocation model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used to clone the model settings so the framework
    /// can create fresh instances with identical behavior.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new AttentionAllocationOptions<T>
        {
            NumAssets = _options.NumAssets,
            HiddenDimension = _hiddenDimension,
            NumHeads = _numHeads,
            SequenceLength = _sequenceLength,
            DropoutRate = _dropout
        };

        return new AttentionAllocation<T>(Architecture, optionsCopy, lossFunction: _lossFunction);
    }

    #endregion
}
