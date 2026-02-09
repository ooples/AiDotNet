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
/// Neural Hierarchical Risk Parity (HRP) for portfolio optimization.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> HRP is a modern portfolio construction technique that uses machine learning
/// to cluster similar assets together and then allocates risk equally across these clusters.
/// Unlike traditional Mean-Variance Optimization, it doesn't require inverting a covariance matrix,
/// making it robust to noise and stable even when assets are highly correlated.
/// </para>
/// </remarks>
public class HierarchicalRiskParity<T> : PortfolioOptimizerBase<T>
{
    #region Shared Fields

    private readonly HierarchicalRiskParityOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _hiddenDimension;
    private readonly double _dropout;

    #endregion

    /// <summary>
    /// Initializes a new instance of the HierarchicalRiskParity model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the HierarchicalRiskParity model, HierarchicalRiskParity sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public HierarchicalRiskParity(
        NeuralNetworkArchitecture<T> architecture,
        HierarchicalRiskParityOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumAssets ?? 10, architecture.CalculatedInputSize, lossFunction)
    {
        _options = options ?? new HierarchicalRiskParityOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenDimension = _options.HiddenDimension;
        _dropout = _options.DropoutRate;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance from ONNX.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the HierarchicalRiskParity model, HierarchicalRiskParity sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public HierarchicalRiskParity(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        HierarchicalRiskParityOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, options?.NumAssets ?? 10, architecture.CalculatedInputSize)
    {
        _options = options ?? new HierarchicalRiskParityOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenDimension = _options.HiddenDimension;
        _dropout = _options.DropoutRate;

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the HierarchicalRiskParity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the HierarchicalRiskParity model, InitializeLayers builds and wires up model components. This sets up the HierarchicalRiskParity architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultHierarchicalRiskParityLayers(
                Architecture,
                NumFeatures,
                _hiddenDimension,
                _numAssets,
                _dropout));
        }
    }

    #endregion

    /// <summary>
    /// Optimizes portfolio weights using HRP logic.
    /// </summary>
    /// <param name="marketData">Market data tensor.</param>
    /// <returns>Optimized weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the optimal percentage of capital to allocate to each asset.
    /// The weights will sum to 1.0 (100%).
    /// </para>
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        var prediction = UseNativeMode ? Forward(marketData) : ForecastOnnx(marketData);
        return prediction.ToVector();
    }

    /// <summary>
    /// Executes a forward pass through the network layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the input through each layer to produce
    /// portfolio weight scores without calling the forecasting wrapper.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Updates the model parameters from a flat parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets the model load or apply all its weights at once,
    /// which is helpful when cloning or restoring trained parameters.
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
    /// Creates a new instance of the HierarchicalRiskParity model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used by the framework to clone the model's configuration
    /// so it can create a fresh instance with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new HierarchicalRiskParityOptions<T>
        {
            NumAssets = _options.NumAssets,
            HiddenDimension = _hiddenDimension,
            DropoutRate = _dropout
        };

        return new HierarchicalRiskParity<T>(Architecture, optionsCopy, lossFunction: _lossFunction);
    }

    #endregion
}
