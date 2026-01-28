using AiDotNet.Finance.Base;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Enums;
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
        : base(architecture, options?.NumAssets ?? 10, options?.NumFeatures ?? architecture.CalculatedInputSize, lossFunction)
    {
        _options = options ?? new HierarchicalRiskParityOptions<T>();
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
        : base(architecture, onnxModelPath, options?.NumAssets ?? 10, options?.NumFeatures ?? architecture.CalculatedInputSize)
    {
        _options = options ?? new HierarchicalRiskParityOptions<T>();
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
                _numFeatures,
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
        var prediction = Predict(marketData);
        return prediction.ToVector();
    }
}
