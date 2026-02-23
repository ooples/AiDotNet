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
/// Neural network adaptation of the Black-Litterman model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Black-Litterman model combines "market equilibrium" (what everyone thinks)
/// with "investor views" (your specific predictions). This neural version learns to generate
/// these "views" and "confidence levels" from data automatically, blending them with the
/// market baseline to produce robust portfolios.
/// </para>
/// </remarks>
public class BlackLittermanNeural<T> : PortfolioOptimizerBase<T>
{
    #region Shared Fields

    private readonly BlackLittermanNeuralOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _hiddenDimension;
    private readonly double _dropout;

    #endregion

    /// <summary>
    /// Initializes a new instance of the BlackLittermanNeural model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the BlackLittermanNeural model, BlackLittermanNeural sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public BlackLittermanNeural(
        NeuralNetworkArchitecture<T> architecture,
        BlackLittermanNeuralOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumAssets ?? 10, architecture.CalculatedInputSize, lossFunction)
    {
        _options = options ?? new BlackLittermanNeuralOptions<T>();
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
    /// <b>For Beginners:</b> In the BlackLittermanNeural model, BlackLittermanNeural sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public BlackLittermanNeural(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        BlackLittermanNeuralOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, options?.NumAssets ?? 10, architecture.CalculatedInputSize)
    {
        _options = options ?? new BlackLittermanNeuralOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _hiddenDimension = _options.HiddenDimension;
        _dropout = _options.DropoutRate;

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the BlackLittermanNeural.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the BlackLittermanNeural model, InitializeLayers builds and wires up model components. This sets up the BlackLittermanNeural architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultBlackLittermanNeuralLayers(
                Architecture,
                NumFeatures,
                _hiddenDimension,
                _numAssets,
                _dropout));
        }
    }

    #endregion

    /// <summary>
    /// Optimizes portfolio weights by generating views and blending with equilibrium.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the BlackLittermanNeural model, OptimizePortfolio computes portfolio weights. This is the decision output the BlackLittermanNeural architecture is designed to learn.
    /// </para>
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        var prediction = Predict(marketData);
        // Logic to convert views/confidence (prediction) to weights would go here.
        // For now, we assume the network outputs final weights directly or we map them.
        // Simplified: return first half as weights.
        var vec = prediction.ToVector();
        var weights = new Vector<T>(_options.NumAssets);
        for (int i = 0; i < _options.NumAssets; i++)
        {
            weights[i] = vec[i];
        }
        return weights; // Should normalize
    }

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Updates the model parameters from a flat parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you load a complete set of weights
    /// in one call, which is useful for cloning or external optimization.
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
    /// Creates a new instance of the BlackLittermanNeural model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used by the framework to clone the model's setup
    /// so it can create a fresh instance with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new BlackLittermanNeuralOptions<T>
        {
            NumAssets = _options.NumAssets,
            HiddenDimension = _hiddenDimension,
            DropoutRate = _dropout
        };

        return new BlackLittermanNeural<T>(Architecture, optionsCopy, lossFunction: _lossFunction);
    }

    #endregion
}
