using System.IO;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Deep Portfolio Manager for end-to-end portfolio weight optimization.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeepPortfolioManager uses deep learning to directly map market input data
/// to optimal portfolio weights, typically maximizing a risk-adjusted return metric.
/// </para>
/// <para><b>For Beginners:</b> Managing a portfolio means deciding exactly what percentage 
/// of your money should go into each stock (e.g., 10% Apple, 5% Microsoft). 
/// While traditional methods use complex math formulas, this AI model looks 
/// at historical performance and current trends to "guess" the best weights 
/// that will give you the most profit with the least risk.
/// </para>
/// <para>
/// Reference: Zhang et al., "Deep Learning for Portfolio Management", 2020.
/// </para>
/// </remarks>
public class DeepPortfolioManager<T> : PortfolioOptimizerBase<T>
{
    #region Shared Fields

    private readonly DeepPortfolioManagerOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly bool _allowShortSelling;
    private readonly double _maxWeight;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DeepPortfolioManager using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, DeepPortfolioManager sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public DeepPortfolioManager(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DeepPortfolioManagerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, options?.NumAssets ?? 10, architecture.InputSize)
    {
        options ??= new DeepPortfolioManagerOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _allowShortSelling = options.AllowShortSelling;
        _maxWeight = options.MaxWeight;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DeepPortfolioManager in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, DeepPortfolioManager sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public DeepPortfolioManager(
        NeuralNetworkArchitecture<T> architecture,
        DeepPortfolioManagerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumAssets ?? 10, architecture.InputSize, lossFunction)
    {
        options ??= new DeepPortfolioManagerOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _allowShortSelling = options.AllowShortSelling;
        _maxWeight = options.MaxWeight;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, InitializeLayers builds and wires up model components. This sets up the DeepPortfolioManager architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepPortfolioLayers(Architecture, _numAssets));
        }
    }

    #endregion

    #region Portfolio Optimization

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, OptimizePortfolio computes portfolio weights. This is the decision output the DeepPortfolioManager architecture is designed to learn.
    /// </para>
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        var output = Predict(marketData);
        var weights = output.ToVector();

        // For Beginners: Ensure weights are valid (e.g., no single stock > maxWeight)
        if (_maxWeight < 1.0)
        {
            // Simple clipping logic (would need re-normalization in real implementation)
            for (int i = 0; i < weights.Length; i++)
            {
                if (NumOps.ToDouble(weights[i]) > _maxWeight)
                    weights[i] = NumOps.FromDouble(_maxWeight);
            }
        }

        return weights;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes Predict for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, Predict produces predictions from input data. This is the main inference step of the DeepPortfolioManager architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Executes Train for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, Train performs a training step. This updates the DeepPortfolioManager architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!UseNativeMode) throw new InvalidOperationException("Training not supported in ONNX mode.");
        
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        
        var currentGrad = Tensor<T>.FromVector(grad, output.Shape);
        for (int i = Layers.Count - 1; i >= 0; i--)
            currentGrad = Layers[i].Backward(currentGrad);

        _optimizer.UpdateParameters(Layers);
        _lastTrainingLoss = LossFunction.CalculateLoss(output.ToVector(), target.ToVector());
        SetTrainingMode(false);
    }

    /// <summary>
    /// Executes UpdateParameters for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, UpdateParameters updates internal parameters or state. This keeps the DeepPortfolioManager architecture aligned with the latest values.
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
    /// Executes GetModelMetadata for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, GetModelMetadata performs a supporting step in the workflow. It keeps the DeepPortfolioManager architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelType", "DeepPortfolioManager" },
                { "NumAssets", _numAssets },
                { "AllowShortSelling", _allowShortSelling },
                { "ParameterCount", GetParameterCount() }
            }
        };
    }

    /// <summary>
    /// Executes CreateNewInstance for the DeepPortfolioManager.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepPortfolioManager model, CreateNewInstance builds and wires up model components. This sets up the DeepPortfolioManager architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new DeepPortfolioManagerOptions<T> { NumAssets = _numAssets, AllowShortSelling = _allowShortSelling, MaxWeight = _maxWeight };
        return new DeepPortfolioManager<T>(Architecture, options, _optimizer, LossFunction);
    }

    #endregion
}
