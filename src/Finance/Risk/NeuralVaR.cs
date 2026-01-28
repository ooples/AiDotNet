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

namespace AiDotNet.Finance.Risk;

/// <summary>
/// Neural Value-at-Risk (VaR) model for non-linear market risk assessment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NeuralVaR uses deep neural networks to estimate the potential loss of a portfolio
/// under various market conditions, accounting for complex non-linear dependencies.
/// </para>
/// <para><b>For Beginners:</b> Value-at-Risk (VaR) is a way to answer the question: 
/// "What is the most I could lose on this investment tomorrow with 95% confidence?"
/// Traditional methods often assume simple patterns, but this AI model "learns" 
/// from historical market crashes and complex trends to give a more realistic 
/// estimate of risk.
/// </para>
/// <para>
/// Reference: Riskfuel, "Neural Value at Risk", 2021.
/// </para>
/// </remarks>
public class NeuralVaR<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a NeuralVaR model using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained NeuralVaR
    /// model saved as an ONNX file and want to run predictions quickly.
    /// </para>
    /// </remarks>
    public NeuralVaR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        VaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, onnxModelPath, options?.NumFeatures ?? 10, 
               options?.ConfidenceLevel ?? 0.95, options?.TimeHorizon ?? 1)
    {
        options ??= new VaROptions<T>();
        options.Validate();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a NeuralVaR model in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train NeuralVaR
    /// from scratch on your own data.
    /// </para>
    /// </remarks>
    public NeuralVaR(
        NeuralNetworkArchitecture<T> architecture,
        VaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, options?.NumFeatures ?? 10, 
               options?.ConfidenceLevel ?? 0.95, options?.TimeHorizon ?? 1, 
               lossFunction)
    {
        options ??= new VaROptions<T>();
        options.Validate();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, InitializeLayers builds and wires up model components. This sets up the NeuralVaR architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralVaRLayers(Architecture, _numFeatures));
        }
    }

    #endregion

    #region Risk Calculation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, CalculateRisk estimates risk values. This is the core signal the NeuralVaR architecture focuses on.
    /// </para>
    /// </remarks>
    public override T CalculateRisk(Tensor<T> input)
    {
        var output = Predict(input);
        return output.Data.Span[0];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, AdjustForRisk estimates risk values. This is the core signal the NeuralVaR architecture focuses on.
    /// </para>
    /// </remarks>
    public override Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget)
    {
        // For Beginners: If the calculated risk exceeds the budget, scale down the action.
        T currentRisk = CalculateRisk(action);
        if (NumOps.ToDouble(currentRisk) > NumOps.ToDouble(riskBudget))
        {
            T scale = NumOps.Divide(riskBudget, currentRisk);
            var scaledData = new T[action.Length];
            for (int i = 0; i < action.Length; i++)
                scaledData[i] = NumOps.Multiply(action.Data.Span[i], scale);
            
            return new Tensor<T>(action.Shape, new Vector<T>(scaledData));
        }

        return action;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes Predict for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, Predict produces predictions from input data. This is the main inference step of the NeuralVaR architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Executes Train for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, Train performs a training step. This updates the NeuralVaR architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training not supported in ONNX mode.");
        
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        
        var currentGrad = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--)
            currentGrad = Layers[i].Backward(currentGrad);

        _optimizer.UpdateParameters(Layers);
        _lastTrainingLoss = LossFunction.CalculateLoss(output.ToVector(), target.ToVector());
        SetTrainingMode(false);
    }

    /// <summary>
    /// Executes UpdateParameters for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, UpdateParameters updates internal parameters or state. This keeps the NeuralVaR architecture aligned with the latest values.
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
    /// Executes GetModelMetadata for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, GetModelMetadata performs a supporting step in the workflow. It keeps the NeuralVaR architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelType", "NeuralVaR" },
                { "ConfidenceLevel", _confidenceLevel },
                { "TimeHorizon", _timeHorizon },
                { "ParameterCount", GetParameterCount() }
            }
        };
    }

    /// <summary>
    /// Executes CreateNewInstance for the NeuralVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NeuralVaR model, CreateNewInstance builds and wires up model components. This sets up the NeuralVaR architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new VaROptions<T> { NumFeatures = _numFeatures, ConfidenceLevel = _confidenceLevel, TimeHorizon = _timeHorizon };
        return new NeuralVaR<T>(Architecture, options, _optimizer, LossFunction);
    }

    #endregion
}
