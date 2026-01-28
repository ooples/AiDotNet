using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.Risk;

/// <summary>
/// A neural network model for estimating Conditional Value at Risk (CVaR), also known as Expected Shortfall.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> CVaR (Conditional Value at Risk) answers the question: "If things get really bad (worse than VaR),
/// how much will I lose on average?"
///
/// While VaR (Value at Risk) gives you a threshold (e.g., "I'm 95% sure I won't lose more than $100"),
/// CVaR tells you the average loss in that remaining 5% of worst-case scenarios (e.g., "$150").
///
/// This model uses a neural network to predict this value based on market conditions. It's often considered
/// a better risk measure than VaR because it accounts for the magnitude of extreme losses ("fat tails").
/// </para>
/// </remarks>
public class NeuralCVaR<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly NeuralCVaROptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    /// <summary>
    /// Initializes a new instance of the NeuralCVaR model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when training a NeuralCVaR model from scratch.
    /// It sets up a small feed-forward network that predicts expected shortfall.
    /// </para>
    /// </remarks>
    public NeuralCVaR(
        NeuralNetworkArchitecture<T> architecture,
        NeuralCVaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.NumFeatures ?? 10,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1,
            lossFunction)
    {
        _options = options ?? new NeuralCVaROptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the NeuralCVaR model from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a pretrained NeuralCVaR
    /// model in ONNX format and want to run predictions.
    /// </para>
    /// </remarks>
    public NeuralCVaR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NeuralCVaROptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.NumFeatures ?? 10,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1)
    {
        _options = options ?? new NeuralCVaROptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for NeuralCVaR.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This builds a simple feed-forward network that maps
    /// input features to a single CVaR risk number. If you provided custom layers,
    /// those are used instead.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLayers(
                Architecture,
                _options.HiddenLayers,
                _options.HiddenDimension,
                1));
        }
    }

    #endregion

    /// <summary>
    /// Calculates the Conditional Value at Risk for the given input.
    /// </summary>
    /// <param name="input">Input tensor representing market conditions.</param>
    /// <returns>The estimated CVaR value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method asks the neural network: "Given these market conditions,
    /// what is the average loss in the worst-case scenarios?"
    /// The output is a single positive number representing the expected loss amount.
    /// </para>
    /// </remarks>
    public override T CalculateRisk(Tensor<T> input)
    {
        var prediction = Predict(input);
        // CVaR is typically positive (loss amount)
        return NumOps.Abs(prediction.ToVector()[0]);
    }

    /// <summary>
    /// Adjusts a proposed action to satisfy CVaR constraints.
    /// </summary>
    /// <param name="action">The proposed trading action.</param>
    /// <param name="riskBudget">The maximum allowable CVaR.</param>
    /// <returns>The risk-adjusted action.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the predicted risk (CVaR) of an action is too high,
    /// this method scales down the action (e.g., reduces position size) until the
    /// risk is within your budget.
    /// </para>
    /// </remarks>
    public override Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget)
    {
        // Simple scaling: if current risk > budget, scale action down
        // Note: This assumes risk scales linearly with action size (approximate)
        var currentRisk = CalculateRisk(action); // This is a simplification; usually risk is function of State + Action
        
        if (NumOps.GreaterThan(currentRisk, riskBudget))
        {
            var ratio = NumOps.Divide(riskBudget, currentRisk);
            // Apply scaling
            return action.Multiply(ratio);
        }

        return action;
    }

    /// <summary>
    /// Calculates VaR as a byproduct of CVaR estimation (using Gaussian assumption or auxiliary output).
    /// </summary>
    /// <param name="portfolioReturns">Historical returns.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <returns>VaR estimate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While this model focuses on CVaR, we can often infer VaR from it.
    /// Since CVaR is the average of losses beyond VaR, VaR is always smaller (less severe) than CVaR.
    /// </para>
    /// </remarks>
    public override T CalculateVaR(Tensor<T> portfolioReturns, Tensor<T> weights)
    {
        // Simplification: VaR ~= CVaR * adjustment_factor (e.g., 0.8)
        // In a real implementation, the network might output both
        var cvar = CalculateRisk(portfolioReturns); // Using returns as input state proxy
        return NumOps.Multiply(cvar, NumOps.FromDouble(0.8));
    }
}
