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
/// Neural network model for generating and evaluating stress test scenarios.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural Stress Testing learns to predict how a portfolio behaves under
/// extreme market conditions. Unlike traditional stress testing which uses fixed historical scenarios,
/// this model can generate new, plausible crisis scenarios (like a "Deep Fake" market crash)
/// to test portfolio resilience against events that haven't happened yet.
/// </para>
/// </remarks>
public class NeuralStressTest<T> : RiskModelBase<T>
{
    #region Shared Fields

    private readonly NeuralStressTestOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    /// <summary>
    /// Initializes a new instance of the NeuralStressTest model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when training a stress test model
    /// from scratch. The model learns to simulate extreme scenarios for your portfolio.
    /// </para>
    /// </remarks>
    public NeuralStressTest(
        NeuralNetworkArchitecture<T> architecture,
        NeuralStressTestOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            options?.NumFeatures ?? 10,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1,
            lossFunction)
    {
        _options = options ?? new NeuralStressTestOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance from ONNX.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a pretrained stress-test
    /// model saved as an ONNX file and want to run predictions.
    /// </para>
    /// </remarks>
    public NeuralStressTest(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NeuralStressTestOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            architecture,
            onnxModelPath,
            options?.NumFeatures ?? 10,
            options?.ConfidenceLevel ?? 0.95,
            options?.TimeHorizon ?? 1)
    {
        _options = options ?? new NeuralStressTestOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for NeuralStressTest.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up the default layers that generate or
    /// evaluate multiple stress scenarios. If you provided custom layers, those
    /// are used instead.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralStressTestLayers(
                Architecture,
                _options.NumFeatures,
                _options.HiddenDimension,
                _options.NumScenarios,
                _options.DropoutRate));
        }
    }

    #endregion

    /// <summary>
    /// Calculates aggregate risk across generated stress scenarios.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model outputs multiple "bad day" scenarios.
    /// This method averages the size of those losses into one risk score.
    /// </para>
    /// </remarks>
    public override T CalculateRisk(Tensor<T> input)
    {
        var scenarios = Predict(input);
        // Average loss across scenarios
        T sum = NumOps.Zero;
        var vec = scenarios.ToVector();
        for (int i = 0; i < vec.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(vec[i]));
        }
        return NumOps.Divide(sum, NumOps.FromDouble(vec.Length));
    }

    /// <summary>
    /// Adjusts action based on stress test results.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the stress-test risk is too high, this scales
    /// the action down so it stays within the allowed budget.
    /// </para>
    /// </remarks>
    public override Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget)
    {
        var risk = CalculateRisk(action);
        if (NumOps.GreaterThan(risk, riskBudget))
        {
            return action.Multiply(NumOps.Divide(riskBudget, risk));
        }
        return action;
    }

    /// <summary>
    /// Generates stress scenarios for the given input state.
    /// </summary>
    /// <param name="input">Market state.</param>
    /// <returns>Tensor of stress scenarios.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generates a set of hypothetical "bad days"
    /// based on the current market state so you can see how the portfolio behaves
    /// under extreme conditions.
    /// </para>
    /// </remarks>
    public override Tensor<T> StressTest(Tensor<T> input, Tensor<T> stressScenarios)
    {
        // In this implementation, the model PREDICTS the impact of scenarios
        // So input is Portfolio Weights + Market State?
        // Or input is Market State and output is Scenario Impacts?
        // We stick to standard Predict which outputs scenario impacts.
        return Predict(input);
    }
}
