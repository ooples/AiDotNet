using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using System.IO;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for risk management models, providing common infrastructure for risk assessment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the <see cref="IRiskModel{T}"/> interface and provides the
/// foundation for neural network-based risk models. It handles common tasks like model configuration,
/// serialization, and basic metric tracking.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "parent" for all risk models. It handles the boring stuff
/// (like saving settings to a file or checking input shapes) so that specific models (like NeuralVaR)
/// can focus on the actual math of predicting risk.
/// </para>
/// </remarks>
public abstract class RiskModelBase<T> : FinancialModelBase<T>, IRiskModel<T>
{
    /// <summary>
    /// The confidence level used for risk calculations (e.g., 0.95 or 0.99).
    /// </summary>
    protected double _confidenceLevel;

    /// <summary>
    /// The time horizon for risk calculations in days.
    /// </summary>
    protected int _timeHorizon;

    /// <summary>
    /// Gets the confidence level for risk calculations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This number represents how sure we want to be.
    /// A confidence level of 0.95 (95%) means we want to be safe 19 days out of 20.
    /// Only on 1 day out of 20 do we expect losses to potentially exceed our estimate.
    /// </para>
    /// </remarks>
    public double ConfidenceLevel => _confidenceLevel;

    /// <summary>
    /// Gets the time horizon for risk calculations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "prediction window" for risk.
    /// A value of 1 means "what's the risk for tomorrow?".
    /// A value of 10 means "what's the risk over the next 10 days?".
    /// </para>
    /// </remarks>
    public int TimeHorizon => _timeHorizon;

    /// <summary>
    /// Initializes a new instance of the RiskModelBase class for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="confidenceLevel">Confidence level (default 0.95).</param>
    /// <param name="timeHorizon">Time horizon in days (default 1).</param>
    /// <param name="lossFunction">Loss function for training.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a risk model ready to be trained.
    /// You specify the network shape, what kind of data goes in, and your risk preferences.
    /// </para>
    /// </remarks>
    protected RiskModelBase(NeuralNetworkArchitecture<T> architecture, int numFeatures, double confidenceLevel = 0.95, int timeHorizon = 1, ILossFunction<T>? lossFunction = null)
        : base(architecture, 1, 1, numFeatures, lossFunction)
    {
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");
        if (timeHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(timeHorizon), "Time horizon must be at least 1.");
        _confidenceLevel = confidenceLevel;
        _timeHorizon = timeHorizon;
        Options = new RiskModelOptions<T>();
    }

    /// <summary>
    /// Initializes a new instance of the RiskModelBase class from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="confidenceLevel">Confidence level (default 0.95).</param>
    /// <param name="timeHorizon">Time horizon in days (default 1).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a risk model that has already been trained.
    /// This is used when you want to calculate risk immediately without training a new network.
    /// </para>
    /// </remarks>
    protected RiskModelBase(NeuralNetworkArchitecture<T> architecture, string onnxModelPath, int numFeatures, double confidenceLevel = 0.95, int timeHorizon = 1)
        : base(architecture, onnxModelPath, 1, 1, numFeatures)
    {
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");
        if (timeHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(timeHorizon), "Time horizon must be at least 1.");
        _confidenceLevel = confidenceLevel;
        _timeHorizon = timeHorizon;
        Options = new RiskModelOptions<T>();
    }

    /// <summary>
    /// Calculates the primary risk metric for the given input.
    /// </summary>
    /// <param name="input">Input tensor representing market conditions or portfolio state.</param>
    /// <returns>The calculated risk value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main "danger meter" method.
    /// You give it the current market situation, and it returns a single number representing risk.
    /// Higher numbers usually mean more danger/potential loss.
    /// </para>
    /// </remarks>
    public abstract T CalculateRisk(Tensor<T> input);

    /// <summary>
    /// Adjusts a proposed action to satisfy risk constraints.
    /// </summary>
    /// <param name="action">The proposed trading action (e.g., portfolio weights).</param>
    /// <param name="riskBudget">The maximum allowable risk.</param>
    /// <returns>The risk-adjusted action.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This acts as a "safety filter" or "governor".
    /// If a trading bot wants to make a trade that is too risky (exceeds the budget),
    /// this method scales it back to a safe level. It ensures you don't bet the farm.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> AdjustForRisk(Tensor<T> action, T riskBudget);

    /// <summary>
    /// Calculates Value at Risk (VaR).
    /// </summary>
    /// <param name="portfolioReturns">Historical portfolio returns.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <returns>The Value at Risk estimate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VaR answers: "What is the worst loss I expect to see in normal markets?"
    /// If 95% VaR is $1000, it means you are 95% confident you won't lose more than $1000.
    /// It helps set aside enough "rainy day" money.
    /// </para>
    /// </remarks>
    public virtual T CalculateVaR(Tensor<T> portfolioReturns, Tensor<T> weights) => NumOps.Zero;

    /// <summary>
    /// Calculates Conditional Value at Risk (CVaR).
    /// </summary>
    /// <param name="portfolioReturns">Historical portfolio returns.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <returns>The Expected Shortfall estimate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CVaR answers: "If things DO go wrong (beyond normal markets), how bad will it be?"
    /// While VaR is the threshold, CVaR is the average of the losses *beyond* that threshold.
    /// It measures the disaster scenario.
    /// </para>
    /// </remarks>
    public virtual T CalculateCVaR(Tensor<T> portfolioReturns, Tensor<T> weights) => NumOps.Zero;

    /// <summary>
    /// Performs stress testing under specific scenarios.
    /// </summary>
    /// <param name="portfolioWeights">Current portfolio weights.</param>
    /// <param name="stressScenarios">Tensor of stress scenarios.</param>
    /// <returns>Estimated loss for each scenario.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stress testing is asking "What if?".
    /// What if the stock market crashes 20%? What if interest rates double?
    /// This method calculates your portfolio's value in these hypothetical disaster situations.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> StressTest(Tensor<T> portfolioWeights, Tensor<T> stressScenarios) => new Tensor<T>(new[] { stressScenarios.Shape[0] });

    /// <summary>
    /// Decomposes total risk into contributions from individual assets.
    /// </summary>
    /// <param name="portfolioReturns">Historical returns.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <returns>Risk contribution for each asset.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This breaks down the risk pie.
    /// It tells you which specific investments are causing the most danger in your portfolio.
    /// Useful for finding the "bad apples" to remove if you need to lower risk.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> DecomposeRisk(Tensor<T> portfolioReturns, Tensor<T> weights) => new Tensor<T>(weights.Shape);

    /// <summary>
    /// Estimates the probability of losses exceeding a threshold.
    /// </summary>
    /// <param name="portfolioReturns">Historical returns.</param>
    /// <param name="weights">Portfolio weights.</param>
    /// <param name="lossThreshold">The loss amount to check against.</param>
    /// <returns>Probability of exceeding the loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This calculates the odds of a specific bad event.
    /// For example: "What is the % chance I lose more than $10,000?"
    /// </para>
    /// </remarks>
    public virtual T EstimateExceedanceProbability(Tensor<T> portfolioReturns, Tensor<T> weights, T lossThreshold) => NumOps.Zero;

    /// <summary>
    /// Gets metrics for risk model evaluation.
    /// </summary>
    /// <returns>Dictionary of risk metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a report card for the risk model.
    /// It lists settings like confidence level and performance stats like the last training error.
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetRiskMetrics()
    {
        var metrics = base.GetFinancialMetrics();
        metrics["ConfidenceLevel"] = NumOps.FromDouble(_confidenceLevel);
        return metrics;
    }

    /// <summary>
    /// Gets overall financial metrics for the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Consolidates all model metrics. For risk models,
    /// this just redirects to the risk-specific metrics.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics() => GetRiskMetrics();

    /// <summary>
    /// Generates a forecast using the native model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles.</param>
    /// <returns>Forecasted risk value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adapts the risk calculation to the standard forecasting interface.
    /// It treats "future risk" as the thing being forecasted.
    /// This method uses Forward directly instead of CalculateRisk to avoid
    /// infinite recursion (Predict → Forecast → ForecastNative → CalculateRisk → Predict).
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        // Default forecasting implementation for risk models
        // Uses Forward directly to avoid recursion with CalculateRisk
        // (CalculateRisk calls Predict which calls Forecast which calls ForecastNative)
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Validates the input tensor shape.
    /// </summary>
    /// <param name="input">Input tensor to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks if the input data has the right "shape" (dimensions)
    /// before the model tries to process it. Prevents errors from mismatched data.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    /// <summary>
    /// Serializes risk-specific model data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the risk settings (confidence level, time horizon) to a file
    /// so the model remembers them when loaded later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_confidenceLevel);
        writer.Write(_timeHorizon);
    }

    /// <summary>
    /// Deserializes risk-specific model data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads the saved risk settings from a file.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _confidenceLevel = reader.ReadDouble();
        _timeHorizon = reader.ReadInt32();

        // Re-validate invariants after deserialize to prevent corrupt state
        // Use same exclusive bounds as constructor: confidenceLevel must be in (0, 1)
        if (_confidenceLevel <= 0 || _confidenceLevel >= 1)
        {
            throw new InvalidOperationException(
                $"Deserialized confidenceLevel ({_confidenceLevel}) is invalid. Must be between 0 and 1 (exclusive).");
        }
        if (_timeHorizon < 1)
        {
            throw new InvalidOperationException(
                $"Deserialized timeHorizon ({_timeHorizon}) is invalid. Must be at least 1.");
        }
    }

    /// <summary>
    /// Core training logic for the risk model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="output">Model output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the learning step. It calculates the error between
    /// predicted risk and actual outcomes, and adjusts the model to be more accurate.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        try
        {
            var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            var gradTensor = Tensor<T>.FromVector(grad, output.Shape);

            Backpropagate(gradTensor);

            // Default SGD-style update for layers
            var learningRate = MathHelper.GetNumericOperations<T>().FromDouble(0.001);
            foreach (var layer in Layers)
            {
                layer.UpdateParameters(learningRate);
            }
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
}
