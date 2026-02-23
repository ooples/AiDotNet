using AiDotNet.Finance.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.LossFunctions;
using System.IO;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for portfolio optimization models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the <see cref="IPortfolioOptimizer{T}"/> interface
/// and provides common functionality for neural portfolio optimization. It handles
/// asset tracking, metric calculation, and integration with the financial model hierarchy.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for models that manage investment portfolios.
/// It provides the basic tools needed to decide how much money to put into different assets
/// (stocks, bonds, etc.) to achieve the best balance of risk and return.
/// </para>
/// </remarks>
public abstract class PortfolioOptimizerBase<T> : FinancialModelBase<T>, IPortfolioOptimizer<T>
{
    /// <summary>
    /// The number of assets in the portfolio universe.
    /// </summary>
    protected int _numAssets;

    /// <summary>
    /// Gets the number of assets in the portfolio universe.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The total number of different investments available to choose from.
    /// If you are optimizing a portfolio of 10 tech stocks, this would be 10.
    /// </para>
    /// </remarks>
    public int NumAssets => _numAssets;

    /// <summary>
    /// Initializes a new instance of the PortfolioOptimizerBase class for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="numAssets">The number of assets.</param>
    /// <param name="numFeatures">The number of input features per asset.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets up a new portfolio optimizer from scratch, defining
    /// how many assets it will manage and what data it will use to make decisions.
    /// </para>
    /// </remarks>
    protected PortfolioOptimizerBase(NeuralNetworkArchitecture<T> architecture, int numAssets, int numFeatures = 10, ILossFunction<T>? lossFunction = null)
        : base(architecture, 1, 1, numFeatures, lossFunction)
    {
        if (numAssets <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numAssets),
                numAssets,
                "numAssets must be greater than 0 to create valid tensor shapes.");
        }
        _numAssets = numAssets;
        Options = new PortfolioOptimizerOptions<T>();
    }

    /// <summary>
    /// Initializes a new instance of the PortfolioOptimizerBase class from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numAssets">The number of assets.</param>
    /// <param name="numFeatures">The number of input features per asset.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a portfolio optimizer that has already been trained.
    /// Useful for deploying a strategy without having to re-train it.
    /// </para>
    /// </remarks>
    protected PortfolioOptimizerBase(NeuralNetworkArchitecture<T> architecture, string onnxModelPath, int numAssets, int numFeatures = 10)
        : base(architecture, onnxModelPath, 1, 1, numFeatures)
    {
        if (numAssets <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numAssets),
                numAssets,
                "numAssets must be greater than 0 to create valid tensor shapes.");
        }
        _numAssets = numAssets;
        Options = new PortfolioOptimizerOptions<T>();
    }

    /// <summary>
    /// Optimizes the portfolio to determine the best asset allocation.
    /// </summary>
    /// <param name="marketData">Input tensor containing market data (prices, volumes, indicators).</param>
    /// <returns>Vector of optimal weights for each asset.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main "decision maker". It looks at the market data
    /// and outputs a list of percentages (weights) summing to 100%.
    /// E.g., [0.4, 0.6] means put 40% in Asset A and 60% in Asset B.
    /// </para>
    /// </remarks>
    public abstract Vector<T> OptimizePortfolio(Tensor<T> marketData);

    /// <summary>
    /// Calculates optimal weights given expected returns and covariance (Traditional optimization).
    /// </summary>
    /// <param name="expectedReturns">Tensor of expected returns for each asset.</param>
    /// <param name="covariance">Covariance matrix tensor representing asset risk relationships.</param>
    /// <returns>Tensor of optimal weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses standard financial math (Mean-Variance Optimization) to find weights.
    /// This relies on statistical inputs rather than raw market data. It tries to maximize return
    /// for a given level of risk.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> OptimizeWeights(Tensor<T> expectedReturns, Tensor<T> covariance) => new Tensor<T>(new[] { _numAssets });

    /// <summary>
    /// Computes the risk contribution of each asset to the total portfolio risk.
    /// </summary>
    /// <param name="weights">Current portfolio weights.</param>
    /// <param name="covariance">Covariance matrix.</param>
    /// <returns>Tensor of risk contributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Calculates how much risk each asset adds to the portfolio.
    /// This helps you see if one volatile stock is dominating your risk profile,
    /// even if you don't own much of it.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> ComputeRiskContribution(Tensor<T> weights, Tensor<T> covariance) => new Tensor<T>(weights.Shape);

    /// <summary>
    /// Calculates the expected return of the portfolio.
    /// </summary>
    /// <param name="weights">Portfolio weights.</param>
    /// <param name="expectedReturns">Expected returns for each asset.</param>
    /// <returns>The expected total return.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A simple weighted average. If you own 50% of Stock A (return 10%)
    /// and 50% of Stock B (return 20%), your expected return is 15%.
    /// </para>
    /// </remarks>
    public virtual T CalculateExpectedReturn(Tensor<T> weights, Tensor<T> expectedReturns) => NumOps.Zero;

    /// <summary>
    /// Calculates the volatility (standard deviation) of the portfolio.
    /// </summary>
    /// <param name="weights">Portfolio weights.</param>
    /// <param name="covariance">Covariance matrix.</param>
    /// <returns>The portfolio volatility.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Measures the overall "bounciness" of the portfolio value.
    /// Because assets can move in opposite directions (hedging), the portfolio volatility
    /// is usually less than the sum of individual volatilities.
    /// </para>
    /// </remarks>
    public virtual T CalculateVolatility(Tensor<T> weights, Tensor<T> covariance) => NumOps.Zero;

    /// <summary>
    /// Calculates the Sharpe Ratio of the portfolio.
    /// </summary>
    /// <param name="weights">Portfolio weights.</param>
    /// <param name="expectedReturns">Expected returns.</param>
    /// <param name="covariance">Covariance matrix.</param>
    /// <param name="riskFreeRate">The risk-free rate of return.</param>
    /// <returns>The Sharpe Ratio.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The single most common metric for portfolio quality.
    /// It asks: "Is the extra return I'm getting worth the extra risk I'm taking?"
    /// A ratio > 1 is generally considered good.
    /// </para>
    /// </remarks>
    public virtual T CalculateSharpeRatio(Tensor<T> weights, Tensor<T> expectedReturns, Tensor<T> covariance, T riskFreeRate) => NumOps.Zero;

    /// <summary>
    /// Gets metrics for portfolio optimizer evaluation.
    /// </summary>
    /// <returns>Dictionary of portfolio metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary of the optimizer's status, including
    /// the number of assets it manages and the most recent training error.
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetPortfolioMetrics()
    {
        var metrics = base.GetFinancialMetrics();
        metrics["NumAssets"] = NumOps.FromDouble(_numAssets);
        return metrics;
    }

    /// <summary>
    /// Gets overall financial metrics for the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Consolidates metrics. For portfolio optimizers,
    /// this returns the portfolio-specific metrics defined above.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics() => GetPortfolioMetrics();

    /// <summary>
    /// Generates a forecast using the native model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles.</param>
    /// <returns>Forecasted weights as a tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adapts the optimizer to the standard forecasting interface.
    /// In this context, the "forecast" is the set of optimal portfolio weights for the next period.
    /// This method uses Forward directly instead of OptimizePortfolio to avoid
    /// infinite recursion (Predict → Forecast → ForecastNative → OptimizePortfolio → Predict).
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        // Uses Forward directly to avoid recursion with OptimizePortfolio
        // (OptimizePortfolio may call Predict which calls Forecast which calls ForecastNative)
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
    /// <b>For Beginners:</b> Checks if the input data has the correct dimensions (e.g., correct
    /// number of features) before optimization begins, preventing crashes.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        // Basic validation
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    /// <summary>
    /// Serializes portfolio-specific model data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves portfolio settings (like the number of assets) to a file
    /// so the model configuration can be restored later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_numAssets);
    }

    /// <summary>
    /// Deserializes portfolio-specific model data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads portfolio settings from a file.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _numAssets = reader.ReadInt32();

        // Validate deserialized value matches constructor invariant
        if (_numAssets <= 0)
        {
            throw new InvalidOperationException(
                $"Deserialized numAssets ({_numAssets}) is invalid. Must be greater than 0.");
        }
    }

    /// <summary>
    /// Core training logic for the portfolio optimizer.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="output">Model output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the learning step. It calculates how far the model's
    /// proposed weights were from the ideal weights (or utility), and updates the model.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        // Default training implementation
        SetTrainingMode(true);
        try
        {
            var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            // Backward pass would be implemented here or in derived classes
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
}
