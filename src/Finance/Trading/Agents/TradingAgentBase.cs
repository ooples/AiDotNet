using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.LossFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// Base class for financial trading agents using reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TradingAgentBase extends the standard RL agent infrastructure with financial-specific
/// functionality. It provides common implementation for portfolio tracking, risk management,
/// and financial metrics that all trading agents share.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for AI trading agents:
///
/// <b>The Key Insight:</b>
/// Trading is a sequential decision-making problem perfectly suited for RL.
/// The agent observes market conditions, makes trading decisions, and learns from
/// the resulting profits or losses. Over many training episodes, it learns strategies
/// that maximize risk-adjusted returns.
///
/// <b>Architecture:</b>
/// - Inherits from ReinforcementLearningAgentBase (standard RL functionality)
/// - Implements ITradingAgent (financial-specific methods)
/// - Adds portfolio tracking (values, returns, drawdown)
/// - Adds financial metrics (Sharpe ratio, win rate)
/// - Integrates risk management (position limits, VaR constraints)
///
/// <b>How Trading Agents Learn:</b>
/// 1. Observe market state (prices, volumes, indicators)
/// 2. Take action (buy, sell, adjust positions)
/// 3. Receive reward (returns, risk-adjusted returns)
/// 4. Store experience in replay buffer
/// 5. Sample experiences and update neural network
/// 6. Repeat thousands of times until convergence
/// </para>
/// </remarks>
public abstract class TradingAgentBase<T> : ReinforcementLearningAgentBase<T>, ITradingAgent<T>
{
    #region Fields

    /// <summary>
    /// Options specific to financial trading.
    /// </summary>
    protected readonly TradingAgentOptions<T> TradingOptions;

    /// <summary>
    /// History of portfolio values over time.
    /// </summary>
    protected readonly List<T> PortfolioHistory;

    /// <summary>
    /// History of individual trade P&amp;L values.
    /// </summary>
    protected readonly List<T> TradeHistory;

    /// <summary>
    /// Current portfolio value.
    /// </summary>
    protected T _portfolioValue;

    /// <summary>
    /// Initial capital at the start of trading.
    /// </summary>
    protected T _initialCapital;

    /// <summary>
    /// Tracks peak portfolio value for drawdown calculation.
    /// </summary>
    protected T _peakValue;

    /// <summary>
    /// Count of profitable trades.
    /// </summary>
    protected int _winningTrades;

    /// <summary>
    /// Total number of trades executed.
    /// </summary>
    protected int _totalTrades;

    #endregion

    #region ITradingAgent Properties

    /// <inheritdoc/>
    public T SharpeRatio => CalculateSharpeRatio();

    /// <inheritdoc/>
    public T MaxDrawdown => CalculateMaxDrawdown();

    /// <inheritdoc/>
    public T CumulativeReturn => CalculateCumulativeReturn();

    /// <inheritdoc/>
    public T WinRate => CalculateWinRate();

    /// <inheritdoc/>
    public int TotalTrades => _totalTrades;

    /// <inheritdoc/>
    public T PortfolioValue => _portfolioValue;

    /// <inheritdoc/>
    public T InitialCapital => _initialCapital;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TradingAgentBase class.
    /// </summary>
    /// <param name="options">Configuration options for the trading agent.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The constructor sets up everything the trading agent needs:
    /// - Initializes the base RL agent (neural networks, replay buffer)
    /// - Sets initial capital for portfolio tracking
    /// - Creates empty histories for metrics calculation
    /// - Configures trading-specific parameters (transaction costs, position limits)
    /// </para>
    /// </remarks>
    protected TradingAgentBase(TradingAgentOptions<T> options)
        : base(CreateBaseOptions(options))
    {
        TradingOptions = options ?? throw new ArgumentNullException(nameof(options));
        if (TradingOptions.LossFunction == null)
        {
            TradingOptions.LossFunction = new MeanSquaredErrorLoss<T>();
        }
        _initialCapital = options.InitialCapital;
        _portfolioValue = options.InitialCapital;
        _peakValue = options.InitialCapital;
        _winningTrades = 0;
        _totalTrades = 0;
        PortfolioHistory = new List<T> { options.InitialCapital };
        TradeHistory = new List<T>();
    }

    /// <summary>
    /// Ensures a trading network architecture has valid input/output sizes and default layers.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="expectedInputSize">Expected input size for the agent network.</param>
    /// <param name="expectedOutputSize">Expected output size for the agent network.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers to create if none are provided.</param>
    /// <param name="hiddenLayerSize">Hidden layer width to use when creating defaults.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trading agents need a neural network with the right input and output sizes:
    /// </para>
    /// <para>
    /// - The <b>input size</b> must match the number of features in the market state.
    /// - The <b>output size</b> must match the number of actions the agent can take.
    /// </para>
    /// <para>
    /// If you don't provide custom layers in the architecture, this method fills in a sensible
    /// default multilayer perceptron so the agent can start learning immediately.
    /// </para>
    /// </remarks>
    protected static void EnsureDefaultLayers(
        NeuralNetworkArchitecture<T> architecture,
        int expectedInputSize,
        int expectedOutputSize,
        int hiddenLayerCount = 2,
        int hiddenLayerSize = 64)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        if (architecture.CalculatedInputSize != expectedInputSize)
            throw new ArgumentException($"Architecture input size {architecture.CalculatedInputSize} does not match expected {expectedInputSize}.", nameof(architecture));

        if (architecture.OutputSize != expectedOutputSize)
            throw new ArgumentException($"Architecture output size {architecture.OutputSize} does not match expected {expectedOutputSize}.", nameof(architecture));

        if (architecture.Layers.Count == 0)
        {
            architecture.Layers.AddRange(LayerHelper<T>.CreateDefaultLayers(
                architecture,
                hiddenLayerCount: hiddenLayerCount,
                hiddenLayerSize: hiddenLayerSize,
                outputSize: expectedOutputSize));
        }
    }

    /// <summary>
    /// Creates base RL options from trading options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, CreateBaseOptions builds and wires up model components. This sets up the TradingAgentBase architecture before use.
    /// </para>
    /// </remarks>
    private static ReinforcementLearningOptions<T> CreateBaseOptions(TradingAgentOptions<T> options)
    {
        if (options == null) throw new ArgumentNullException(nameof(options));

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.LossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            TargetUpdateFrequency = options.TargetUpdateFrequency,
            WarmupSteps = options.WarmupSteps,
            EpsilonStart = options.EpsilonStart,
            EpsilonEnd = options.EpsilonEnd,
            EpsilonDecay = options.EpsilonDecay
        };
    }

    #endregion

    #region ITradingAgent Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, SelectTradingAction performs a supporting step in the workflow. It keeps the TradingAgentBase architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public virtual Vector<T> SelectTradingAction(Vector<T> marketState, bool training = true)
    {
        return SelectAction(marketState, training);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, ExecuteTradeWithRiskManagement estimates risk values. This is the core signal the TradingAgentBase architecture focuses on.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ExecuteTradeWithRiskManagement(Vector<T> marketState, T riskBudget)
    {
        var baseAction = SelectTradingAction(marketState, training: false);

        // Apply position size limits based on risk budget
        return ApplyRiskConstraints(baseAction, riskBudget);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, ResetTrading performs a supporting step in the workflow. It keeps the TradingAgentBase architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public virtual void ResetTrading(T initialCapital)
    {
        _initialCapital = initialCapital;
        _portfolioValue = initialCapital;
        _peakValue = initialCapital;
        _winningTrades = 0;
        _totalTrades = 0;
        PortfolioHistory.Clear();
        PortfolioHistory.Add(initialCapital);
        TradeHistory.Clear();
        ResetEpisode();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, GetTradingMetrics calculates evaluation metrics. This summarizes how the TradingAgentBase architecture is performing.
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetTradingMetrics()
    {
        var baseMetrics = GetMetrics();

        // Add financial metrics
        baseMetrics["SharpeRatio"] = SharpeRatio;
        baseMetrics["MaxDrawdown"] = MaxDrawdown;
        baseMetrics["CumulativeReturn"] = CumulativeReturn;
        baseMetrics["WinRate"] = WinRate;
        baseMetrics["TotalTrades"] = NumOps.FromDouble(_totalTrades);
        baseMetrics["PortfolioValue"] = _portfolioValue;
        baseMetrics["InitialCapital"] = _initialCapital;

        return baseMetrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, StoreTradingExperience performs a supporting step in the workflow. It keeps the TradingAgentBase architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public virtual void StoreTradingExperience(Vector<T> marketState, Vector<T> action, T reward,
        Vector<T> nextMarketState, bool done, T pnl)
    {
        // Store in base replay buffer
        StoreExperience(marketState, action, reward, nextMarketState, done);

        // Track trade P&L
        if (!NumOps.Equals(pnl, NumOps.Zero))
        {
            TradeHistory.Add(pnl);
            _totalTrades++;
            if (NumOps.ToDouble(pnl) > 0)
            {
                _winningTrades++;
            }
        }

        // Update portfolio value
        _portfolioValue = NumOps.Add(_portfolioValue, pnl);
        PortfolioHistory.Add(_portfolioValue);

        // Update peak for drawdown calculation
        if (NumOps.ToDouble(_portfolioValue) > NumOps.ToDouble(_peakValue))
        {
            _peakValue = _portfolioValue;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, TrainOnExperiences performs a training step. This updates the TradingAgentBase architecture so it learns from data.
    /// </para>
    /// </remarks>
    public virtual T TrainOnExperiences()
    {
        return Train();
    }

    #endregion

    #region Financial Metrics Calculation

    /// <summary>
    /// Calculates the Sharpe ratio from portfolio history.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sharpe ratio is calculated as:
    /// (Average Return - Risk-Free Rate) / Standard Deviation of Returns
    ///
    /// This measures how much excess return you get per unit of risk.
    /// We annualize assuming 252 trading days per year.
    /// </para>
    /// </remarks>
    protected virtual T CalculateSharpeRatio()
    {
        if (PortfolioHistory.Count < 2) return NumOps.Zero;

        // Calculate returns
        var returns = new List<T>();
        for (int i = 1; i < PortfolioHistory.Count; i++)
        {
            T prevValue = PortfolioHistory[i - 1];
            T currValue = PortfolioHistory[i];
            if (NumOps.ToDouble(prevValue) > 0)
            {
                T ret = NumOps.Divide(NumOps.Subtract(currValue, prevValue), prevValue);
                returns.Add(ret);
            }
        }

        if (returns.Count < 2) return NumOps.Zero;

        // Calculate mean return
        T sumReturns = NumOps.Zero;
        foreach (var r in returns)
        {
            sumReturns = NumOps.Add(sumReturns, r);
        }
        T meanReturn = NumOps.Divide(sumReturns, NumOps.FromDouble(returns.Count));

        // Calculate standard deviation
        T sumSquaredDiff = NumOps.Zero;
        foreach (var r in returns)
        {
            T diff = NumOps.Subtract(r, meanReturn);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(returns.Count - 1));
        T stdDev = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(variance)));

        if (NumOps.ToDouble(stdDev) < 1e-10) return NumOps.Zero;

        // Calculate Sharpe ratio (annualized, assuming risk-free rate = 0)
        T riskFreeRate = NumOps.FromDouble(TradingOptions.RiskFreeRate / 252.0); // Daily rate
        T excessReturn = NumOps.Subtract(meanReturn, riskFreeRate);
        T sharpe = NumOps.Divide(excessReturn, stdDev);

        // Annualize
        return NumOps.Multiply(sharpe, NumOps.FromDouble(Math.Sqrt(252)));
    }

    /// <summary>
    /// Calculates the maximum drawdown from portfolio history.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Maximum drawdown measures the worst loss from a peak.
    /// It's calculated by finding the largest percentage decline from any historical
    /// high point. This tells you the worst-case scenario an investor would have faced.
    /// </para>
    /// </remarks>
    protected virtual T CalculateMaxDrawdown()
    {
        if (PortfolioHistory.Count < 2) return NumOps.Zero;

        T maxDrawdown = NumOps.Zero;
        T peak = PortfolioHistory[0];

        foreach (var value in PortfolioHistory)
        {
            if (NumOps.ToDouble(value) > NumOps.ToDouble(peak))
            {
                peak = value;
            }

            if (NumOps.ToDouble(peak) > 0)
            {
                T drawdown = NumOps.Divide(NumOps.Subtract(peak, value), peak);
                if (NumOps.ToDouble(drawdown) > NumOps.ToDouble(maxDrawdown))
                {
                    maxDrawdown = drawdown;
                }
            }
        }

        return maxDrawdown;
    }

    /// <summary>
    /// Calculates the cumulative return since trading began.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cumulative return is simply:
    /// (Current Portfolio Value - Initial Capital) / Initial Capital
    ///
    /// A return of 0.5 means 50% profit, -0.2 means 20% loss.
    /// </para>
    /// </remarks>
    protected virtual T CalculateCumulativeReturn()
    {
        if (NumOps.ToDouble(_initialCapital) <= 0) return NumOps.Zero;

        return NumOps.Divide(
            NumOps.Subtract(_portfolioValue, _initialCapital),
            _initialCapital);
    }

    /// <summary>
    /// Calculates the win rate (percentage of profitable trades).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Win rate is simply the number of winning trades
    /// divided by total trades. A 60% win rate means 6 out of 10 trades were
    /// profitable. Note that win rate alone doesn't determine profitability -
    /// the size of wins vs losses matters too.
    /// </para>
    /// </remarks>
    protected virtual T CalculateWinRate()
    {
        if (_totalTrades == 0) return NumOps.Zero;

        return NumOps.FromDouble((double)_winningTrades / _totalTrades);
    }

    #endregion

    #region Risk Management

    /// <summary>
    /// Applies risk constraints to a trading action.
    /// </summary>
    /// <param name="action">Raw trading action from the policy.</param>
    /// <param name="riskBudget">Maximum risk allowed.</param>
    /// <returns>Risk-constrained trading action.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method ensures the agent doesn't take on too much risk.
    /// Even if the neural network wants to go 100% long, risk limits might scale this down.
    /// This is crucial in real trading to prevent catastrophic losses.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ApplyRiskConstraints(Vector<T> action, T riskBudget)
    {
        // Default implementation: scale positions if they exceed risk budget
        T maxPositionSize = TradingOptions.MaxPositionSize;
        var constrainedAction = new Vector<T>(action.Length);

        for (int i = 0; i < action.Length; i++)
        {
            T value = action[i];
            T absValue = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(value)));

            if (NumOps.ToDouble(absValue) > NumOps.ToDouble(maxPositionSize))
            {
                // Scale down to max position size
                T sign = NumOps.ToDouble(value) >= 0 ? NumOps.One : NumOps.FromDouble(-1);
                constrainedAction[i] = NumOps.Multiply(sign, maxPositionSize);
            }
            else
            {
                constrainedAction[i] = value;
            }
        }

        // Further scale based on risk budget if needed
        T riskMultiplier = NumOps.ToDouble(riskBudget) < 1.0 ? riskBudget : NumOps.One;
        for (int i = 0; i < constrainedAction.Length; i++)
        {
            constrainedAction[i] = NumOps.Multiply(constrainedAction[i], riskMultiplier);
        }

        return constrainedAction;
    }

    /// <summary>
    /// Calculates reward with optional risk adjustment.
    /// </summary>
    /// <param name="rawReturn">Raw return from the trade.</param>
    /// <returns>Risk-adjusted reward for training.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The reward signal tells the agent how well it did.
    /// Using raw returns can lead to high-risk strategies. Risk-adjusted rewards
    /// penalize variance, encouraging the agent to find stable profitable strategies.
    /// </para>
    /// </remarks>
    protected virtual T CalculateReward(T rawReturn)
    {
        if (!TradingOptions.UseRiskAdjustedReward)
        {
            return rawReturn;
        }

        // Subtract penalty for variance (encourages stable returns)
        // This is a simple approximation; more sophisticated methods exist
        T variancePenalty = NumOps.Multiply(
            NumOps.FromDouble(TradingOptions.VariancePenalty),
            NumOps.Multiply(rawReturn, rawReturn));

        return NumOps.Subtract(rawReturn, variancePenalty);
    }

    #endregion

    #region Override Base Class

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, ResetEpisode performs a supporting step in the workflow. It keeps the TradingAgentBase architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override void ResetEpisode()
    {
        base.ResetEpisode();
        Episodes++;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TradingAgentBase model, GetMetrics calculates evaluation metrics. This summarizes how the TradingAgentBase architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetMetrics()
    {
        return GetTradingMetrics();
    }

    #endregion
}
