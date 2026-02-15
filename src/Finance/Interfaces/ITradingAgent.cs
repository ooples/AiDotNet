using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for financial trading agents using reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// ITradingAgent extends standard RL agent capabilities with financial-specific
/// functionality for trading, portfolio management, and risk assessment.
/// </para>
/// <para><b>For Beginners:</b> Trading agents learn to make buy/sell decisions:
///
/// <b>The Key Insight:</b>
/// Financial markets are complex environments where RL agents can learn optimal
/// trading strategies through trial and error. Trading agents observe market state
/// (prices, volumes, indicators) and learn to take actions (buy, sell, hold) that
/// maximize risk-adjusted returns over time.
///
/// <b>Key Components:</b>
/// - State: Market observations (prices, volumes, technical indicators)
/// - Actions: Trading decisions (buy, sell, hold, position sizes)
/// - Rewards: Returns, risk-adjusted metrics (Sharpe ratio)
/// - Environment: Market simulator or live trading interface
///
/// <b>Financial Metrics:</b>
/// Trading agents track specialized metrics not found in standard RL:
/// - Sharpe Ratio: Risk-adjusted return measure
/// - Maximum Drawdown: Largest peak-to-trough decline
/// - Win Rate: Percentage of profitable trades
/// - Cumulative Return: Total portfolio growth
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TradingAgent")]
public interface ITradingAgent<T>
{
    /// <summary>
    /// Gets the current Sharpe ratio of the trading strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sharpe ratio measures risk-adjusted returns.
    /// It's calculated as (Average Return - Risk-Free Rate) / Standard Deviation of Returns.
    /// A higher Sharpe ratio indicates better risk-adjusted performance.
    /// Typical benchmarks: &gt;1 is good, &gt;2 is very good, &gt;3 is excellent.
    /// </para>
    /// </remarks>
    T SharpeRatio { get; }

    /// <summary>
    /// Gets the maximum drawdown experienced during trading.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Maximum drawdown is the largest peak-to-trough decline
    /// in portfolio value before a new peak is reached. It measures the worst-case
    /// loss an investor would have experienced. Lower is better.
    /// Example: A 20% max drawdown means the portfolio once fell 20% from its peak.
    /// </para>
    /// </remarks>
    T MaxDrawdown { get; }

    /// <summary>
    /// Gets the cumulative return since trading began.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cumulative return is the total percentage gain or loss
    /// from the initial investment. A value of 0.5 means 50% gain, -0.2 means 20% loss.
    /// Unlike Sharpe ratio, this doesn't account for risk.
    /// </para>
    /// </remarks>
    T CumulativeReturn { get; }

    /// <summary>
    /// Gets the win rate (percentage of profitable trades).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Win rate is the percentage of trades that were profitable.
    /// A win rate of 0.55 means 55% of trades made money. Note that a high win rate
    /// doesn't guarantee profitability - the sizes of wins vs losses matter too.
    /// </para>
    /// </remarks>
    T WinRate { get; }

    /// <summary>
    /// Gets the total number of trades executed.
    /// </summary>
    int TotalTrades { get; }

    /// <summary>
    /// Gets the current portfolio value.
    /// </summary>
    T PortfolioValue { get; }

    /// <summary>
    /// Gets the initial capital when trading started.
    /// </summary>
    T InitialCapital { get; }

    /// <summary>
    /// Selects a trading action given the current market state.
    /// </summary>
    /// <param name="marketState">Current market observation (prices, volumes, indicators).</param>
    /// <param name="training">Whether the agent is in training mode (affects exploration).</param>
    /// <returns>Trading action vector (positions or discrete actions).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the core decision-making function. Given the current
    /// market state (prices, technical indicators, etc.), the agent decides what to do:
    /// - For discrete actions: Buy, Sell, or Hold
    /// - For continuous actions: Target portfolio weights or position sizes
    /// During training, the agent may explore suboptimal actions to learn.
    /// </para>
    /// </remarks>
    Vector<T> SelectTradingAction(Vector<T> marketState, bool training = true);

    /// <summary>
    /// Executes a trade with risk management constraints.
    /// </summary>
    /// <param name="marketState">Current market observation.</param>
    /// <param name="riskBudget">Maximum risk allowed for this trade (e.g., VaR limit).</param>
    /// <returns>Risk-adjusted trading action.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adds risk management to trading decisions.
    /// The agent first decides what it wants to do, then adjusts based on risk limits.
    /// For example, if the agent wants to go 100% long but the risk budget only allows
    /// 50% exposure, the position is scaled down.
    /// </para>
    /// </remarks>
    Vector<T> ExecuteTradeWithRiskManagement(Vector<T> marketState, T riskBudget);

    /// <summary>
    /// Resets the trading episode and portfolio state.
    /// </summary>
    /// <param name="initialCapital">Starting capital for the new episode.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Resets the trading simulation to start fresh.
    /// This is called at the beginning of each training episode or when
    /// backtesting a new time period. The portfolio is reset to the initial capital.
    /// </para>
    /// </remarks>
    void ResetTrading(T initialCapital);

    /// <summary>
    /// Gets comprehensive financial metrics for the trading strategy.
    /// </summary>
    /// <returns>Dictionary of metric names to values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns all relevant trading performance metrics
    /// in a single dictionary. This includes standard RL metrics (training loss,
    /// episode count) plus financial metrics (Sharpe ratio, drawdown, returns).
    /// Useful for logging, visualization, and strategy evaluation.
    /// </para>
    /// </remarks>
    Dictionary<string, T> GetTradingMetrics();

    /// <summary>
    /// Stores a trading experience for learning.
    /// </summary>
    /// <param name="marketState">Market state before action.</param>
    /// <param name="action">Action taken.</param>
    /// <param name="reward">Reward received (typically return or risk-adjusted return).</param>
    /// <param name="nextMarketState">Market state after action.</param>
    /// <param name="done">Whether the episode ended.</param>
    /// <param name="pnl">Profit/loss from this trade.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Records one step of trading experience for learning.
    /// The agent remembers what the market looked like, what action it took,
    /// and what reward it received. This data is used later to train the agent.
    /// The PnL (profit/loss) is stored separately for financial analysis.
    /// </para>
    /// </remarks>
    void StoreTradingExperience(Vector<T> marketState, Vector<T> action, T reward,
        Vector<T> nextMarketState, bool done, T pnl);

    /// <summary>
    /// Performs one training step using stored experiences.
    /// </summary>
    /// <returns>Training loss value.</returns>
    T TrainOnExperiences();

    /// <summary>
    /// Gets the agent's model metadata with financial context.
    /// </summary>
    /// <returns>Model metadata including trading-specific information.</returns>
    ModelMetadata<T> GetModelMetadata();
}
