using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for financial trading agents.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TradingAgentOptions extends standard RL options with financial-specific parameters
/// for trading environments, risk management, and reward calculation.
/// </para>
/// <para><b>For Beginners:</b> These options control how the trading agent behaves:
///
/// <b>RL Parameters:</b>
/// - LearningRate: How fast the agent learns (higher = faster but unstable)
/// - DiscountFactor: How much to value future rewards (higher = more patient)
/// - BatchSize: How many experiences to learn from at once
/// - ReplayBufferSize: How many past experiences to remember
///
/// <b>Trading Parameters:</b>
/// - InitialCapital: Starting money for the portfolio
/// - TransactionCost: Cost per trade (brokers charge fees)
/// - MaxPositionSize: Maximum allowed position (prevents over-concentration)
/// - RiskFreeRate: Benchmark return (usually government bond rate)
///
/// <b>Risk Management:</b>
/// - UseRiskAdjustedReward: Whether to penalize high variance
/// - VariancePenalty: How much to penalize risky behavior
/// </para>
/// </remarks>
public class TradingAgentOptions<T> : ModelOptions
{
    #region RL Parameters

    /// <summary>
    /// Learning rate for gradient updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how much the agent's neural network changes
    /// with each training step. Too high = unstable learning, too low = slow learning.
    /// Typical values: 1e-4 to 1e-3.
    /// </para>
    /// </remarks>
    public T LearningRate { get; set; } = default!;

    /// <summary>
    /// Discount factor (gamma) for future rewards.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much to value future rewards vs immediate rewards.
    /// 0.99 means the agent cares a lot about long-term returns.
    /// 0.9 means more focus on short-term profits.
    /// </para>
    /// </remarks>
    public T DiscountFactor { get; set; } = default!;

    /// <summary>
    /// Loss function for training the agent's neural networks.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public new int? Seed { get; set; }

    /// <summary>
    /// Batch size for training updates.
    /// </summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Size of the experience replay buffer.
    /// </summary>
    public int ReplayBufferSize { get; set; } = 100000;

    /// <summary>
    /// How often to update the target network.
    /// </summary>
    public int TargetUpdateFrequency { get; set; } = 1000;

    /// <summary>
    /// Number of steps before training begins.
    /// </summary>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>
    /// Initial exploration rate.
    /// </summary>
    public double EpsilonStart { get; set; } = 1.0;

    /// <summary>
    /// Final exploration rate.
    /// </summary>
    public double EpsilonEnd { get; set; } = 0.01;

    /// <summary>
    /// Exploration decay rate.
    /// </summary>
    public double EpsilonDecay { get; set; } = 0.995;

    #endregion

    #region State/Action Space

    /// <summary>
    /// Dimension of the market state observation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many features the agent observes.
    /// This might include prices, volumes, technical indicators, etc.
    /// More features can help but also make learning harder.
    /// </para>
    /// </remarks>
    public int StateSize { get; set; } = 64;

    /// <summary>
    /// Number of possible actions or continuous action dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For discrete actions: number of choices (e.g., 3 for buy/hold/sell).
    /// For continuous actions: number of assets or position dimensions.
    /// </para>
    /// </remarks>
    public int ActionSize { get; set; } = 3;

    /// <summary>
    /// Whether the action space is continuous.
    /// </summary>
    public bool ContinuousActions { get; set; } = false;

    /// <summary>
    /// Hidden layer sizes for the neural network.
    /// </summary>
    public int[] HiddenLayers { get; set; } = new[] { 256, 128, 64 };

    #endregion

    #region Trading Parameters

    /// <summary>
    /// Initial capital for the trading portfolio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Starting money for the simulation.
    /// Results are typically reported as percentages so the absolute
    /// value doesn't matter much, but it affects numerical stability.
    /// Common values: 10000, 100000, 1000000.
    /// </para>
    /// </remarks>
    public T InitialCapital { get; set; } = default!;

    /// <summary>
    /// Transaction cost as a fraction of trade value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cost to execute a trade (broker fees, slippage).
    /// 0.001 means 0.1% of the trade value is lost to costs.
    /// This discourages excessive trading.
    /// </para>
    /// </remarks>
    public double TransactionCost { get; set; } = 0.001;

    /// <summary>
    /// Maximum position size as a fraction of portfolio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Maximum allowed position in any asset.
    /// 1.0 means up to 100% of portfolio in one position.
    /// 0.5 means maximum 50% exposure to any single asset.
    /// Lower values enforce diversification.
    /// </para>
    /// </remarks>
    public T MaxPositionSize { get; set; } = default!;

    /// <summary>
    /// Annual risk-free rate for Sharpe ratio calculation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The return you could get with no risk (e.g., government bonds).
    /// Used as benchmark for risk-adjusted returns.
    /// Typical values: 0.01 to 0.05 (1% to 5% annually).
    /// </para>
    /// </remarks>
    public double RiskFreeRate { get; set; } = 0.02;

    /// <summary>
    /// Whether to allow short selling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Short selling means betting on prices going down.
    /// If true, the agent can take negative positions (profit when prices fall).
    /// If false, positions must be >= 0.
    /// </para>
    /// </remarks>
    public bool AllowShortSelling { get; set; } = true;

    #endregion

    #region Reward Configuration

    /// <summary>
    /// Whether to use risk-adjusted rewards for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If true, the reward signal penalizes variance.
    /// This encourages the agent to find stable strategies rather than
    /// high-risk/high-reward approaches. Recommended for real trading.
    /// </para>
    /// </remarks>
    public bool UseRiskAdjustedReward { get; set; } = true;

    /// <summary>
    /// Penalty coefficient for return variance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How much to penalize risky behavior.
    /// Higher values make the agent more risk-averse.
    /// Typical values: 0.1 to 1.0.
    /// </para>
    /// </remarks>
    public double VariancePenalty { get; set; } = 0.5;

    /// <summary>
    /// Reward scaling factor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiplier for all rewards.
    /// Helps with numerical stability during training.
    /// If rewards are very small, increase this; if very large, decrease.
    /// </para>
    /// </remarks>
    public double RewardScale { get; set; } = 1.0;

    #endregion

    #region Algorithm-Specific

    /// <summary>
    /// Clip range for PPO algorithm.
    /// </summary>
    public double PPOClipRange { get; set; } = 0.2;

    /// <summary>
    /// Entropy coefficient for exploration.
    /// </summary>
    public double EntropyCoefficient { get; set; } = 0.01;

    /// <summary>
    /// Value function coefficient for actor-critic methods.
    /// </summary>
    public double ValueCoefficient { get; set; } = 0.5;

    /// <summary>
    /// GAE lambda for advantage estimation.
    /// </summary>
    public double GAELambda { get; set; } = 0.95;

    /// <summary>
    /// Temperature parameter for SAC algorithm.
    /// </summary>
    public double SACAlpha { get; set; } = 0.2;

    /// <summary>
    /// Whether to automatically tune SAC temperature.
    /// </summary>
    public bool AutoTuneAlpha { get; set; } = true;

    /// <summary>
    /// Soft update coefficient for target networks.
    /// </summary>
    public double Tau { get; set; } = 0.005;

    #endregion

    /// <summary>
    /// Validates the options and throws if invalid.
    /// </summary>
    public virtual void Validate()
    {
        if (StateSize <= 0)
            throw new ArgumentException("StateSize must be positive.", nameof(StateSize));
        if (ActionSize <= 0)
            throw new ArgumentException("ActionSize must be positive.", nameof(ActionSize));
        if (BatchSize <= 0)
            throw new ArgumentException("BatchSize must be positive.", nameof(BatchSize));
        if (ReplayBufferSize <= 0)
            throw new ArgumentException("ReplayBufferSize must be positive.", nameof(ReplayBufferSize));
        if (EpsilonStart < EpsilonEnd)
            throw new ArgumentException("EpsilonStart must be >= EpsilonEnd.", nameof(EpsilonStart));
        if (TransactionCost < 0)
            throw new ArgumentException("TransactionCost cannot be negative.", nameof(TransactionCost));
    }
}
