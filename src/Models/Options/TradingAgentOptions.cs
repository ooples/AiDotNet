using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

#pragma warning disable CS8618 // Generic T properties use default(T) - always used with value types
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
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public TradingAgentOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.0003);
        DiscountFactor = numOps.FromDouble(0.99);
        InitialCapital = numOps.FromDouble(100000.0);
        MaxPositionSize = numOps.FromDouble(1.0);
    }

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
    public required T LearningRate { get; set; }

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
    public required T DiscountFactor { get; set; }

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
    /// <value>64 transitions by default; must be positive.</value>
    /// <remarks><para>For A2C this is the minimum current-rollout size, not a random replay sample count.
    /// An eligible update consumes the complete bounded rollout in one actor/critic step.
    /// Configure this no larger than <see cref="ReplayBufferSize"/> for autonomous replay/rollout
    /// training; a larger minimum can never be reached by the bounded buffer. Explicit one-shot
    /// supervised updates bypass this readiness requirement.</para>
    /// <para><b>For Beginners:</b> This is how many observations the agent waits to collect for a normal
    /// update. A2C then learns once from all observations in its current rollout.</para></remarks>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Size of the experience replay buffer.
    /// </summary>
    /// <value>100000 transitions by default; must be positive.</value>
    /// <remarks><para>For on-policy A2C this bounds pending current-policy transitions only; consumed
    /// transitions are discarded, and the oldest pending transition is dropped at capacity.</para>
    /// <para><b>For Beginners:</b> This limits how many observations an agent retains, so memory use
    /// remains bounded. A2C does not reuse observations after changing its policy.</para></remarks>
    public int ReplayBufferSize { get; set; } = 100000;

    /// <summary>
    /// How often to update the target network.
    /// </summary>
    public int TargetUpdateFrequency { get; set; } = 1000;

    /// <summary>
    /// Number of environment steps (stored transitions) before training begins.
    /// </summary>
    /// <value>1000 transitions by default; must be non-negative.</value>
    /// <remarks>
    /// <para>
    /// Replay-based agents (DQN, SAC, market-making) apply no gradient update until their replay buffer
    /// holds at least this many transitions (capped at <see cref="ReplayBufferSize"/>), in addition to
    /// needing one full <see cref="BatchSize"/>. During warmup they still act with their initial exploration
    /// (for DQN, epsilon stays at <see cref="EpsilonStart"/> because it only decays per update), so the
    /// first updates sample from a diverse buffer. A one-shot supervised <c>Train(state, target)</c> call is
    /// not gated. Set 0 to start updating as soon as a minibatch is available.
    /// </para>
    /// <para>
    /// A2C applies this threshold only to its initial current-policy rollout, capped at
    /// <see cref="ReplayBufferSize"/>. Later updates require a fresh <see cref="BatchSize"/> without
    /// repeating initial warmup. Each update consumes that rollout once, even if the update fails.
    /// </para>
    /// <para><b>For Beginners:</b> Warmup lets an agent collect observations before its first lesson.
    /// Zero removes this initial wait, but does not remove the normal batch-size requirement.</para>
    /// <para>
    /// The on-policy PPO agent learns only from the rollout it just collected and has no replay buffer to
    /// warm up, so it does not use this option.
    /// </para>
    /// </remarks>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>
    /// Initial exploration rate for epsilon-greedy agents (the financial DQN agent).
    /// </summary>
    /// <value>1.0 by default; a finite probability in [0, 1], at least <see cref="EpsilonEnd"/>.</value>
    /// <remarks>
    /// <para>
    /// The probability of taking a uniformly random action before any gradient update has been applied.
    /// Epsilon then follows <c>max(EpsilonEnd, EpsilonStart * EpsilonDecay^updates)</c>.
    /// </para>
    /// <para><b>For Beginners:</b> At 1.0 every exploratory decision is random initially;
    /// at 0.0 the agent always chooses its currently preferred action.</para>
    /// </remarks>
    public double EpsilonStart { get; set; } = 1.0;

    /// <summary>
    /// Final (floor) exploration rate for epsilon-greedy agents; epsilon never decays below this value.
    /// </summary>
    /// <value>0.01 by default; a finite probability in [0, 1], no greater than <see cref="EpsilonStart"/>.</value>
    /// <remarks><para><b>For Beginners:</b> The default keeps a 1% chance of exploring even after
    /// training has reduced the initial randomness. It is not a percentage-valued setting: use
    /// 0.01, not 1, to request one percent.</para></remarks>
    public double EpsilonEnd { get; set; } = 0.01;

    /// <summary>
    /// Multiplicative exploration decay applied once per gradient update, in (0, 1].
    /// </summary>
    /// <value>0.995 by default; a finite factor in (0, 1].</value>
    /// <remarks>
    /// <para>
    /// After <c>k</c> gradient updates epsilon is <c>max(EpsilonEnd, EpsilonStart * EpsilonDecay^k)</c> — the
    /// same per-update schedule the library's <c>DQNAgent</c> uses. With the defaults (1.0, 0.01, 0.995)
    /// epsilon reaches its floor after about 920 updates. 1.0 disables decay.
    /// </para>
    /// <para><b>For Beginners:</b> Each successful update multiplies the random-action probability by
    /// this factor. Values closer to 1 reduce exploration more slowly; 1 leaves it unchanged.</para>
    /// </remarks>
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
    /// Hidden layer sizes for the networks an agent builds itself.
    /// </summary>
    /// <value>Two layers of 64 neurons: <c>[64, 64]</c>. Empty is valid; every supplied width must be positive.</value>
    /// <remarks>
    /// <para>
    /// When an actor / critic / Q-network architecture is passed with no layers, the agent builds a
    /// multilayer perceptron with one hidden layer per entry (ReLU; Tanh for PPO) and a linear output layer.
    /// Architectures that already contain layers are used as-is and this option does not change them.
    /// An empty array builds a single linear layer. Every entry must be positive.
    /// </para>
    /// <para>
    /// The default <c>[64, 64]</c> is the network the agents have always built by default; before this option
    /// was honoured its documented default of <c>[256, 128, 64]</c> was never applied.
    /// This is an AiDotNet compatibility default, not a universal architecture prescribed by the
    /// DQN, asynchronous actor-critic, SAC or market-making papers. PPO's reported MuJoCo MLP used
    /// 64-by-64 Tanh layers, whereas SAC's original experiments used 256-by-256 ReLU layers.
    /// Select task-appropriate widths explicitly when reproducing a particular experiment.
    /// </para>
    /// <para><b>For Beginners:</b> Each number is one hidden layer's neuron count. For example,
    /// <c>[128, 64]</c> builds a wider first layer and a smaller second layer; it does not modify
    /// layers that you supplied yourself.</para>
    /// </remarks>
    /// <seealso href="https://arxiv.org/pdf/1707.06347">PPO, section 6.1 (MuJoCo policy architecture).</seealso>
    /// <seealso href="https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf">SAC, appendix D, table 1.</seealso>
    public int[] HiddenLayers { get; set; } = new[] { 64, 64 };

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
    public required T InitialCapital { get; set; }

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
    public required T MaxPositionSize { get; set; }

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
        if (!(EpsilonStart >= 0.0 && EpsilonStart <= 1.0))
            throw new ArgumentException("EpsilonStart must be a finite probability in [0, 1].", nameof(EpsilonStart));
        if (!(EpsilonEnd >= 0.0 && EpsilonEnd <= 1.0))
            throw new ArgumentException("EpsilonEnd must be a finite probability in [0, 1].", nameof(EpsilonEnd));
        if (EpsilonStart < EpsilonEnd)
            throw new ArgumentException("EpsilonStart must be >= EpsilonEnd.", nameof(EpsilonStart));
        if (!(EpsilonDecay > 0.0 && EpsilonDecay <= 1.0))
            throw new ArgumentException("EpsilonDecay must be in (0, 1].", nameof(EpsilonDecay));
        if (WarmupSteps < 0)
            throw new ArgumentException("WarmupSteps cannot be negative.", nameof(WarmupSteps));
        if (HiddenLayers is null || Array.Exists(HiddenLayers, size => size <= 0))
            throw new ArgumentException("HiddenLayers must be non-null with positive widths.", nameof(HiddenLayers));
        if (TransactionCost < 0)
            throw new ArgumentException("TransactionCost cannot be negative.", nameof(TransactionCost));
    }
}
