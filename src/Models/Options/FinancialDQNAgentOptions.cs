namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial DQN (Deep Q-Network) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DQN is a value-based RL algorithm suited for discrete
/// action spaces (e.g., buy/hold/sell). These options extend the base trading
/// agent options with DQN-specific parameters.
/// </para>
/// </remarks>
public class FinancialDQNAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>
    /// Whether to use double DQN for more stable Q-value estimates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Double DQN reduces overestimation of Q-values,
    /// leading to more stable and reliable trading decisions.
    /// </para>
    /// </remarks>
    public bool UseDoubleDQN { get; set; } = true;

    /// <summary>
    /// Whether to use dueling network architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dueling DQN separates state value and action advantage,
    /// which can improve learning in states where actions don't matter much.
    /// </para>
    /// </remarks>
    public bool UseDuelingNetwork { get; set; } = false;
}
