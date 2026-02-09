namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial SAC (Soft Actor-Critic) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAC is an off-policy actor-critic algorithm that maximizes
/// both reward and entropy (exploration). It is well-suited for continuous action
/// spaces like portfolio weight allocation. These options extend the base trading
/// agent options with SAC-specific parameters.
/// </para>
/// </remarks>
public class FinancialSACAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>
    /// Initial log alpha value for automatic temperature tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much the agent explores vs exploits.
    /// When auto-tuning is enabled, this is just the starting point.
    /// </para>
    /// </remarks>
    public double InitialLogAlpha { get; set; } = 0.0;

    /// <summary>
    /// Target entropy ratio relative to action dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets the target randomness level as a fraction of
    /// the maximum possible entropy. Values closer to 1.0 encourage more exploration.
    /// </para>
    /// </remarks>
    public double TargetEntropyRatio { get; set; } = -1.0;
}
