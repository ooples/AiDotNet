namespace AiDotNet.Enums;

/// <summary>
/// Which of the two Soft Actor-Critic papers a SAC agent follows.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAC was published twice. The first version keeps a separate network that estimates how good
/// each state is and a fixed exploration strength; the second drops that network and learns the exploration strength
/// as it trains.
/// </para>
/// </remarks>
public enum SacVariant
{
    /// <summary>
    /// Haarnoja et al. 2018a, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    /// Stochastic Actor" (ICML): a state-value network V with a Polyak-averaged target, twin soft Q-networks trained
    /// toward r + γV̄(s'), and a fixed temperature (the paper's reward scale).
    /// </summary>
    ValueNetwork = 0,

    /// <summary>
    /// Haarnoja et al. 2018b, "Soft Actor-Critic Algorithms and Applications": no value network; twin soft Q-networks
    /// with Polyak-averaged targets trained toward r + γ(min Q̄(s', a') − α log π(a'|s')), and the temperature α
    /// learned against a target entropy (§5).
    /// </summary>
    AutomaticTemperature = 1,
}
