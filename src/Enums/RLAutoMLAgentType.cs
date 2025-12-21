namespace AiDotNet.Enums;

/// <summary>
/// Defines which reinforcement learning agent families can be explored by AutoML.
/// </summary>
/// <remarks>
/// <para>
/// This enum is used by facade configuration options to select which RL agent types AutoML is allowed to try.
/// </para>
/// <para><b>For Beginners:</b> Different RL agents are better suited for different problems:
/// <list type="bullet">
/// <item><description><see cref="DQN"/> is popular for discrete action spaces (like left/right).</description></item>
/// <item><description><see cref="PPO"/> is a strong general-purpose agent (discrete or continuous).</description></item>
/// <item><description><see cref="A2C"/> is a simple actor-critic baseline.</description></item>
/// <item><description><see cref="DDPG"/> and <see cref="SAC"/> are commonly used for continuous control.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum RLAutoMLAgentType
{
    /// <summary>
    /// Deep Q-Network (DQN) for discrete action spaces.
    /// </summary>
    DQN,

    /// <summary>
    /// Proximal Policy Optimization (PPO) for discrete or continuous control.
    /// </summary>
    PPO,

    /// <summary>
    /// Advantage Actor-Critic (A2C) for discrete or continuous control.
    /// </summary>
    A2C,

    /// <summary>
    /// Deep Deterministic Policy Gradient (DDPG) for continuous control.
    /// </summary>
    DDPG,

    /// <summary>
    /// Soft Actor-Critic (SAC) for continuous control.
    /// </summary>
    SAC
}

