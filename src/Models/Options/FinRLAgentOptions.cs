namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the FinRL unified trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FinRL is a unified framework that can switch between
/// multiple RL algorithms (DQN, PPO, A2C, SAC). These options extend the base
/// trading agent options with algorithm selection and unified configuration.
/// </para>
/// </remarks>
public class FinRLAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>
    /// Whether to automatically select the best algorithm based on the action space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If true, the agent will pick DQN for discrete actions
    /// and SAC for continuous actions automatically.
    /// </para>
    /// </remarks>
    public bool AutoSelectAlgorithm { get; set; } = false;

    /// <summary>
    /// Whether to use ensemble of multiple algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble mode trains multiple agents with different
    /// algorithms and combines their decisions, which can improve robustness.
    /// </para>
    /// </remarks>
    public bool UseEnsemble { get; set; } = false;
}
