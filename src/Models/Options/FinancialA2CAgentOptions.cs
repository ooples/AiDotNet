namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial A2C (Advantage Actor-Critic) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A2C is a fast actor-critic RL algorithm that trains
/// synchronously. These options extend the base trading agent options with
/// A2C-specific parameters.
/// </para>
/// </remarks>
public class FinancialA2CAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>
    /// Number of parallel environments for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A2C benefits from running multiple simulations at once.
    /// More environments = faster training but more memory usage.
    /// </para>
    /// </remarks>
    public int NumEnvironments { get; set; } = 4;

    /// <summary>
    /// Number of steps between updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many actions the agent takes before updating
    /// its policy. Shorter intervals mean more frequent learning.
    /// </para>
    /// </remarks>
    public int NSteps { get; set; } = 5;
}
