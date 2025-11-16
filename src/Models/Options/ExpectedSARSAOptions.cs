using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Expected SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExpectedSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Gets or initializes the size of the state space.
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Gets or initializes the size of the action space.
    /// </summary>
    public int ActionSize { get; init; }
}
