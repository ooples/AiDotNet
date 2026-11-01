using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for N-step SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NStepSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; } = 16;
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; } = 4;
    /// <summary>Number of rewards (n) accumulated before bootstrapping from the Q-value estimate.</summary>
    public int NSteps { get; init; } = 3;
}
