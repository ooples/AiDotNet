using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for N-step SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NStepSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; } = 16;
    public int ActionSize { get; init; } = 4;
    public int NSteps { get; init; } = 3;
}
