using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class NStepQLearningOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public int NSteps { get; init; } = 3;
}
