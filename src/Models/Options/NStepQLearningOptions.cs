using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class NStepQLearningOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public int NSteps { get; init; } = 3;
    public double EpsilonStart { get; init; } = 1.0;
    public double EpsilonEnd { get; init; } = 0.01;
    public double EpsilonDecay { get; init; } = 0.995;
}
