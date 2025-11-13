using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class QLambdaOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public double EpsilonStart { get; init; } = 1.0;
    public double EpsilonEnd { get; init; } = 0.01;
    public double EpsilonDecay { get; init; } = 0.995;
    public double Lambda { get; init; } = 0.9;
}
