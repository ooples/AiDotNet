using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class EpsilonGreedyBanditOptions<T> : ReinforcementLearningOptions<T>
{
    public int NumArms { get; init; }
    public double Epsilon { get; init; } = 0.1;
}
