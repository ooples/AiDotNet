using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class UCBBanditOptions<T> : ReinforcementLearningOptions<T>
{
    public int NumArms { get; init; }
    public double ExplorationParameter { get; init; } = 2.0;  // c parameter in UCB
}
