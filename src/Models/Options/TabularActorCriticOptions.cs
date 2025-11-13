using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class TabularActorCriticOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public double ActorLearningRate { get; init; } = 0.01;
    public double CriticLearningRate { get; init; } = 0.1;
}
