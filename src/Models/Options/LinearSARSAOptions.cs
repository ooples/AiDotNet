using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class LinearSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    public int FeatureSize { get; init; }
    public int ActionSize { get; init; }
    public double EpsilonStart { get; init; } = 1.0;
    public double EpsilonEnd { get; init; } = 0.01;
    public double EpsilonDecay { get; init; } = 0.995;
}
