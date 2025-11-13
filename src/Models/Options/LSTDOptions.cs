using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class LSTDOptions<T> : ReinforcementLearningOptions<T>
{
    public int FeatureSize { get; init; }
    public int ActionSize { get; init; }
    public double RegularizationParam { get; init; } = 0.01;
}
