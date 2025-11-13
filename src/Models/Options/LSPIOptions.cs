using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class LSPIOptions<T> : ReinforcementLearningOptions<T>
{
    public int FeatureSize { get; init; }
    public int ActionSize { get; init; }
    public double RegularizationParam { get; init; } = 0.01;
    public int MaxIterations { get; init; } = 20;
    public double ConvergenceThreshold { get; init; } = 0.01;
}
