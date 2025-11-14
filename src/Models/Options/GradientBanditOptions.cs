using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class GradientBanditOptions<T> : ReinforcementLearningOptions<T>
{
    public int NumArms { get; init; }
    public double Alpha { get; init; } = 0.1;  // Step size
    public bool UseBaseline { get; init; } = true;
}
