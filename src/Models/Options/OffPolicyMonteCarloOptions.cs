using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Off-Policy Monte Carlo Control agents with importance sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OffPolicyMonteCarloOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public double BehaviorEpsilon { get; init; } = 0.3;
}
