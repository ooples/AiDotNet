using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for On-Policy Monte Carlo Control agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OnPolicyMonteCarloOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public double EpsilonStart { get; init; } = 1.0;
    public double EpsilonEnd { get; init; } = 0.01;
    public double EpsilonDecay { get; init; } = 0.995;
}
