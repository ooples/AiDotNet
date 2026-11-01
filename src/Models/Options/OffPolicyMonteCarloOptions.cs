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
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Exploration rate of the epsilon-greedy behavior policy that generates episodes; returns are reweighted by importance sampling toward the greedy target policy.</summary>
    public double BehaviorEpsilon { get; init; } = 0.3;
}
