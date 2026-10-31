using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Policy Iteration agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PolicyIterationOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Maximum number of sweeps of each policy evaluation.</summary>
    public int MaxEvaluationIterations { get; init; } = 100;
    /// <summary>Convergence threshold: policy evaluation stops once the largest value change in a sweep falls below it.</summary>
    public double Theta { get; init; } = 1e-6;  // Convergence threshold
}
