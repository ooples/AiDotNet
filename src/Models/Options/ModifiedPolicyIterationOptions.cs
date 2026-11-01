using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Modified Policy Iteration agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ModifiedPolicyIterationOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Maximum number of policy-evaluation sweeps between policy improvements (the partial evaluation of modified policy iteration).</summary>
    public int MaxEvaluationSweeps { get; init; } = 10;  // Limited evaluation sweeps
    /// <summary>Convergence threshold: evaluation stops once the largest value change in a sweep falls below it.</summary>
    public double Theta { get; init; } = 1e-6;
}
