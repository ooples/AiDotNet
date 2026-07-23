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
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public int MaxEvaluationIterations { get; init; } = 100;
    public double Theta { get; init; } = 1e-6;  // Convergence threshold
}
