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
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public int MaxEvaluationSweeps { get; init; } = 10;  // Limited evaluation sweeps
    public double Theta { get; init; } = 1e-6;
}
