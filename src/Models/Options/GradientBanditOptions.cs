using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for the gradient bandit, which learns action preferences with a softmax policy.</summary>
public class GradientBanditOptions<T> : ReinforcementLearningOptions<T>
{
    // A multi-armed bandit needs arms: with the previous implicit default of 0 the preference vector was
    // empty, so the agent exposed no parameters and training could not change anything. Default to a usable
    // 10-arm bandit (overridable); callers with a known action space still set this explicitly.
    /// <summary>Number of arms (actions) the bandit chooses among.</summary>
    public int NumArms { get; init; } = 10;
    /// <summary>Step size for updating the action preferences.</summary>
    public double Alpha { get; init; } = 0.1;  // Step size
    /// <summary>Whether to subtract the average reward as a baseline from each reward before updating preferences.</summary>
    public bool UseBaseline { get; init; } = true;
}
