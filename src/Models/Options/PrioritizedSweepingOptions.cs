using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for prioritized sweeping, which plans first where the value estimates would change most.</summary>
public class PrioritizedSweepingOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Number of queued planning updates per real step.</summary>
    public int PlanningSteps { get; init; } = 50;
    /// <summary>Minimum priority (absolute TD error) for a state-action pair to be queued for a planning update.</summary>
    public double PriorityThreshold { get; init; } = 0.01;
}
