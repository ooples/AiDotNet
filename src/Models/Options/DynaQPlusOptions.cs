using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for the Dyna-Q+ agent, which plans with a learned model and rewards revisiting long-untried actions.</summary>
public class DynaQPlusOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Number of simulated planning updates from the learned model per real step.</summary>
    public int PlanningSteps { get; init; } = 50;
    /// <summary>Weight of the exploration bonus kappa * sqrt(tau) given to state-action pairs not tried for tau steps (Sutton and Barto, Dyna-Q+).</summary>
    public double Kappa { get; init; } = 0.001;  // Bonus for exploration
}
