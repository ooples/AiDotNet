using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Validation;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dyna-Q reinforcement learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dyna-Q combines model-free Q-learning with a learned environment model to perform
/// additional planning steps, accelerating learning from limited real experience.
/// </para>
/// <para><b>For Beginners:</b>
/// Dyna-Q is like an agent that not only learns from real experience but also
/// "imagines" additional experiences using a model of the environment. The planning
/// steps parameter controls how many imagined experiences the agent uses per real step.
/// </para>
/// </remarks>
public class DynaQOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DynaQOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying values from another instance.
    /// </summary>
    /// <param name="other">The options to copy from.</param>
    public DynaQOptions(DynaQOptions<T> other)
    {
        Guard.NotNull(other);

        StateSize = other.StateSize;
        ActionSize = other.ActionSize;
        PlanningSteps = other.PlanningSteps;

        // Copy base class properties
        LearningRate = other.LearningRate;
        DiscountFactor = other.DiscountFactor;
        LossFunction = other.LossFunction;
        BatchSize = other.BatchSize;
        ReplayBufferSize = other.ReplayBufferSize;
        TargetUpdateFrequency = other.TargetUpdateFrequency;
        UsePrioritizedReplay = other.UsePrioritizedReplay;
        EpsilonStart = other.EpsilonStart;
        EpsilonEnd = other.EpsilonEnd;
    }

    /// <summary>
    /// Dimension of the environment state vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of input features the agent receives each step.</para>
    /// </remarks>
    public int StateSize { get; init; } = 16;

    /// <summary>
    /// Number of discrete actions available to the agent.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of different actions the agent can choose from.</para>
    /// </remarks>
    public int ActionSize { get; init; } = 4;

    /// <summary>
    /// Number of planning steps per real environment step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many "imagined" experiences the agent generates
    /// from its learned model per real interaction. Higher values accelerate learning
    /// but increase computation per step.</para>
    /// </remarks>
    public int PlanningSteps { get; init; } = 50;
}
