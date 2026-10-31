using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TD3 agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TD3Options<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Learning rate of the actor network.</summary>
    public T ActorLearningRate { get; init; }
    /// <summary>Learning rate of the critic network.</summary>
    public T CriticLearningRate { get; init; }
    /// <summary>Soft-update coefficient for the target networks: each update moves them this fraction of the way toward the online networks.</summary>
    public T TargetUpdateTau { get; init; }
    /// <summary>Loss function used to train the critic network.</summary>
    public ILossFunction<T> CriticLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Number of critic updates per actor and target update (delayed policy updates).</summary>
    public int PolicyUpdateFrequency { get; init; } = 2;
    /// <summary>Standard deviation of the Gaussian noise added to actions for exploration.</summary>
    public double ExplorationNoise { get; init; } = 0.1;
    /// <summary>Standard deviation of the noise added to target actions (target policy smoothing).</summary>
    public double TargetPolicyNoise { get; init; } = 0.2;
    /// <summary>Clip range of the noise added to target actions (target policy smoothing).</summary>
    public double TargetNoiseClip { get; init; } = 0.5;
    /// <summary>Hidden layer sizes of the actor network.</summary>
    public List<int> ActorHiddenLayers { get; init; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the critic network.</summary>
    public List<int> CriticHiddenLayers { get; init; } = new List<int> { 256, 256 };
    /// <summary>Number of environment steps taken with random actions before learning starts.</summary>
    public new int WarmupSteps { get; init; } = 25000;

    /// <summary>Creates options with the default values.</summary>
    public TD3Options()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.001);
        CriticLearningRate = numOps.FromDouble(0.001);
        TargetUpdateTau = numOps.FromDouble(0.005);
    }
}
