using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DDPG agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DDPGOptions<T> : ModelOptions
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Learning rate of the actor network.</summary>
    public T ActorLearningRate { get; set; }
    /// <summary>Learning rate of the critic network.</summary>
    public T CriticLearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Soft-update coefficient for the target networks: each update moves them this fraction of the way toward the online networks.</summary>
    public T TargetUpdateTau { get; set; }
    /// <summary>Loss function used to train the critic network.</summary>
    public ILossFunction<T> CriticLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Number of samples in each training batch.</summary>
    public int BatchSize { get; set; } = 64;
    /// <summary>Maximum number of transitions kept in the replay buffer.</summary>
    public int ReplayBufferSize { get; set; } = 1000000;
    /// <summary>Number of environment steps collected before learning starts.</summary>
    public int WarmupSteps { get; set; } = 1000;
    /// <summary>Standard deviation of the Gaussian noise added to actions for exploration.</summary>
    public double ExplorationNoise { get; set; } = 0.1;
    /// <summary>Hidden layer sizes of the actor network.</summary>
    public List<int> ActorHiddenLayers { get; set; } = new List<int> { 400, 300 };
    /// <summary>Hidden layer sizes of the critic network.</summary>
    public List<int> CriticHiddenLayers { get; set; } = new List<int> { 400, 300 };

    /// <summary>Creates options with the default values.</summary>
    public DDPGOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.0001);
        CriticLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.001);
    }
}
