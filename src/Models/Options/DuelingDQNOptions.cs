using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dueling DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DuelingDQNOptions<T> : ModelOptions
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Learning rate of the optimizer.</summary>
    public T LearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Loss function used to fit Q-values to their TD targets.</summary>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Initial exploration rate for epsilon-greedy action selection.</summary>
    public double EpsilonStart { get; set; } = 1.0;
    /// <summary>Final, minimum exploration rate for epsilon-greedy action selection.</summary>
    public double EpsilonEnd { get; set; } = 0.01;
    /// <summary>Rate at which the exploration rate decays from EpsilonStart toward EpsilonEnd.</summary>
    public double EpsilonDecay { get; set; } = 0.995;
    /// <summary>Number of samples in each training batch.</summary>
    public int BatchSize { get; set; } = 32;
    /// <summary>Maximum number of transitions kept in the replay buffer.</summary>
    public int ReplayBufferSize { get; set; } = 10000;
    /// <summary>Number of training steps between target-network updates.</summary>
    public int TargetUpdateFrequency { get; set; } = 1000;
    /// <summary>Number of environment steps collected before learning starts.</summary>
    public int WarmupSteps { get; set; } = 1000;
    /// <summary>Hidden layer sizes of the shared trunk that feeds both streams.</summary>
    public List<int> SharedLayers { get; set; } = new List<int> { 128 };
    /// <summary>Hidden layer sizes of the value stream, which estimates the value of the state.</summary>
    public List<int> ValueStreamLayers { get; set; } = new List<int> { 128 };
    /// <summary>Hidden layer sizes of the advantage stream, which estimates how much better each action is than average.</summary>
    public List<int> AdvantageStreamLayers { get; set; } = new List<int> { 128 };

    /// <summary>Creates options with the default values.</summary>
    public DuelingDQNOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
