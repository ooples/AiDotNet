using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Double DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DoubleDQNOptions<T> : ModelOptions
{
    /// <summary>
    /// Dimension of the environment state vector.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of input features the agent receives each step.</para>
    /// </remarks>
    public int StateSize { get; set; } = 4;

    /// <summary>
    /// Number of discrete actions available to the agent.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of different actions the agent can choose from.</para>
    /// </remarks>
    public int ActionSize { get; set; } = 2;

    /// <summary>Learning rate for gradient updates.</summary>
    /// <value>Default: 0.001.</value>
    public T LearningRate { get; set; }

    /// <summary>Discount factor (gamma) for future rewards.</summary>
    /// <value>Default: 0.99.</value>
    public T DiscountFactor { get; set; }

    /// <summary>Loss function for training. Default is MSE.</summary>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>Initial exploration rate for epsilon-greedy policy.</summary>
    public double EpsilonStart { get; set; } = 1.0;

    /// <summary>Final exploration rate.</summary>
    public double EpsilonEnd { get; set; } = 0.01;

    /// <summary>Multiplicative decay factor applied to epsilon each episode.</summary>
    public double EpsilonDecay { get; set; } = 0.995;

    /// <summary>Number of experiences sampled per training update.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Maximum number of experiences stored in the replay buffer.</summary>
    public int ReplayBufferSize { get; set; } = 10000;

    /// <summary>Number of steps between target network updates.</summary>
    public int TargetUpdateFrequency { get; set; } = 1000;

    /// <summary>Number of random steps before training begins.</summary>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>Hidden layer sizes for the Q-network.</summary>
    public List<int> HiddenLayers { get; set; } = new List<int> { 64, 64 };

    public DoubleDQNOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
