using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Deep Q-Network (DQN) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DQNOptions<T> : ModelOptions
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Learning rate of the optimizer.</summary>
    public T LearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Initial exploration rate for epsilon-greedy action selection.</summary>
    public double EpsilonStart { get; set; } = 1.0;
    /// <summary>Final, minimum exploration rate for epsilon-greedy action selection.</summary>
    public double EpsilonEnd { get; set; } = 0.01;
    /// <summary>Rate at which the exploration rate decays from EpsilonStart toward EpsilonEnd.</summary>
    public double EpsilonDecay { get; set; } = 0.995;
    /// <summary>Number of samples in each training batch.</summary>
    public int BatchSize { get; set; } = 64;
    /// <summary>Maximum number of transitions kept in the replay buffer.</summary>
    public int ReplayBufferSize { get; set; } = 100000;
    /// <summary>Number of training steps between target-network updates.</summary>
    public int TargetUpdateFrequency { get; set; } = 1000;
    /// <summary>Number of environment steps collected before learning starts.</summary>
    public int WarmupSteps { get; set; } = 1000;
    /// <summary>Loss function used to fit Q-values to their TD targets.</summary>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Optimizer for Q-network training. Default: Adam (per Mnih et al. 2015).
    /// </summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    /// <summary>Hidden layer sizes of the Q-network.</summary>
    public List<int> HiddenLayers { get; set; } = new List<int> { 128, 128 };

    /// <summary>Creates options with the default values.</summary>
    public DQNOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
