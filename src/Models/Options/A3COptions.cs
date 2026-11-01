using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Asynchronous Advantage Actor-Critic (A3C) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A3C runs multiple agents in parallel, each learning from different experiences.
/// The parallel exploration provides diverse training data and stabilizes learning.
/// </para>
/// <para><b>For Beginners:</b>
/// A3C is like having multiple students learn the same subject simultaneously,
/// each with different experiences. They periodically share what they learned
/// with a central "teacher" (global network), and everyone benefits from the
/// combined knowledge.
///
/// Key features:
/// - **Asynchronous**: Multiple agents run in parallel
/// - **Actor-Critic**: Learns both policy and value function
/// - **No Replay Buffer**: Uses on-policy learning
/// - **Diverse Exploration**: Different agents explore different strategies
///
/// Famous for: DeepMind's breakthrough paper (2016), enables CPU-only training
/// </para>
/// </remarks>
public class A3COptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Whether the action space is continuous rather than discrete.</summary>
    public bool IsContinuous { get; init; } = false;
    /// <summary>Learning rate of the policy network.</summary>
    public T PolicyLearningRate { get; init; }
    /// <summary>Learning rate of the value network.</summary>
    public T ValueLearningRate { get; init; }
    /// <summary>Weight of the entropy bonus in the objective, which encourages exploration.</summary>
    public T EntropyCoefficient { get; init; }
    /// <summary>Weight of the value loss in the combined objective.</summary>
    public T ValueLossCoefficient { get; init; }

    // A3C-specific parameters
    /// <summary>Number of parallel worker agents that explore and update the shared global network.</summary>
    public int NumWorkers { get; init; } = 4;  // Number of parallel agents
    /// <summary>Maximum number of steps a worker takes before it pushes its gradients to the global network (t_max in Mnih et al. 2016).</summary>
    public int TMax { get; init; } = 5;  // Steps before updating global network

    /// <summary>Loss function used to train the value network.</summary>
    public ILossFunction<T> ValueLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer sizes of the policy network.</summary>
    public List<int> PolicyHiddenLayers { get; init; } = new List<int> { 128, 128 };
    /// <summary>Hidden layer sizes of the value network.</summary>
    public List<int> ValueHiddenLayers { get; init; } = new List<int> { 128, 128 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }

    /// <summary>Creates options with the default values.</summary>
    public A3COptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0001);
        ValueLearningRate = numOps.FromDouble(0.0005);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
