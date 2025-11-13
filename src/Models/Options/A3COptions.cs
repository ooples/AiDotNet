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
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public bool IsContinuous { get; init; } = false;
    public T PolicyLearningRate { get; init; }
    public T ValueLearningRate { get; init; }
    public T EntropyCoefficient { get; init; }
    public T ValueLossCoefficient { get; init; }

    // A3C-specific parameters
    public int NumWorkers { get; init; } = 4;  // Number of parallel agents
    public int TMax { get; init; } = 5;  // Steps before updating global network

    public ILossFunction<T> ValueLossFunction { get; init; } = new MeanSquaredError<T>();
    public List<int> PolicyHiddenLayers { get; init; } = new List<int> { 128, 128 };
    public List<int> ValueHiddenLayers { get; init; } = new List<int> { 128, 128 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }

    public A3COptions()
    {
        var numOps = NumericOperations<T>.Instance;
        PolicyLearningRate = numOps.FromDouble(0.0001);
        ValueLearningRate = numOps.FromDouble(0.0005);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
    }
}
