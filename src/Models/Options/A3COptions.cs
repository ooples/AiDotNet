using AiDotNet.LossFunctions;

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
public class A3COptions<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public bool IsContinuous { get; set; } = false;
    public T PolicyLearningRate { get; set; }
    public T ValueLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T EntropyCoefficient { get; set; }
    public T ValueLossCoefficient { get; set; }

    // A3C-specific parameters
    public int NumWorkers { get; set; } = 4;  // Number of parallel agents
    public int TMax { get; set; } = 5;  // Steps before updating global network
    public double MaxGradNorm { get; set; } = 0.5;  // Gradient clipping

    public ILossFunction<T> ValueLossFunction { get; set; } = new MeanSquaredError<T>();
    public List<int> PolicyHiddenLayers { get; set; } = [128, 128];
    public List<int> ValueHiddenLayers { get; set; } = [128, 128];
    public int? Seed { get; set; }

    public A3COptions()
    {
        var numOps = NumericOperations<T>.Instance;
        PolicyLearningRate = numOps.FromDouble(0.0001);
        ValueLearningRate = numOps.FromDouble(0.0005);
        DiscountFactor = numOps.FromDouble(0.99);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
    }
}
