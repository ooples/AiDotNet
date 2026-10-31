using AiDotNet.Attributes;
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
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    /// <summary>
    /// Discount factor gamma applied to future rewards in the Q-learning target.
    /// </summary>
    /// <value>A value in [0, 1]. Default: 0.99, the paper's value.</value>
    /// <remarks>
    /// <para>
    /// The Q-learning target is r + gamma * max_a' Q(s', a'), so gamma sets how far ahead the agent plans: the
    /// weight of a reward k steps away is gamma^k.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the agent cares about future rewards compared with
    /// immediate ones. At 0.99 a reward 100 steps away still counts about a third as much as one now; values near 0
    /// make the agent short-sighted.
    /// </para>
    /// </remarks>
    [PaperDefault(0.99, Source = "Mnih et al. 2015 (Nature 518), Extended Data Table 1: discount factor")]
    public T DiscountFactor { get; set; }

    /// <summary>
    /// Initial epsilon of the epsilon-greedy exploration schedule.
    /// </summary>
    /// <value>A probability in [0, 1]. Default: 1.0, the paper's value.</value>
    /// <remarks>
    /// <para>
    /// With probability epsilon the agent takes a uniformly random action instead of the greedy one. Epsilon starts
    /// here and decays toward <see cref="EpsilonEnd"/> by <see cref="EpsilonDecay"/>.
    /// </para>
    /// <para><b>For Beginners:</b> At 1.0 the agent starts by acting completely at random, which lets it see many
    /// situations before it trusts its own estimates; it then explores less and less as it learns.
    /// </para>
    /// </remarks>
    [PaperDefault(1.0, Source = "Mnih et al. 2015 (Nature 518), Extended Data Table 1: initial exploration")]
    public double EpsilonStart { get; set; } = 1.0;
    public double EpsilonEnd { get; set; } = 0.01;
    public double EpsilonDecay { get; set; } = 0.995;
    public int BatchSize { get; set; } = 64;
    public int ReplayBufferSize { get; set; } = 100000;
    public int TargetUpdateFrequency { get; set; } = 1000;
    public int WarmupSteps { get; set; } = 1000;
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Optimizer for Q-network training. Default: Adam (per Mnih et al. 2015).
    /// </summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    public List<int> HiddenLayers { get; set; } = new List<int> { 128, 128 };

    public DQNOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }

    /// <summary>Initializes a new instance as a copy of <paramref name="other"/>, with every setting copied.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public DQNOptions(DQNOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        StateSize = other.StateSize;
        ActionSize = other.ActionSize;
        LearningRate = other.LearningRate;
        DiscountFactor = other.DiscountFactor;
        EpsilonStart = other.EpsilonStart;
        EpsilonEnd = other.EpsilonEnd;
        EpsilonDecay = other.EpsilonDecay;
        BatchSize = other.BatchSize;
        ReplayBufferSize = other.ReplayBufferSize;
        TargetUpdateFrequency = other.TargetUpdateFrequency;
        WarmupSteps = other.WarmupSteps;
        LossFunction = other.LossFunction;
        Optimizer = other.Optimizer;
        HiddenLayers = new List<int>(other.HiddenLayers);
    }
}
