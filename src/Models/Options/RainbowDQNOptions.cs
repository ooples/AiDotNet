using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Rainbow DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Rainbow DQN combines six extensions to DQN:
/// 1. Double Q-learning: Reduces overestimation bias
/// 2. Dueling networks: Separates value and advantage streams
/// 3. Prioritized replay: Samples important experiences more frequently
/// 4. Multi-step learning: Uses n-step returns for better credit assignment
/// 5. Distributional RL: Learns full distribution of returns (C51)
/// 6. Noisy networks: Parameter noise for exploration
/// </remarks>
public class RainbowDQNOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public bool UseNoisyNetworks { get; init; } = true;

    // Dueling network architecture
    public List<int> SharedLayers { get; init; } = new List<int> { 128 };

    /// <summary>
    /// Gets or sets the widths of the value stream's hidden ReLU layers, between the shared trunk and V(s).
    /// </summary>
    /// <value>One width per hidden layer, in order; empty for a value head straight off the trunk. Default: { 128 }.</value>
    /// <remarks>
    /// <para>The dueling network (Wang et al. 2016, Fig. 1) gives the value and advantage streams their own fully
    /// connected layers after the shared trunk. With distributional RL the value stream ends in one output per atom.</para>
    /// <para><b>For Beginners:</b> How big the part of the network is that judges "how good is this state".</para>
    /// </remarks>
    public List<int> ValueStreamLayers { get; init; } = new List<int> { 128 };

    /// <summary>
    /// Gets or sets the widths of the advantage stream's hidden ReLU layers, between the shared trunk and A(s, a).
    /// </summary>
    /// <value>One width per hidden layer, in order; empty for an advantage head straight off the trunk. Default: { 128 }.</value>
    /// <remarks>
    /// <para>The advantage stream (Wang et al. 2016, Fig. 1) is combined with the value stream as
    /// Q = V + (A − mean over actions of A); with distributional RL the mean is taken per atom.</para>
    /// <para><b>For Beginners:</b> How big the part of the network is that judges "how much better is each action
    /// than average".</para>
    /// </remarks>
    public List<int> AdvantageStreamLayers { get; init; } = new List<int> { 128 };

    // Prioritized experience replay parameters (base has UsePrioritizedReplay)
    public double PriorityAlpha { get; init; } = 0.6;
    public double PriorityBeta { get; init; } = 0.4;
    public double PriorityBetaIncrement { get; init; } = 0.001;
    public double PriorityEpsilon { get; init; } = 1e-6;

    // Multi-step learning parameters
    public int NSteps { get; init; } = 3;

    // Distributional RL (C51) parameters
    public bool UseDistributional { get; init; } = true;
    public int NumAtoms { get; init; } = 51;
    public double VMin { get; init; } = -10.0;
    public double VMax { get; init; } = 10.0;

    // Noisy networks parameters
    public double NoisyNetSigma { get; init; } = 0.5;

    /// <summary>
    /// Gets or sets a factory for the optimizer each of this agent's networks trains with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, the networks train with Adam at the agent's learning rate, with ε = 1e-8.
    /// Hessel et al. 2018 used ε = 1.5e-4; supply a factory to train with the paper's ε.
    /// </para>
    /// <para>
    /// The agent invokes the factory once for every network it builds, so the factory must create a new
    /// optimizer instance on each call. A shared instance would share its step counter, and Adam's bias
    /// correction would then advance on every network's step instead of only on that network's own.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the rule that turns each network's gradients into weight
    /// changes. Leave this unset to get Adam at the learning rate above; supply a factory such as
    /// <c>() =&gt; new AdamOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;(null, myOptions)</c> to choose a
    /// different optimizer or its settings. Return a new optimizer from every call.
    /// </para>
    /// </remarks>
    /// <value>A delegate that returns a new optimizer on each call, or <c>null</c> (the default) for Adam at the
    /// agent's learning rate.</value>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
