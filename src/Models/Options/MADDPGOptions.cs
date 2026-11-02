using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Multi-Agent DDPG (MADDPG) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MADDPG extends DDPG to multi-agent settings with centralized training and
/// decentralized execution. Critics observe all agents during training.
/// </para>
/// <para><b>For Beginners:</b>
/// MADDPG allows multiple agents to learn together in shared environments.
/// During training, agents can "see" what others are doing (centralized critics),
/// but during execution, each agent acts independently (decentralized actors).
///
/// Key features:
/// - **Centralized Training**: Critics see all agents' observations and actions
/// - **Decentralized Execution**: Actors only use their own observations
/// - **Continuous Actions**: Based on DDPG for continuous control
/// - **Cooperative or Competitive**: Works for both settings
///
/// Think of it like: Team sports where players practice together (centralized)
/// but during the game each player makes their own decisions (decentralized).
///
/// Examples: Robot coordination, traffic control, multi-player games
/// </para>
/// </remarks>
public class MADDPGOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>The number of agents trained jointly.</summary>
    /// <value>Default: 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Increase this when your environment has more decision-makers.</para>
    /// </remarks>
    public int NumAgents { get; init; } = 2;

    /// <summary>The per-agent observation vector size.</summary>
    /// <value>Default: 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many input features each agent sees at every step.</para>
    /// </remarks>
    public int StateSize { get; init; } = 4;

    /// <summary>The per-agent action vector size.</summary>
    /// <value>Default: 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many continuous action values each agent outputs per step.</para>
    /// </remarks>
    public int ActionSize { get; init; } = 2;

    /// <summary>Learning rate for actor network updates.</summary>
    /// <value>Default: 0.0001.</value>
    public T ActorLearningRate { get; init; }

    /// <summary>Learning rate for critic network updates.</summary>
    /// <value>Default: 0.001.</value>
    public T CriticLearningRate { get; init; }

    /// <summary>Soft update coefficient for target networks.</summary>
    /// <value>Default: 0.001.</value>
    public T TargetUpdateTau { get; init; }

    /// <summary>Standard deviation of Gaussian noise added for exploration.</summary>
    /// <value>Default: 0.1.</value>
    public double ExplorationNoise { get; init; } = 0.1;

    /// <summary>Hidden layer sizes for each agent's actor network.</summary>
    public List<int> ActorHiddenLayers { get; init; } = new List<int> { 64, 64 };

    /// <summary>Hidden layer sizes for each agent's critic network.</summary>
    public List<int> CriticHiddenLayers { get; init; } = new List<int> { 64, 64 };

    /// <summary>
    /// Gets or sets a factory for the optimizer each of this agent's actor and critic networks trains with.
    /// </summary>
    /// <value>
    /// A delegate that returns a new optimizer on each call, or <c>null</c> (the default): every actor then trains
    /// with Adam at <see cref="ActorLearningRate"/> and every critic with Adam at <see cref="CriticLearningRate"/>.
    /// </value>
    /// <remarks>
    /// <para>
    /// The agent invokes the factory once for every actor and critic it builds (two networks per agent), so the
    /// factory must create a new optimizer instance on each call. A shared instance would share its step counter, and
    /// Adam's bias correction would then advance on every network's step instead of only on that network's own.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer decides how the network weights change after each update. Leave this
    /// empty to use Adam at the actor and critic learning rates. Supply a factory only to use a different optimizer,
    /// and make it build a new optimizer every time it is called.</para>
    /// </remarks>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }

    /// <summary>
    /// Initializes a new instance with the paper's training settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Lowe et al. 2017 (Sec. 5 and Appendix) train every actor and critic with Adam at a learning rate of 0.01,
    /// soft-update the target networks with τ = 0.01, discount by γ = 0.95, sample minibatches of 1024 from a replay
    /// buffer of 10^6 transitions, and use two-layer ReLU MLPs of 64 units. Those are this constructor's values; the
    /// earlier 1e-4 / 1e-3 learning rates and τ = 0.001 were single-agent DDPG's, not MADDPG's. Any of them can
    /// still be overridden in an object initializer.
    /// </para>
    /// <para><b>For Beginners:</b> Creating the options with no arguments gives you the settings the MADDPG paper
    /// trained with.</para>
    /// </remarks>
    public MADDPGOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.01);
        CriticLearningRate = numOps.FromDouble(0.01);
        TargetUpdateTau = numOps.FromDouble(0.01);
        DiscountFactor = numOps.FromDouble(0.95);
        BatchSize = 1024;
        ReplayBufferSize = 1000000;
    }

    /// <summary>
    /// Initializes a new instance by copying values from another instance.
    /// </summary>
    /// <param name="other">The options to copy from.</param>
    public MADDPGOptions(MADDPGOptions<T> other) : this()
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        // Copy base class properties
        // Seed included: it is declared on the base rather than in this file, so a copy constructor
        // written from the local declarations alone misses it, and a clone then initializes from a
        // different random stream than the instance it was cloned from.
        Seed = other.Seed;
        LearningRate = other.LearningRate;
        DiscountFactor = other.DiscountFactor;
        LossFunction = other.LossFunction;
        BatchSize = other.BatchSize;
        ReplayBufferSize = other.ReplayBufferSize;
        TargetUpdateFrequency = other.TargetUpdateFrequency;
        UsePrioritizedReplay = other.UsePrioritizedReplay;
        EpsilonStart = other.EpsilonStart;
        EpsilonEnd = other.EpsilonEnd;

        // Copy MADDPG-specific properties
        NumAgents = other.NumAgents;
        StateSize = other.StateSize;
        ActionSize = other.ActionSize;
        ActorLearningRate = other.ActorLearningRate;
        CriticLearningRate = other.CriticLearningRate;
        TargetUpdateTau = other.TargetUpdateTau;
        ExplorationNoise = other.ExplorationNoise;
        // Cloned, not shared: a bare assignment leaves the clone and the original writing
        // through the SAME buffer, so mutating one silently reconfigures the other.
        // A null here would be stored as a non-nullable property holding null, so the failure
        // surfaces much later as a null reference in model code with nothing pointing back at the
        // clone that produced it. Reject it at the boundary instead.
        if (other.ActorHiddenLayers is null)
        {
            throw new ArgumentException(
                $"{nameof(other.ActorHiddenLayers)} must not be null.", nameof(other));
        }
        ActorHiddenLayers = new List<int>(other.ActorHiddenLayers);
        // Cloned, not shared: a bare assignment leaves the clone and the original writing
        // through the SAME buffer, so mutating one silently reconfigures the other.
        // A null here would be stored as a non-nullable property holding null, so the failure
        // surfaces much later as a null reference in model code with nothing pointing back at the
        // clone that produced it. Reject it at the boundary instead.
        if (other.CriticHiddenLayers is null)
        {
            throw new ArgumentException(
                $"{nameof(other.CriticHiddenLayers)} must not be null.", nameof(other));
        }
        CriticHiddenLayers = new List<int>(other.CriticHiddenLayers);
        OptimizerFactory = other.OptimizerFactory;
    }

    /// <summary>
    /// Validates that required properties are set.
    /// </summary>
    public void Validate()
    {
        if (NumAgents <= 0)
            throw new ArgumentException("NumAgents must be greater than 0", nameof(NumAgents));
        if (StateSize <= 0)
            throw new ArgumentException("StateSize must be greater than 0", nameof(StateSize));
        if (ActionSize <= 0)
            throw new ArgumentException("ActionSize must be greater than 0", nameof(ActionSize));
    }
}
