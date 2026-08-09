using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Base class for deep reinforcement learning agents that use neural networks as function approximators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This class extends ReinforcementLearningAgentBase to provide specific support for neural network-based
/// RL algorithms. It manages neural network instances and provides infrastructure for deep RL methods.
/// </para>
/// <para><b>For Beginners:</b> This is the base class for modern "deep" RL agents.
///
/// Deep RL uses neural networks to approximate the policy and/or value functions, enabling
/// agents to handle high-dimensional state spaces (like images) and complex decision problems.
///
/// Classical RL methods (tabular Q-learning, linear approximation) inherit directly from
/// ReinforcementLearningAgentBase, while deep RL methods (DQN, PPO, A3C, etc.) inherit from
/// this class which adds neural network support.
///
/// Examples of deep RL algorithms:
/// - DQN family (DQN, Double DQN, Rainbow)
/// - Policy gradient methods (PPO, TRPO, A3C)
/// - Actor-Critic methods (SAC, TD3, DDPG)
/// - Model-based methods (Dreamer, MuZero, World Models)
/// - Transformer-based methods (Decision Transformer)
/// </para>
/// <para><b>Auto-Compile:</b> Policy inference goes through the standard neural-network path,
/// which is auto-compiled by Tensors' AutoTracer once the input-shape pattern repeats. No
/// explicit compile call is required. Users can opt out via
/// <c>TensorCodecOptions.Current.EnableCompilation = false</c>.
/// </para>
/// </remarks>
public abstract class DeepReinforcementLearningAgentBase<T> : ReinforcementLearningAgentBase<T>
{
    /// <summary>
    /// Gets the global execution engine for hardware-accelerated vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The neural network(s) used by this agent for function approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deep RL agents typically use one or more neural networks:
    /// - Value-based: Q-network (and possibly target network)
    /// - Policy-based: Policy network
    /// - Actor-Critic: Separate policy and value networks
    /// - Model-based: Dynamics model, reward model, etc.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Neural networks are the "brains" of deep RL agents. They learn to map states to:
    /// - Action values (Q-networks in DQN)
    /// - Action probabilities (Policy networks in PPO)
    /// - State values (Value networks in A3C)
    /// - Or combinations of these
    ///
    /// This list holds all the networks this agent uses. For example:
    /// - DQN: 1-2 networks (Q-network, optional target network)
    /// - A3C: 2 networks (policy network, value network)
    /// - SAC: 4+ networks (policy, two Q-networks, two target Q-networks)
    /// </para>
    /// </remarks>
    protected List<INeuralNetwork<T>> Networks;

    /// <summary>
    /// Initializes a new instance of the DeepReinforcementLearningAgentBase class.
    /// </summary>
    /// <param name="options">Configuration options for the agent.</param>
    protected DeepReinforcementLearningAgentBase(ReinforcementLearningOptions<T> options)
        : base(options)
    {
        Networks = new List<INeuralNetwork<T>>();
    }

    /// <summary>
    /// Gets the total number of trainable parameters across all networks.
    /// </summary>
    /// <remarks>
    /// This sums the parameter counts from all neural networks used by the agent.
    /// Useful for monitoring model complexity and memory requirements.
    /// </remarks>
    /// <summary>
    /// Folded from <see cref="GetParameters"/>, so the count and the vector cannot disagree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This used to sum <c>ParameterCount</c> across every entry in <c>Networks</c>. That is not
    /// what the subclasses actually expose: a deep RL agent holds both trainable and DERIVED
    /// networks, and <c>GetParameters</c> returns only the trainable ones. DQNAgent is the clearest
    /// case — <c>GetParameters()</c> is <c>_qNetwork.GetParameters()</c>, and <c>SetParameters</c>
    /// writes <c>_qNetwork</c> and then RECOMPUTES the target by copying weights across. The target
    /// network is derived state, not an independent parameter, so counting it overstated the total
    /// by exactly its size on all ten agents deriving from this base.
    /// </para>
    /// <para>
    /// A mismatch here is not cosmetic: callers pair the two by length, so a saved vector sized by
    /// the count restores into the wrong slots, and the agent silently keeps its initial weights.
    /// Deriving the count from the vector is the same rule PyTorch relies on — the count is a fold
    /// over the one registry, never a second opinion about it.
    /// </para>
    /// <para>
    /// This materialises the vector to measure it. Deliberate: correctness first, and no subclass's
    /// <c>GetParameters</c> reads <c>ParameterCount</c>, so there is no recursion. If profiling ever
    /// shows the allocation matters, the answer is a length-only walk over the SAME source the
    /// vector is built from, never a second hand-maintained sum.
    /// </para>
    /// </remarks>
    public override long ParameterCount => GetParameters().Length;

    /// <summary>
    /// Disposes of resources used by the agent, including neural networks.
    /// </summary>
    public override void Dispose()
    {
        foreach (var network in Networks)
        {
            if (network is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
        base.Dispose();
    }

    // ===== JIT Compilation Support =====

    /// <summary>
    /// Gets the policy network used for action selection.
    /// </summary>
    /// <returns>The policy network, or null if no policy network is available.</returns>
    /// <remarks>
    /// <para>
    /// Override this method in derived classes to return the network responsible for action selection.
    /// This enables JIT compilation support for policy inference.
    /// </para>
    /// <para><b>Examples:</b></para>
    /// <list type="bullet">
    /// <item><description><b>DQN:</b> Returns the Q-network (actions selected via argmax Q(s,a))</description></item>
    /// <item><description><b>PPO/A3C:</b> Returns the policy network (actor)</description></item>
    /// <item><description><b>SAC/TD3:</b> Returns the policy network (actor)</description></item>
    /// </list>
    /// </remarks>
    protected virtual INeuralNetworkModel<T>? GetPolicyNetworkForJit()
    {
        // JIT compilation has been removed — always returns null
        return null;
    }
}
