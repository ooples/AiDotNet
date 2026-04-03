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
/// <para><b>JIT Compilation Support:</b> Deep RL agents support JIT compilation for policy inference
/// when their underlying neural networks support IJitCompilable. The JIT-compiled policy network
/// provides fast, deterministic action selection (without exploration) suitable for deployment.
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
    public override int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var network in Networks)
            {
                count += network.ParameterCount;
            }
            return count;
        }
    }

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
    protected virtual IJitCompilable<T>? GetPolicyNetworkForJit()
    {
        // Try to find a network that supports JIT compilation
        foreach (var network in Networks)
        {
            if (network is IJitCompilable<T> jitCompilable && jitCompilable.SupportsJitCompilation)
            {
                return jitCompilable;
            }
        }
        return null;
    }
}
