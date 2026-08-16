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
public abstract partial class DeepReinforcementLearningAgentBase<T> : ReinforcementLearningAgentBase<T>
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
    /// Copies a complete, materialized parameter state between topology-compatible networks.
    /// </summary>
    /// <remarks>
    /// Target-network agents must not copy the lazy, pre-forward view: a concrete architecture can
    /// have shape-resolved weights whose storage has not been allocated yet, in which case a bare
    /// <c>GetParameters()</c> honestly returns an empty vector. Materializing both peers here gives
    /// every DQN/actor-critic implementation the same lifecycle rule. Comparing stable manifest
    /// fingerprints before applying values is stronger than a length check: it also rejects shifted
    /// slot boundaries or semantic-role changes that happen to preserve the aggregate count.
    /// </remarks>
    protected static void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (ReferenceEquals(source, target)) return;

        if (source is NeuralNetworkBase<T> sourceBase) sourceBase.MaterializeParameters();
        if (target is NeuralNetworkBase<T> targetBase) targetBase.MaterializeParameters();

        if (source is AiDotNet.Models.Parameters.IParameterManifestProvider sourceManifest &&
            target is AiDotNet.Models.Parameters.IParameterManifestProvider targetManifest)
        {
            var sourceLayout = sourceManifest.ParameterLayout;
            var targetLayout = targetManifest.ParameterLayout;
            if (!string.Equals(sourceLayout.Fingerprint, targetLayout.Fingerprint, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Cannot copy network weights from {source.GetType().Name} to {target.GetType().Name}: " +
                    $"their stable parameter manifests differ ({sourceLayout.Fingerprint} vs " +
                    $"{targetLayout.Fingerprint}).");
            }
        }

        target.UpdateParameters(source.GetParameters());
    }

    /// <summary>
    /// Gets the total number of trainable parameters across all networks.
    /// </summary>
    /// <remarks>
    /// Target-network agents must not copy the lazy, pre-forward view: a concrete architecture can
    /// have shape-resolved weights whose storage has not been allocated yet, in which case a bare
    /// <c>GetParameters()</c> honestly returns an empty vector. Materializing both peers here gives
    /// every DQN/actor-critic implementation the same lifecycle rule. Comparing stable manifest
    /// fingerprints before applying values is stronger than a length check: it also rejects shifted
    /// slot boundaries or semantic-role changes that happen to preserve the aggregate count.
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

    /// <summary>
    /// Applies pre-computed parameter-space gradients as a single gradient-descent step.
    /// </summary>
    /// <param name="gradients">Gradients laid out like <c>GetParameters()</c>.</param>
    /// <param name="learningRate">The step size.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradients"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="gradients"/> is not the same length as the parameter vector.
    /// </exception>
    /// <remarks>
    /// The length check is the point of this method rather than an afterthought: it is what
    /// separates a genuine parameter-space gradient from the output-space vector these agents used
    /// to return, which is shorter and means something else entirely. Failing loudly beats applying
    /// a mismatched vector by index.
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));

        var currentParams = GetParameters();
        if (gradients.Length != currentParams.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter vector length "
                + $"({currentParams.Length}).",
                nameof(gradients));
        }

        var updated = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
        {
            updated[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(updated);
    }

    /// <summary>
    /// Computes gradients of the loss with respect to this agent's PARAMETERS, laid out to match
    /// <c>GetParameters()</c> so the result can be handed straight to <c>ApplyGradients</c>.
    /// </summary>
    /// <param name="trainedNetwork">The network on the forward path, whose parameters receive gradients.</param>
    /// <param name="parameterOrder">
    /// Every network contributing to <c>GetParameters()</c>, in the exact order that method
    /// concatenates them. Networks other than <paramref name="trainedNetwork"/> get a zero slice.
    /// </param>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when a required argument is null.</exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when <paramref name="trainedNetwork"/> cannot produce parameter gradients.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>This returns parameter-space gradients, not output-space ones.</b> That is what
    /// <see cref="IGradientComputable{T, TInput, TOutput}.ComputeGradients"/> promises -- "gradients
    /// with respect to all model parameters" -- and what every consumer assumes: ApplyGradients
    /// subtracts the vector from the parameters element-wise, and Elastic Weight Consolidation,
    /// Gradient Episodic Memory and Memory Aware Synapses all build Fisher-information estimates
    /// out of it. Returning the loss gradient with respect to the network OUTPUT, as these agents
    /// previously did, produced a vector of the wrong length and the wrong meaning.
    /// </para>
    /// <para>
    /// The gradient itself comes from the network's own tape pass, so the agent never expresses a
    /// derivative of its own; a frozen target network is simply absent from the forward path and
    /// therefore correctly receives zeros rather than being skipped and shifting the layout.
    /// </para>
    /// </remarks>
    protected Vector<T> ComputeGradientsForNetwork(
        INeuralNetwork<T> trainedNetwork,
        IReadOnlyList<INeuralNetwork<T>> parameterOrder,
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction)
    {
        if (trainedNetwork is null) throw new ArgumentNullException(nameof(trainedNetwork));
        if (parameterOrder is null) throw new ArgumentNullException(nameof(parameterOrder));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (target is null) throw new ArgumentNullException(nameof(target));

        if (trainedNetwork is not IGradientComputable<T, Tensor<T>, Tensor<T>> computable)
        {
            throw new NotSupportedException(
                $"{GetType().Name} cannot compute parameter gradients because its network "
                + $"({trainedNetwork.GetType().Name}) does not implement "
                + "IGradientComputable<T, Tensor<T>, Tensor<T>>. Networks deriving from "
                + "NeuralNetworkBase<T> do.");
        }

        var trainedGradients = computable.ComputeGradients(
            Tensor<T>.FromVector(input),
            Tensor<T>.FromVector(target),
            lossFunction ?? LossFunction);

        // Single-network agents are the common case and need no copying.
        if (parameterOrder.Count == 1)
        {
            return trainedGradients;
        }

        int total = 0;
        foreach (var network in parameterOrder)
        {
            total += (int)network.ParameterCount;
        }

        var gradients = new Vector<T>(total);
        int offset = 0;
        foreach (var network in parameterOrder)
        {
            int width = (int)network.ParameterCount;
            if (ReferenceEquals(network, trainedNetwork))
            {
                if (trainedGradients.Length != width)
                {
                    throw new InvalidOperationException(
                        $"{trainedNetwork.GetType().Name} returned {trainedGradients.Length} gradients "
                        + $"for {width} parameters. The gradient layout must match GetParameters().");
                }

                for (int i = 0; i < width; i++)
                {
                    gradients[offset + i] = trainedGradients[i];
                }
            }

            offset += width;
        }

        return gradients;
    }
}
