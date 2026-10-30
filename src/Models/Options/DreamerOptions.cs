using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dreamer agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dreamer learns a world model in latent space and uses it for planning.
/// It combines representation learning, dynamics modeling, and policy learning.
/// </para>
/// <para><b>For Beginners:</b>
/// Dreamer learns a "mental model" of how the environment works, then uses that
/// model to imagine future scenarios and plan actions - like playing chess in your head.
///
/// Key components:
/// - **World Model**: Learns environment dynamics in compact latent space
/// - **Representation Network**: Encodes observations to latent states
/// - **Transition Model**: Predicts next latent state
/// - **Reward Model**: Predicts rewards
/// - **Actor-Critic**: Learns policy by imagining trajectories
///
/// Think of it like: Learning physics by observation, then using that knowledge
/// to predict "what happens if I do X" without actually doing it.
///
/// Advantages: Sample efficient, works with image observations, enables planning
/// </para>
/// </remarks>
public class DreamerOptions<T> : ReinforcementLearningOptions<T>
{
    private int _observationSize = 1;
    private int _actionSize = 1;

    public int ObservationSize
    {
        get => _observationSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("ObservationSize must be positive", nameof(ObservationSize));
            }
            _observationSize = value;
        }
    }

    public int ActionSize
    {
        get => _actionSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("ActionSize must be positive", nameof(ActionSize));
            }
            _actionSize = value;
        }
    }

    // World model architecture
    public int LatentSize { get; init; } = 200;
    public int DeterministicSize { get; init; } = 200;
    public int StochasticSize { get; init; } = 30;
    public int HiddenSize { get; init; } = 200;

    // Training parameters
    public int BatchLength { get; init; } = 50;
    public int ImaginationHorizon { get; init; } = 15;

    // Model losses
    public double KLScale { get; init; } = 1.0;
    public double RewardScale { get; init; } = 1.0;
    public double ContinueScale { get; init; } = 1.0;

    /// <summary>
    /// Gets or sets a factory for the optimizer each of this agent's networks trains with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, each network trains with Adam (β1 0.9, β2 0.999, ε 1e-8) at the agent's
    /// <c>LearningRate</c>.
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
    /// agent's <c>LearningRate</c>.</value>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
