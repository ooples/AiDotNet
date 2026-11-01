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

    /// <summary>Number of features in one observation.</summary>
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

    /// <summary>Dimension of the action vector.</summary>
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
    /// <summary>Size of the latent representation the observation encoder produces.</summary>
    public int LatentSize { get; init; } = 200;
    /// <summary>Size of the deterministic (recurrent) part of the latent state.</summary>
    public int DeterministicSize { get; init; } = 200;
    /// <summary>Size of the stochastic part of the latent state.</summary>
    public int StochasticSize { get; init; } = 30;
    /// <summary>Hidden width of the model's prediction networks.</summary>
    public int HiddenSize { get; init; } = 200;

    // Training parameters
    /// <summary>Length of the sequences sampled for each world-model training batch.</summary>
    public int BatchLength { get; init; } = 50;
    /// <summary>Number of steps imagined in the learned model when training the actor and critic.</summary>
    public int ImaginationHorizon { get; init; } = 15;

    // Model losses
    /// <summary>Weight of the KL divergence between the posterior and prior latent states.</summary>
    public double KLScale { get; init; } = 1.0;
    /// <summary>Weight of the reward-prediction loss in the world-model objective.</summary>
    public double RewardScale { get; init; } = 1.0;
    /// <summary>Weight of the continuation-prediction loss in the world-model objective.</summary>
    public double ContinueScale { get; init; } = 1.0;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
