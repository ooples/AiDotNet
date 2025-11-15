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
    private int _observationSize;
    private int _actionSize;

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
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
