using AiDotNet.LossFunctions;

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
public class DreamerOptions<T>
{
    public int ObservationSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    // World model architecture
    public int LatentSize { get; set; } = 200;
    public int DeterministicSize { get; set; } = 200;
    public int StochasticSize { get; set; } = 30;
    public int HiddenSize { get; set; } = 200;

    // Training parameters
    public int BatchSize { get; set; } = 50;
    public int BatchLength { get; set; } = 50;
    public int ImaginationHorizon { get; set; } = 15;
    public T DiscountFactor { get; set; }

    // Model losses
    public double KLScale { get; set; } = 1.0;
    public double RewardScale { get; set; } = 1.0;
    public double ContinueScale { get; set; } = 1.0;

    public int ReplayBufferSize { get; set; } = 1000000;
    public int? Seed { get; set; }

    public DreamerOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.0001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
