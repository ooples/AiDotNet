using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for World Models agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// World Models learns compact spatial and temporal representations using VAE and RNN.
/// The agent learns entirely within the "dream" of its learned world model.
/// </para>
/// <para><b>For Beginners:</b>
/// World Models is inspired by how humans learn: we build mental models of the world,
/// then make decisions based on those models rather than raw sensory input.
///
/// Key components:
/// - **VAE (V)**: Compresses visual observations into compact latent codes
/// - **MDN-RNN (M)**: Learns temporal dynamics (what happens next)
/// - **Controller (C)**: Simple linear/neural policy acting in latent space
/// - **Learning in Dreams**: Agent trains entirely in imagined rollouts
///
/// Think of it like: First, learn to compress images (VAE). Then, learn how
/// compressed images change over time (RNN). Finally, learn to act based on
/// compressed predictions (controller).
///
/// Famous for: Car racing from pixels, learning with limited real environment samples
/// </para>
/// </remarks>
public class WorldModelsOptions<T>
{
    public int ObservationWidth { get; set; } = 64;
    public int ObservationHeight { get; set; } = 64;
    public int ObservationChannels { get; set; } = 3;
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    // VAE parameters
    public int LatentSize { get; set; } = 32;
    public List<int> VAEEncoderChannels { get; set; } = [32, 64, 128, 256];
    public double VAEBeta { get; set; } = 1.0;  // KL weight

    // MDN-RNN parameters
    public int RNNHiddenSize { get; set; } = 256;
    public int RNNLayers { get; set; } = 1;
    public int NumMixtures { get; set; } = 5;  // For mixture density network
    public double Temperature { get; set; } = 1.0;

    // Controller parameters
    public List<int> ControllerLayers { get; set; } = [32];

    // Training parameters
    public int VAEEpochs { get; set; } = 10;
    public int RNNEpochs { get; set; } = 20;
    public int ControllerGenerations { get; set; } = 100;  // For CMA-ES
    public int ControllerPopulationSize { get; set; } = 64;

    public int BatchSize { get; set; } = 100;
    public int RolloutLength { get; set; } = 1000;
    public T DiscountFactor { get; set; }
    public int? Seed { get; set; }

    public WorldModelsOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
