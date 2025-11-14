using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

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
public class WorldModelsOptions<T> : ReinforcementLearningOptions<T>
{
    public int ObservationWidth { get; init; } = 64;
    public int ObservationHeight { get; init; } = 64;
    public int ObservationChannels { get; init; } = 3;
    public int ActionSize { get; init; }

    // VAE parameters
    public int LatentSize { get; init; } = 32;
    public List<int> VAEEncoderChannels { get; init; } = new List<int> { 32, 64, 128, 256 };
    public double VAEBeta { get; init; } = 1.0;  // KL weight

    // MDN-RNN parameters
    public int RNNHiddenSize { get; init; } = 256;
    public int RNNLayers { get; init; } = 1;
    public int NumMixtures { get; init; } = 5;  // For mixture density network
    public double Temperature { get; init; } = 1.0;

    // Controller parameters
    public List<int> ControllerLayers { get; init; } = new List<int> { 32 };

    // Training parameters
    public int VAEEpochs { get; init; } = 10;
    public int RNNEpochs { get; init; } = 20;
    public int ControllerGenerations { get; init; } = 100;  // For CMA-ES
    public int ControllerPopulationSize { get; init; } = 64;
    public int RolloutLength { get; init; } = 1000;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
