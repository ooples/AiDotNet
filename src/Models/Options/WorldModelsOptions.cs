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
    /// <summary>Width in pixels of each observation image.</summary>
    public int ObservationWidth { get; init; } = 64;
    /// <summary>Height in pixels of each observation image.</summary>
    public int ObservationHeight { get; init; } = 64;
    /// <summary>Number of channels of each observation image.</summary>
    public int ObservationChannels { get; init; } = 3;
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }

    // VAE parameters
    /// <summary>Size of the VAE latent vector z.</summary>
    public int LatentSize { get; init; } = 32;
    /// <summary>Output channels of each convolutional layer of the VAE encoder.</summary>
    public List<int> VAEEncoderChannels { get; init; } = new List<int> { 32, 64, 128, 256 };
    /// <summary>Weight of the KL term in the VAE loss.</summary>
    public double VAEBeta { get; init; } = 1.0;  // KL weight

    // MDN-RNN parameters
    /// <summary>Hidden size of the MDN-RNN.</summary>
    public int RNNHiddenSize { get; init; } = 256;
    /// <summary>Number of recurrent layers of the MDN-RNN.</summary>
    public int RNNLayers { get; init; } = 1;
    /// <summary>Number of Gaussian mixture components of the MDN-RNN's output.</summary>
    public int NumMixtures { get; init; } = 5;  // For mixture density network
    /// <summary>Sampling temperature of the MDN-RNN when dreaming: higher values make imagined futures more uncertain.</summary>
    public double Temperature { get; init; } = 1.0;

    // Controller parameters
    /// <summary>Hidden layer sizes of the controller.</summary>
    public List<int> ControllerLayers { get; init; } = new List<int> { 32 };

    // Training parameters
    /// <summary>Number of training epochs of the VAE.</summary>
    public int VAEEpochs { get; init; } = 10;
    /// <summary>Number of training epochs of the MDN-RNN.</summary>
    public int RNNEpochs { get; init; } = 20;
    /// <summary>Number of CMA-ES generations used to train the controller.</summary>
    public int ControllerGenerations { get; init; } = 100;  // For CMA-ES
    /// <summary>Population size of the CMA-ES search that trains the controller.</summary>
    public int ControllerPopulationSize { get; init; } = 64;
    /// <summary>Number of steps of each rollout used to evaluate a controller candidate.</summary>
    public int RolloutLength { get; init; } = 1000;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
