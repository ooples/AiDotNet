using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for World Models agents (Ha &amp; Schmidhuber 2018, "Recurrent World Models Facilitate
/// Policy Evolution", arXiv 1803.10122).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// World Models has three parts. V is a variational autoencoder that compresses each observation into a latent
/// code z. M is a mixture-density recurrent network (an LSTM whose output parameterises a Gaussian mixture over
/// the next z) that models how z evolves under an action. C is a small controller mapping [z, h] to an action,
/// trained with CMA-ES. The defaults below are the paper's Car Racing configuration unless a property says
/// otherwise.
/// </para>
/// <para><b>For Beginners:</b>
/// World Models is inspired by how humans learn: we build mental models of the world,
/// then make decisions based on those models rather than raw sensory input.
///
/// Key components:
/// - **VAE (V)**: Compresses visual observations into compact latent codes
/// - **MDN-RNN (M)**: Learns temporal dynamics (what happens next)
/// - **Controller (C)**: A tiny policy acting on V's code and M's memory
/// - **Learning in Dreams**: The controller is evolved inside rollouts imagined by M
///
/// Think of it like: First, learn to compress images (VAE). Then, learn how
/// compressed images change over time (RNN). Finally, learn to act based on
/// compressed predictions (controller).
/// </para>
/// </remarks>
public class WorldModelsOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Gets the observation width.
    /// </summary>
    /// <remarks>
    /// <para>The paper resizes every frame to 64x64 before V sees it (appendix, "Variational Autoencoder").</para>
    /// <para><b>For Beginners:</b> Observations are flattened vectors of Width x Height x Channels numbers.</para>
    /// </remarks>
    /// <value>The observation width; defaults to 64.</value>
    public int ObservationWidth { get; init; } = 64;

    /// <summary>
    /// Gets the observation height.
    /// </summary>
    /// <remarks>
    /// <para>The paper resizes every frame to 64x64 before V sees it (appendix, "Variational Autoencoder").</para>
    /// <para><b>For Beginners:</b> Observations are flattened vectors of Width x Height x Channels numbers.</para>
    /// </remarks>
    /// <value>The observation height; defaults to 64.</value>
    public int ObservationHeight { get; init; } = 64;

    /// <summary>
    /// Gets the number of observation channels.
    /// </summary>
    /// <remarks>
    /// <para>The paper's frames are RGB, three channels.</para>
    /// <para><b>For Beginners:</b> 3 for colour images, 1 for grayscale or plain feature vectors.</para>
    /// </remarks>
    /// <value>The channel count; defaults to 3.</value>
    public int ObservationChannels { get; init; } = 3;

    /// <summary>
    /// Gets the dimension of the continuous action.
    /// </summary>
    /// <remarks>
    /// <para>Car Racing's action has three components (steering, gas, brake). The controller's tanh bounds every
    /// component to [-1, 1] (appendix, "Controller").</para>
    /// <para><b>For Beginners:</b> How many numbers make up one action.</para>
    /// </remarks>
    /// <value>The action dimension; must be set (the parameterless agent constructor uses 2).</value>
    public int ActionSize { get; init; }

    /// <summary>
    /// Gets the size of V's latent code z.
    /// </summary>
    /// <remarks>
    /// <para>N_z is 32 in Car Racing and 64 in Doom (appendix, "Variational Autoencoder").</para>
    /// <para><b>For Beginners:</b> How many numbers the VAE squeezes each observation into.</para>
    /// </remarks>
    /// <value>The latent size; defaults to 32.</value>
    public int LatentSize { get; init; } = 32;

    /// <summary>
    /// Gets the widths of V's hidden encoder layers (mirrored, reversed, in the decoder).
    /// </summary>
    /// <remarks>
    /// <para>The paper's encoder is four stride-2 convolutions with 32, 64, 128 and 256 channels (appendix, "Variational Autoencoder").
    /// This agent takes flattened observations, so each entry is the width of a ReLU dense layer instead.</para>
    /// <para><b>For Beginners:</b> The sizes of the layers that progressively compress an observation.</para>
    /// </remarks>
    /// <value>The hidden widths; defaults to 32, 64, 128, 256.</value>
    public List<int> VAEEncoderChannels { get; init; } = new List<int> { 32, 64, 128, 256 };

    /// <summary>
    /// Gets the weight of the KL term in V's loss.
    /// </summary>
    /// <remarks>
    /// <para>V is trained on the L2 reconstruction error plus the KL divergence of q(z|x) from N(0, I)
    /// (appendix, "Variational Autoencoder"): loss = ||x - D(z)||^2 + VAEBeta * KL. The paper's plain VAE is VAEBeta = 1; a larger value
    /// pulls the codes harder towards the unit Gaussian prior (the beta-VAE trade-off).</para>
    /// <para><b>For Beginners:</b> How strongly the compressed codes are kept "tidy" versus exact. Zero turns the
    /// VAE into a plain autoencoder.</para>
    /// </remarks>
    /// <value>The KL weight; defaults to 1.0. Must be non-negative and finite.</value>
    public double VAEBeta { get; init; } = 1.0;

    /// <summary>
    /// Gets the number of hidden units in each of M's LSTM layers.
    /// </summary>
    /// <remarks>
    /// <para>256 in Car Racing, 512 in Doom (appendix, "MDN-RNN").</para>
    /// <para><b>For Beginners:</b> The size of the world model's memory, h, which the controller also reads.</para>
    /// </remarks>
    /// <value>The hidden size; defaults to 256.</value>
    public int RNNHiddenSize { get; init; } = 256;

    /// <summary>
    /// Gets the number of stacked LSTM layers in M.
    /// </summary>
    /// <remarks>
    /// <para>The paper uses one LSTM layer. With more, each layer feeds the next and the controller reads the top
    /// layer's hidden state.</para>
    /// <para><b>For Beginners:</b> How many memory layers are stacked in the world model.</para>
    /// </remarks>
    /// <value>The layer count; defaults to 1.</value>
    public int RNNLayers { get; init; } = 1;

    /// <summary>
    /// Gets the number of Gaussian components in M's mixture density output.
    /// </summary>
    /// <remarks>
    /// <para>"In both tasks, we used 5 Gaussian mixtures" with a factored (diagonal) Gaussian per latent dimension
    /// (appendix, "MDN-RNN").</para>
    /// <para><b>For Beginners:</b> How many alternative futures M can predict for each number of z.</para>
    /// </remarks>
    /// <value>The mixture count; defaults to 5.</value>
    public int NumMixtures { get; init; } = 5;

    /// <summary>
    /// Gets the sampling temperature tau used when M generates the next latent in a dream.
    /// </summary>
    /// <remarks>
    /// <para>Section "Cheating the World Model": raising tau makes the dream more uncertain and harder to
    /// exploit. As in the authors' reference implementation, the mixture logits are divided by tau and each
    /// component's standard deviation is multiplied by sqrt(tau). The Doom agent was trained at tau = 1.15.</para>
    /// <para><b>For Beginners:</b> Higher values make imagined futures noisier, which stops the controller from
    /// learning tricks that only work in an over-confident dream.</para>
    /// </remarks>
    /// <value>The temperature; defaults to 1.0. Must be positive and finite.</value>
    public double Temperature { get; init; } = 1.0;

    /// <summary>
    /// Gets the widths of the controller's hidden tanh layers.
    /// </summary>
    /// <remarks>
    /// <para>The paper's full controller is a single linear layer, a_t = tanh(W_c [z_t h_t] + b_c), with 867
    /// parameters in Car Racing (section "Controller (C) Model"). Its ablation adds one hidden layer of 40 tanh units (appendix, "Evolution Strategies").
    /// Empty means the paper's linear controller.</para>
    /// <para><b>For Beginners:</b> Leave empty for the paper's tiny linear policy; add widths for a small neural
    /// policy.</para>
    /// </remarks>
    /// <value>The hidden widths; defaults to empty (linear).</value>
    public List<int> ControllerLayers { get; init; } = new List<int>();

    /// <summary>
    /// Gets the number of passes over the stored observations V trains for in each training cycle.
    /// </summary>
    /// <remarks>
    /// <para>"We trained the model for 1 epoch over the data collected from a random policy" (appendix, "Variational Autoencoder").</para>
    /// <para><b>For Beginners:</b> How many times the VAE studies every stored observation before M is trained.</para>
    /// </remarks>
    /// <value>The epoch count; defaults to 1.</value>
    public int VAEEpochs { get; init; } = 1;

    /// <summary>
    /// Gets the number of passes over the stored sequences M trains for in each training cycle.
    /// </summary>
    /// <remarks>
    /// <para>"The MDN-RNNs were trained for 20 epochs on the data collected from a random policy agent"
    /// (appendix, "MDN-RNN").</para>
    /// <para><b>For Beginners:</b> How many times the world model studies every stored sequence.</para>
    /// </remarks>
    /// <value>The epoch count; defaults to 20.</value>
    public int RNNEpochs { get; init; } = 20;

    /// <summary>
    /// Gets the maximum length of the sequences M is trained on.
    /// </summary>
    /// <remarks>
    /// <para>The paper trains M on whole episodes; 1000 is the Car Racing episode length. The stored transitions
    /// are cut into consecutive windows of at most this length that never cross an episode end; each training batch
    /// is padded to this length and the padded steps are masked out of the loss, so a value far above the typical
    /// episode length spends compute on padding.</para>
    /// <para><b>For Beginners:</b> How many consecutive steps M sees at once while learning.</para>
    /// </remarks>
    /// <value>The window length; defaults to 1000.</value>
    public int RNNSequenceLength { get; init; } = 1000;

    /// <summary>
    /// Gets the number of CMA-ES generations the controller is evolved for in each training cycle.
    /// </summary>
    /// <remarks>
    /// <para>"After 1800 generations, an agent was able to achieve an average score of 900.46" (appendix, "Evolution Strategies").</para>
    /// <para><b>For Beginners:</b> How many rounds of "try a population of controllers, keep the best ideas" run.</para>
    /// </remarks>
    /// <value>The generation count; defaults to 1800.</value>
    public int ControllerGenerations { get; init; } = 1800;

    /// <summary>
    /// Gets the number of candidate controllers CMA-ES samples per generation.
    /// </summary>
    /// <remarks>
    /// <para>"We used a population size of 64" (appendix, "Evolution Strategies"). CMA-ES needs at least 4.</para>
    /// <para><b>For Beginners:</b> How many slightly different controllers are tried each round.</para>
    /// </remarks>
    /// <value>The population size; defaults to 64.</value>
    public int ControllerPopulationSize { get; init; } = 64;

    /// <summary>
    /// Gets the number of dream rollouts each candidate controller's fitness is averaged over.
    /// </summary>
    /// <remarks>
    /// <para>Each agent performed "the task 16 times with different initial random seeds"; the fitness is the
    /// average cumulative reward (appendix, "Evolution Strategies").</para>
    /// <para><b>For Beginners:</b> Each candidate is judged on several imagined episodes so one lucky dream does
    /// not decide.</para>
    /// </remarks>
    /// <value>The rollout count; defaults to 16.</value>
    public int ControllerEvaluationRollouts { get; init; } = 16;

    /// <summary>
    /// Gets the initial CMA-ES step size (the standard deviation of the first generation around the current
    /// controller).
    /// </summary>
    /// <remarks>
    /// <para>The paper does not state it; 0.1 suits a controller whose weights start near zero.</para>
    /// <para><b>For Beginners:</b> How far the first candidates may stray from the current controller.</para>
    /// </remarks>
    /// <value>The step size; defaults to 0.1. Must be positive and finite.</value>
    public double ControllerInitialStepSize { get; init; } = 0.1;

    /// <summary>
    /// Gets the maximum number of steps of each dream rollout used to score a controller.
    /// </summary>
    /// <remarks>
    /// <para>Car Racing episodes last 1000 steps; a dream rollout also ends early once M predicts the episode is
    /// done with probability above 50% (appendix, "MDN-RNN").</para>
    /// <para><b>For Beginners:</b> How long each imagined episode may run.</para>
    /// </remarks>
    /// <value>The rollout length; defaults to 1000.</value>
    public int RolloutLength { get; init; } = 1000;

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
