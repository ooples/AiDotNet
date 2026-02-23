namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabDDPM (Tabular Denoising Diffusion Probabilistic Model),
/// a diffusion-based model for generating realistic synthetic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabDDPM applies denoising diffusion models to tabular data with separate processes
/// for numerical and categorical features:
/// - <b>Gaussian diffusion</b> for continuous/numerical columns (adds Gaussian noise)
/// - <b>Multinomial diffusion</b> for categorical columns (transitions toward uniform distribution)
/// - A shared MLP denoiser with timestep embedding predicts the original data from noisy input
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDDPM works by gradually destroying data with noise, then learning
/// to reverse the process. Think of it like a restoration expert:
///
/// <b>Training (learning to restore):</b>
/// 1. Take a real data row
/// 2. Add a random amount of noise (more noise = more destroyed)
/// 3. Tell the model "this was step t out of 1000" and ask it to predict the noise
/// 4. The model learns to undo any amount of noise
///
/// <b>Generation (creating new data):</b>
/// 1. Start with pure random noise
/// 2. Ask the model to remove a tiny bit of noise (step 999 to 998)
/// 3. Repeat 1000 times until you have a clean, realistic row
///
/// For numbers: regular Gaussian noise (like static on a TV)
/// For categories: noise means randomly changing the category toward "equally likely"
///
/// Example:
/// <code>
/// var options = new TabDDPMOptions&lt;double&gt;
/// {
///     NumTimesteps = 1000,
///     MLPDimensions = new[] { 256, 256, 256 },
///     BatchSize = 4096,
///     Epochs = 1000
/// };
/// var tabddpm = new TabDDPMGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabDDPM: Modelling Tabular Data with Diffusion Models" (Kotelnikov et al., ICML 2023)
/// </para>
/// </remarks>
public class TabDDPMOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of diffusion timesteps.
    /// </summary>
    /// <value>The number of timesteps, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More timesteps means finer-grained noise addition/removal,
    /// which generally improves quality but slows down generation. 1000 is standard.
    /// Fewer steps (100-500) trade quality for speed.
    /// </para>
    /// </remarks>
    public int NumTimesteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the denoiser MLP.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256, 256].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The denoiser is a multi-layer perceptron (MLP) that predicts
    /// noise from noisy data + timestep. Wider/deeper networks can learn more complex
    /// patterns but require more data and compute.
    /// </para>
    /// </remarks>
    public int[] MLPDimensions { get; set; } = [256, 256, 256];

    /// <summary>
    /// Gets or sets the dropout rate for the denoiser MLP.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0 (no dropout).</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the beta (noise variance) schedule type.
    /// </summary>
    /// <value>The schedule type: "linear" or "cosine". Defaults to "linear".</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how noise increases over the diffusion steps:
    /// - <b>"linear"</b>: Noise increases at a constant rate (standard DDPM)
    /// - <b>"cosine"</b>: Noise increases slowly at first, then accelerates (often better quality)
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "linear";

    /// <summary>
    /// Gets or sets the starting value of the beta schedule.
    /// </summary>
    /// <value>The beta start value, defaulting to 1e-4.</value>
    public double BetaStart { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the ending value of the beta schedule.
    /// </summary>
    /// <value>The beta end value, defaulting to 0.02.</value>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 4096.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabDDPM typically uses larger batch sizes than CTGAN/TVAE
    /// because diffusion models benefit from batch statistics. Adjust based on your
    /// available memory and dataset size.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 1000.</value>
    public int Epochs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the learning rate for the optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of diffusion steps for the multinomial (categorical) diffusion process.
    /// </summary>
    /// <value>Number of categorical diffusion steps, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Categories need fewer steps than numbers because the noise
    /// process is simpler (gradually mixing toward uniform distribution). Using fewer
    /// steps for categories speeds up training without losing quality.
    /// </para>
    /// </remarks>
    public int NumCategoricalDiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dimension of the timestep embedding.
    /// </summary>
    /// <value>The timestep embedding dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model needs to know "how noisy is this input?"
    /// The timestep is converted to a rich vector representation using sinusoidal
    /// positional encoding (like in transformers). This dimension controls how
    /// detailed that representation is.
    /// </para>
    /// </remarks>
    public int TimestepEmbeddingDimension { get; set; } = 128;

}
