namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabSyn, a state-of-the-art synthetic tabular data generator
/// that combines a VAE with latent diffusion for high-quality generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabSyn operates in two phases:
/// 1. <b>VAE pretraining</b>: Learns a compact latent representation of the tabular data
/// 2. <b>Latent diffusion</b>: Trains a diffusion model in the VAE's latent space
///
/// Generation: Sample from the diffusion model in latent space, then decode with the VAE.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabSyn is like TVAE + TabDDPM combined in a clever way:
///
/// 1. First, a VAE learns to "compress" your data into a small summary (latent codes)
/// 2. Then, a diffusion model learns the distribution of those compressed summaries
/// 3. To generate new data: the diffusion model creates new summaries, and the VAE's
///    decoder converts them back to realistic data rows
///
/// This two-step approach often produces the highest quality synthetic data because:
/// - The VAE handles the complex mixed-type structure
/// - The diffusion model learns a simpler, continuous distribution in latent space
///
/// Example:
/// <code>
/// var options = new TabSynOptions&lt;double&gt;
/// {
///     EncoderDimensions = new[] { 256, 256 },
///     DecoderDimensions = new[] { 256, 256 },
///     LatentDimension = 64,
///     DiffusionSteps = 1000,
///     VAEEpochs = 100,
///     DiffusionEpochs = 100
/// };
/// var tabsyn = new TabSynGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabSyn: Bridging the Gap" (Zhang et al., NeurIPS 2023)
/// </para>
/// </remarks>
public class TabSynOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the hidden layer sizes for the VAE encoder.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] EncoderDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the VAE decoder.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] DecoderDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the dimension of the VAE latent space.
    /// </summary>
    /// <value>The latent dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the compressed representation.
    /// The diffusion model operates in this space. Smaller values mean more compression.
    /// </para>
    /// </remarks>
    public int LatentDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the diffusion denoiser MLP.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] DiffusionMLPDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the number of diffusion timesteps.
    /// </summary>
    /// <value>The number of timesteps, defaulting to 1000.</value>
    public int DiffusionSteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the beta schedule start value for diffusion.
    /// </summary>
    /// <value>Beta start, defaulting to 1e-4.</value>
    public double BetaStart { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the beta schedule end value for diffusion.
    /// </summary>
    /// <value>Beta end, defaulting to 0.02.</value>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the number of epochs for VAE pretraining.
    /// </summary>
    /// <value>VAE training epochs, defaulting to 100.</value>
    public int VAEEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of epochs for diffusion model training.
    /// </summary>
    /// <value>Diffusion training epochs, defaulting to 100.</value>
    public int DiffusionEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 500.</value>
    public int BatchSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the learning rate for the VAE.
    /// </summary>
    /// <value>The VAE learning rate, defaulting to 1e-3.</value>
    public double VAELearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the learning rate for the diffusion model.
    /// </summary>
    /// <value>The diffusion learning rate, defaulting to 1e-3.</value>
    public double DiffusionLearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column normalization.
    /// </summary>
    /// <value>Number of mixture modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dimension of the timestep embedding for the diffusion model.
    /// </summary>
    /// <value>The timestep embedding dimension, defaulting to 64.</value>
    public int TimestepEmbeddingDimension { get; set; } = 64;

}
