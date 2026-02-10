namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TVAE (Tabular Variational Autoencoder), a VAE-based model
/// for generating realistic synthetic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TVAE uses a variational autoencoder architecture adapted for tabular data:
/// - Same VGM normalization and one-hot encoding as CTGAN for preprocessing
/// - Encoder compresses data into a latent Gaussian distribution (mean, logvar)
/// - Decoder reconstructs data from sampled latent codes via reparameterization trick
/// - ELBO loss balances reconstruction quality with KL divergence regularization
/// </para>
/// <para>
/// <b>For Beginners:</b> TVAE learns to compress your data into a small "summary" (latent space)
/// and then reconstruct it back. Think of it like learning a recipe:
///
/// 1. <b>Encoder</b>: Looks at a data row and writes a compact "recipe" (latent code)
/// 2. <b>Decoder</b>: Reads the recipe and recreates the data row
/// 3. <b>Training</b>: The model learns to write good recipes that can recreate realistic data
/// 4. <b>Generation</b>: Sample random recipes from the latent space and decode them into new rows
///
/// TVAE is often faster to train than CTGAN and works well for moderate-sized datasets.
///
/// Example:
/// <code>
/// var options = new TVAEOptions&lt;double&gt;
/// {
///     EncoderDimensions = new[] { 128, 128 },
///     DecoderDimensions = new[] { 128, 128 },
///     LatentDimension = 128,
///     Epochs = 300
/// };
/// var tvae = new TVAEGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)
/// </para>
/// </remarks>
public class TVAEOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the hidden layer sizes for the encoder network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [128, 128].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The encoder compresses each data row into a latent code.
    /// These dimensions control the "compression pipeline" - wider layers can learn
    /// more complex encodings but take more compute.
    /// </para>
    /// </remarks>
    public int[] EncoderDimensions { get; set; } = [128, 128];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the decoder network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [128, 128].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The decoder takes a latent code and reconstructs the data row.
    /// Often mirrors the encoder dimensions but doesn't have to.
    /// </para>
    /// </remarks>
    public int[] DecoderDimensions { get; set; } = [128, 128];

    /// <summary>
    /// Gets or sets the dimension of the latent space.
    /// </summary>
    /// <value>The latent dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the compact "recipe" the encoder creates.
    /// Smaller values force more compression (simpler patterns), larger values allow
    /// more detail but may overfit on small datasets.
    /// Common values: 32, 64, 128, 256.
    /// </para>
    /// </remarks>
    public int LatentDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 500.</value>
    public int BatchSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate for the optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the weight for reconstruction loss relative to KL divergence.
    /// </summary>
    /// <value>The loss weight, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This balances two competing goals:
    /// - Reconstruction: How accurately the model recreates the input (higher weight = more accurate)
    /// - KL divergence: How "normal" the latent space is (helps generation quality)
    ///
    /// Higher values emphasize reconstruction accuracy, lower values emphasize
    /// smooth latent space (better generation diversity). Default 1.0 is usually fine.
    /// </para>
    /// </remarks>
    public double LossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of Gaussian mixture components for VGM normalization.
    /// </summary>
    /// <value>Number of mixture modes per continuous column, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

}
