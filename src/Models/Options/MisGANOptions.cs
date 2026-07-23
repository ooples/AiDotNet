namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MisGAN, a GAN for learning from incomplete data with
/// dual generator/discriminator pairs for data and missingness patterns.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MisGAN uses four networks:
/// - <b>Data generator</b>: Generates complete data samples
/// - <b>Mask generator</b>: Generates realistic missingness patterns
/// - <b>Data discriminator</b>: Judges realism of data values
/// - <b>Mask discriminator</b>: Judges realism of missingness patterns
/// </para>
/// <para>
/// <b>For Beginners:</b> MisGAN handles datasets with missing values:
///
/// Real datasets often have missing values (e.g., patients who skip tests).
/// MisGAN learns both:
/// 1. What the data looks like when complete
/// 2. Which values tend to be missing (and why)
///
/// This produces synthetic data with realistic patterns of completeness.
///
/// Example:
/// <code>
/// var options = new MisGANOptions&lt;double&gt;
/// {
///     MissingRate = 0.2,   // 20% of values are missing
///     Epochs = 300
/// };
/// var misgan = new MisGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "MisGAN: Learning from Incomplete Data with GANs" (Li et al., ICLR 2019)
/// </para>
/// </remarks>
public class MisGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the expected rate of missing values in the data.
    /// </summary>
    /// <value>Missing rate, defaulting to 0.2.</value>
    public double MissingRate { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the noise dimension.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer sizes.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] HiddenDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 256.</value>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 2e-4.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column encoding.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both the data discriminator and mask discriminator use dropout
    /// to prevent overfitting. This randomly deactivates neurons during training to force the
    /// networks to learn robust features rather than memorizing specific examples.</para>
    /// </remarks>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double DiscriminatorDropout { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the weight for the WGAN-GP gradient penalty term.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both discriminators use gradient penalty from WGAN-GP to keep
    /// training stable. The penalty constrains the discriminator's gradient norm to be close to 1,
    /// preventing the discriminator from becoming too powerful too quickly.</para>
    /// </remarks>
    /// <value>Gradient penalty weight, defaulting to 10.0.</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the number of discriminator training steps per generator step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both discriminators train multiple times per generator step.
    /// This is standard WGAN practice: the discriminator (critic) must be more well-trained
    /// than the generator for the Wasserstein distance estimate to be meaningful.</para>
    /// </remarks>
    /// <value>Discriminator steps per generator step, defaulting to 5.</value>
    public int DiscriminatorSteps { get; set; } = 5;
}
