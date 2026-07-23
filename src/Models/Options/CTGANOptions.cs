namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CTGAN (Conditional Tabular GAN), a generative adversarial network
/// specifically designed for generating realistic synthetic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CTGAN uses a conditional GAN architecture with several innovations for tabular data:
/// - Variational Gaussian Mixture (VGM) mode-specific normalization for continuous columns
/// - Training-by-sampling with conditional vectors for handling imbalanced categories
/// - WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training
/// - PacGAN packing to prevent mode collapse
/// </para>
/// <para>
/// <b>For Beginners:</b> CTGAN is like having two neural networks compete to create fake data:
///
/// - The <b>Generator</b> creates fake rows of data from random noise
/// - The <b>Discriminator</b> tries to distinguish real rows from fake ones
/// - As they compete, the generator learns to produce increasingly realistic data
///
/// Special features for tabular data:
/// - Handles mixed data types (numbers and categories) in the same table
/// - Uses conditional generation to ensure rare categories are well-represented
/// - Uses Wasserstein loss with gradient penalty for stable training
///
/// Example:
/// <code>
/// var options = new CTGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     GeneratorDimensions = new[] { 256, 256 },
///     DiscriminatorDimensions = new[] { 256, 256 },
///     BatchSize = 500,
///     Epochs = 300
/// };
/// var ctgan = new CTGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)
/// </para>
/// </remarks>
public class CTGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the random noise vector fed to the generator.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The generator starts with random noise of this size and
    /// transforms it into a fake data row. Larger values give the generator more
    /// "raw material" to work with, but too large can make training harder.
    /// Common values: 64, 128, 256.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the generator network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These define the "width" of each hidden layer in the generator.
    /// More/wider layers can learn more complex patterns but take longer to train.
    /// The generator uses residual connections between layers.
    /// </para>
    /// </remarks>
    public int[] GeneratorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the discriminator network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These define the "width" of each hidden layer in the discriminator.
    /// The discriminator uses dropout and LeakyReLU activation for regularization.
    /// </para>
    /// </remarks>
    public int[] DiscriminatorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 500.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many rows are processed together in each training step.
    /// Larger batches give more stable gradients but require more memory.
    /// Must be divisible by <see cref="PacSize"/>.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the number of discriminator training steps per generator step.
    /// </summary>
    /// <value>Discriminator steps per generator step, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training the discriminator more often than the generator
    /// can improve training stability. The original CTGAN paper uses 1.
    /// For WGAN-GP, values of 1-5 are common.
    /// </para>
    /// </remarks>
    public int DiscriminatorSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate for both generator and discriminator optimizers.
    /// </summary>
    /// <value>The learning rate, defaulting to 2e-4.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the gradient penalty weight (lambda) for WGAN-GP.
    /// </summary>
    /// <value>The gradient penalty coefficient, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how strongly the discriminator is penalized
    /// for having steep gradients. It enforces the Lipschitz constraint that makes
    /// Wasserstein GAN training stable. The standard value is 10.0.
    /// </para>
    /// </remarks>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the PacGAN packing size.
    /// </summary>
    /// <value>The packing size, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PacGAN feeds multiple samples to the discriminator at once
    /// (packed together) to prevent mode collapse - where the generator only learns
    /// to produce one type of row. The batch size must be divisible by this value.
    /// </para>
    /// </remarks>
    public int PacSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of Gaussian mixture components for VGM normalization.
    /// </summary>
    /// <value>Number of mixture modes per continuous column, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each continuous column's distribution is approximated as a
    /// mixture of this many Gaussian (bell curve) components. More modes can capture
    /// more complex distributions but increase the transformed data width.
    /// </para>
    /// </remarks>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for the discriminator.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.5.</value>
    public double DiscriminatorDropout { get; set; } = 0.5;

}
