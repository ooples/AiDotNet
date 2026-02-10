namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CopulaGAN, a synthetic tabular data generator that combines
/// Gaussian copula transformations with the CTGAN training pipeline.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CopulaGAN extends CTGAN by first transforming continuous columns using Gaussian copulas.
/// This maps each continuous column's distribution to a standard normal via CDF-then-quantile,
/// making the CTGAN generator's job easier and often improving generation quality for
/// columns with complex or skewed distributions.
/// </para>
/// <para>
/// <b>For Beginners:</b> CopulaGAN improves on CTGAN by "normalizing" each numerical column
/// before training. Think of it like this:
///
/// 1. <b>Copula Transform</b>: Each number column is reshaped to look like a bell curve
///    (Gaussian distribution) using a mathematical trick called a "copula."
/// 2. <b>CTGAN Training</b>: The standard CTGAN generator/discriminator pipeline trains
///    on this nicely-shaped data, which is easier to learn.
/// 3. <b>Inverse Transform</b>: Generated data is converted back to the original
///    distribution shapes.
///
/// This often produces better results than plain CTGAN for columns with unusual distributions
/// (e.g., heavily skewed income data, bimodal distributions).
///
/// Example:
/// <code>
/// var options = new CopulaGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     GeneratorDimensions = new[] { 256, 256 },
///     DiscriminatorDimensions = new[] { 256, 256 },
///     BatchSize = 500,
///     Epochs = 300
/// };
/// var copulaGan = new CopulaGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Synthesizing Tabular Data using Copulas" (2020)
/// </para>
/// </remarks>
public class CopulaGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the random noise vector fed to the generator.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The generator starts with random noise of this size and
    /// transforms it into a fake data row. Common values: 64, 128, 256.
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
    /// for having steep gradients. The standard value is 10.0.
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
    /// to prevent mode collapse. The batch size must be divisible by this value.
    /// </para>
    /// </remarks>
    public int PacSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of Gaussian mixture components for VGM normalization.
    /// </summary>
    /// <value>Number of mixture modes per continuous column, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for the discriminator.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.5.</value>
    public double DiscriminatorDropout { get; set; } = 0.5;

}
