namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CTAB-GAN+, an enhanced conditional tabular GAN with auxiliary
/// classifier discriminator and mixed-type encoder for high-quality synthetic data generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CTAB-GAN+ extends CTGAN with several architectural improvements:
/// - <b>Auxiliary Classifier GAN (ACGAN)</b>: Discriminator also predicts class labels
/// - <b>Mixed-type encoder</b>: Log-frequency encoding for long-tail categoricals
/// - <b>Downstream losses</b>: Additional classification/regression losses for utility
/// - <b>Conditional vector</b>: Same training-by-sampling as CTGAN
/// </para>
/// <para>
/// <b>For Beginners:</b> CTAB-GAN+ is an improved version of CTGAN that produces
/// higher-quality synthetic data by using smarter training signals:
///
/// 1. <b>Better discriminator</b>: Not only decides real/fake, but also classifies data categories
/// 2. <b>Better encoding</b>: Handles rare categories better with log-frequency encoding
/// 3. <b>Better evaluation</b>: Checks if generated data is useful for downstream ML tasks
///
/// Example:
/// <code>
/// var options = new CTABGANPlusOptions&lt;double&gt;
/// {
///     GeneratorDimensions = new[] { 256, 256 },
///     DiscriminatorDimensions = new[] { 256, 256 },
///     ClassifierWeight = 0.5,
///     Epochs = 300
/// };
/// var ctabgan = new CTABGANPlusGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "CTAB-GAN: Effective Table Data Synthesizing" (Zhao et al., 2021)
/// </para>
/// </remarks>
public class CTABGANPlusOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the random noise vector fed to the generator.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the generator network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] GeneratorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the discriminator network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] DiscriminatorDimensions { get; set; } = [256, 256];

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
    /// Gets or sets the number of discriminator training steps per generator step.
    /// </summary>
    /// <value>Discriminator steps per generator step, defaulting to 1.</value>
    public int DiscriminatorSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate for both generator and discriminator.
    /// </summary>
    /// <value>The learning rate, defaulting to 2e-4.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the weight for the auxiliary classifier loss in the discriminator.
    /// </summary>
    /// <value>The classifier loss weight, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much the discriminator cares about
    /// correctly classifying data categories vs just detecting real/fake.
    /// Higher values produce more class-consistent synthetic data.
    /// </para>
    /// </remarks>
    public double ClassifierWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight for the information loss (statistical similarity).
    /// </summary>
    /// <value>The information loss weight, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This penalizes the generator if its output statistics
    /// (mean, variance) drift too far from the real data's statistics.
    /// </para>
    /// </remarks>
    public double InformationWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the gradient penalty weight for WGAN-GP.
    /// </summary>
    /// <value>The gradient penalty coefficient, defaulting to 10.0.</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column normalization.
    /// </summary>
    /// <value>Number of mixture modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for the discriminator.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.5.</value>
    public double DiscriminatorDropout { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the index of the target/label column for the auxiliary classifier.
    /// </summary>
    /// <value>The target column index, or -1 to auto-detect the first categorical column.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The auxiliary classifier needs a "target" column to classify.
    /// Set this to the index of your label column. If -1, the first categorical column is used.
    /// </para>
    /// </remarks>
    public int TargetColumnIndex { get; set; } = -1;

    /// <summary>
    /// Gets or sets the PacGAN packing size.
    /// </summary>
    /// <value>The packing size, defaulting to 10.</value>
    public int PacSize { get; set; } = 10;

}
