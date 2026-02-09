namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TableGAN, a DCGAN-style generative adversarial network for
/// synthesizing tabular data with classification and information loss regularization.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TableGAN uses a DCGAN architecture adapted for tabular data with three loss components:
/// - <b>Adversarial loss</b>: Standard GAN objective for realism
/// - <b>Classification loss</b>: Ensures generated data preserves label column relationships
/// - <b>Information loss</b>: Minimizes statistical divergence between real and synthetic data
/// </para>
/// <para>
/// <b>For Beginners:</b> TableGAN generates fake tabular data using three quality checks:
///
/// 1. "Does it look real?" — the discriminator judges overall realism
/// 2. "Are the labels correct?" — a classifier checks that labels match the data
/// 3. "Are the statistics right?" — mean/variance of synthetic data should match the real data
///
/// This triple-loss approach produces higher quality synthetic data than a standard GAN alone.
///
/// Example:
/// <code>
/// var options = new TableGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 100,
///     LabelColumnIndex = 4,    // Which column is the label
///     Epochs = 300
/// };
/// var tablegan = new TableGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Data Synthesis based on Generative Adversarial Networks" (Park et al., 2018)
/// </para>
/// </remarks>
public class TableGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the random noise vector for the generator.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 100.</value>
    public int EmbeddingDimension { get; set; } = 100;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the generator.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] GeneratorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the discriminator.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] DiscriminatorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the classifier head.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [128].</value>
    public int[] ClassifierDimensions { get; set; } = [128];

    /// <summary>
    /// Gets or sets the index of the label column (for classification loss).
    /// </summary>
    /// <value>Label column index, defaulting to -1 (disabled).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If your data has a target/label column (e.g., "diagnosis" or "fraud"),
    /// set this to its column index. The classification loss will ensure generated data preserves
    /// the relationship between features and the label. Set to -1 to disable.
    /// </para>
    /// </remarks>
    public int LabelColumnIndex { get; set; } = -1;

    /// <summary>
    /// Gets or sets the weight of the classification loss.
    /// </summary>
    /// <value>Classification loss weight, defaulting to 1.0.</value>
    public double ClassificationWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight of the information loss (statistical similarity).
    /// </summary>
    /// <value>Information loss weight, defaulting to 1.0.</value>
    public double InformationWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 500.</value>
    public int BatchSize { get; set; } = 500;

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
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly deactivates a fraction of neurons during training,
    /// which prevents the discriminator from memorizing specific examples and forces it to learn
    /// generalizable features. A value of 0.25 means 25% of neurons are randomly turned off each step.</para>
    /// </remarks>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double DiscriminatorDropout { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the weight for the WGAN-GP gradient penalty term.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient penalty keeps the discriminator's gradients close to 1,
    /// which prevents training instability. A value of 10 is standard from the WGAN-GP paper.
    /// Higher values enforce smoother discriminator outputs but may slow convergence.</para>
    /// </remarks>
    /// <value>Gradient penalty weight, defaulting to 10.0.</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the number of discriminator training steps per generator step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The discriminator trains multiple times for each generator step.
    /// This helps the discriminator stay ahead of the generator, providing better gradient signals.
    /// The standard WGAN recommendation is 5 steps.</para>
    /// </remarks>
    /// <value>Discriminator steps per generator step, defaulting to 5.</value>
    public int DiscriminatorSteps { get; set; } = 5;
}
