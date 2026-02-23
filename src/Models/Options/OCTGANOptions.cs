namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for OCT-GAN (One-Class Tabular GAN), designed for generating
/// synthetic data with a focus on minority/imbalanced classes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// OCT-GAN addresses class imbalance by using a one-class discriminator that focuses
/// on the minority class characteristics, making it ideal for oversampling.
/// </para>
/// <para>
/// <b>For Beginners:</b> OCT-GAN is designed for imbalanced datasets (e.g., fraud detection
/// where 99% of transactions are normal and only 1% are fraud).
///
/// Instead of generating all data equally, it focuses on the minority class:
/// 1. The discriminator learns what minority samples look like
/// 2. The generator tries to create realistic minority samples
/// 3. This produces better synthetic oversampling than random duplication
///
/// Example:
/// <code>
/// var options = new OCTGANOptions&lt;double&gt;
/// {
///     MinorityClassValue = 1,     // Which class is the minority
///     LabelColumnIndex = 4,       // Which column contains the label
///     Epochs = 300
/// };
/// var octgan = new OCTGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class OCTGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the index of the label column.
    /// </summary>
    /// <value>Label column index, defaulting to -1 (auto-detect last column).</value>
    public int LabelColumnIndex { get; set; } = -1;

    /// <summary>
    /// Gets or sets the minority class value to focus on.
    /// </summary>
    /// <value>Minority class value, defaulting to 1.</value>
    public int MinorityClassValue { get; set; } = 1;

    /// <summary>
    /// Gets or sets the dimension of the noise vector.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

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
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 128.</value>
    public int BatchSize { get; set; } = 128;

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
    /// <remarks>
    /// <para><b>For Beginners:</b> VGM (Variational Gaussian Mixture) splits each numeric column
    /// into a mixture of bell curves, which helps the GAN learn multi-modal distributions
    /// (columns with multiple peaks, like age distributions with clusters at 25 and 55).</para>
    /// </remarks>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly deactivates a fraction of neurons during training,
    /// which prevents the discriminator from memorizing specific examples and forces it to learn
    /// generalizable features of the minority class boundary.</para>
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
    /// This helps the discriminator stay ahead, providing better gradient signals to the generator.
    /// Too many steps waste computation; too few lead to poor generator updates.</para>
    /// </remarks>
    /// <value>Discriminator steps per generator step, defaulting to 5.</value>
    public int DiscriminatorSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the dimension of the discriminator's embedding space for the SVDD objective.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The discriminator maps each data sample into a compact embedding space.
    /// The SVDD (Support Vector Data Description) objective then measures how close each embedding is
    /// to a learned center point. Real minority data should cluster near the center, while fake data
    /// should be pushed away. This dimension controls the size of that embedding space.</para>
    /// </remarks>
    /// <value>SVDD embedding dimension, defaulting to 64.</value>
    public int SVDDEmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the momentum for exponential moving average updates of the SVDD center.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The center of the hypersphere is updated slowly during training
    /// using an exponential moving average. A small value (0.01) means the center moves slowly and is
    /// more stable; a larger value makes it adapt faster to new data but can be less stable.</para>
    /// </remarks>
    /// <value>Center update momentum, defaulting to 0.01.</value>
    public double CenterUpdateMomentum { get; set; } = 0.01;
}
