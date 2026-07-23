namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DP-CTGAN, a differentially private version of CTGAN that provides
/// formal privacy guarantees while generating synthetic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// DP-CTGAN adds differential privacy to CTGAN through:
/// - <b>Per-sample gradient clipping</b>: Bounds the sensitivity of each training sample
/// - <b>Gaussian noise injection</b>: Adds calibrated noise to clipped gradients
/// - <b>Privacy accountant</b>: Tracks cumulative privacy loss (epsilon)
/// </para>
/// <para>
/// <b>For Beginners:</b> DP-CTGAN generates fake data while mathematically guaranteeing
/// that the synthetic data doesn't reveal too much about any individual in the real data.
///
/// Think of it like CTGAN with a "privacy filter":
/// 1. During training, gradients are clipped (bounded) so no single person's data
///    has too much influence on the model
/// 2. Random noise is added to further obscure individual contributions
/// 3. A "privacy budget" (epsilon) tracks how much privacy has been spent
///
/// Lower epsilon = more privacy but lower data quality.
/// Typical values: epsilon 1-10 for reasonable utility.
///
/// Example:
/// <code>
/// var options = new DPCTGANOptions&lt;double&gt;
/// {
///     Epsilon = 3.0,        // Privacy budget
///     Delta = 1e-5,         // Failure probability
///     ClipNorm = 1.0,       // Gradient clip bound
///     Epochs = 300
/// };
/// var dpctgan = new DPCTGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class DPCTGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the total privacy budget (epsilon) for the training process.
    /// </summary>
    /// <value>The epsilon value, defaulting to 3.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Epsilon controls the privacy-utility tradeoff:
    /// - Lower epsilon (0.1-1.0) = strong privacy, lower quality
    /// - Medium epsilon (1.0-10.0) = good balance
    /// - Higher epsilon (10.0+) = weak privacy, high quality
    /// A value of 3.0 provides reasonable privacy for most applications.
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the delta parameter for (epsilon, delta)-differential privacy.
    /// </summary>
    /// <value>The delta value, defaulting to 1e-5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Delta is the probability that the privacy guarantee fails.
    /// Should be much smaller than 1/n where n is the dataset size.
    /// Default of 1e-5 is safe for datasets up to ~100,000 rows.
    /// </para>
    /// </remarks>
    public double Delta { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the L2 norm clipping bound for per-sample gradients.
    /// </summary>
    /// <value>The clipping norm, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This limits how much any single training sample can
    /// influence the model. Lower values = more privacy but slower learning.
    /// </para>
    /// </remarks>
    public double ClipNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the noise multiplier for Gaussian mechanism.
    /// </summary>
    /// <value>The noise multiplier, defaulting to 0 (auto-computed from epsilon/delta).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If set to 0, the noise level is automatically calculated
    /// from epsilon and delta. Set manually only if you know what you're doing.
    /// </para>
    /// </remarks>
    public double NoiseMultiplier { get; set; } = 0;

    /// <summary>
    /// Gets or sets the dimension of the random noise vector.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
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
    /// <value>The batch size, defaulting to 500.</value>
    public int BatchSize { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 2e-4.</value>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Gets or sets the gradient penalty weight.
    /// </summary>
    /// <value>The penalty weight, defaulting to 10.0.</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the PacGAN packing size.
    /// </summary>
    /// <value>The packing size, defaulting to 10.</value>
    public int PacSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of VGM modes.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the discriminator dropout rate.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.5.</value>
    public double DiscriminatorDropout { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of discriminator steps per generator step.
    /// </summary>
    /// <value>Discriminator steps, defaulting to 1.</value>
    public int DiscriminatorSteps { get; set; } = 1;

}
