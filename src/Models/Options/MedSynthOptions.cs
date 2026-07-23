namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MedSynth, a privacy-preserving medical tabular data synthesis
/// model combining a VAE/GAN hybrid with clinical validity constraints.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MedSynth is specialized for medical/health data generation with:
/// - <b>Clinical validity constraints</b>: Ensures generated values fall within valid clinical ranges
/// - <b>Differential privacy</b>: Optional epsilon-differential privacy via gradient clipping and noise
/// - <b>Referential integrity</b>: Maintains consistency between related medical fields
/// - <b>Constraint satisfaction layer</b>: Post-processing to enforce domain rules
/// </para>
/// <para>
/// <b>For Beginners:</b> MedSynth generates fake patient data that is:
///
/// 1. <b>Clinically valid</b>: Lab values are within normal/plausible ranges
/// 2. <b>Internally consistent</b>: Related fields make sense together
///    (e.g., a 5-year-old won't have adult blood pressure values)
/// 3. <b>Private</b>: No real patient's data can be extracted
///
/// Example:
/// <code>
/// var options = new MedSynthOptions&lt;double&gt;
/// {
///     LatentDimension = 64,
///     EnablePrivacy = true,
///     Epsilon = 3.0,
///     Epochs = 500
/// };
/// var medsynth = new MedSynthGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class MedSynthOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the VAE latent space dimension.
    /// </summary>
    /// <value>Latent dimension, defaulting to 64.</value>
    public int LatentDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the encoder/decoder.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] EncoderDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the discriminator.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] DiscriminatorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets whether to enable differential privacy.
    /// </summary>
    /// <value>True to enable privacy; defaults to false.</value>
    public bool EnablePrivacy { get; set; } = false;

    /// <summary>
    /// Gets or sets the privacy budget (epsilon) when privacy is enabled.
    /// </summary>
    /// <value>Epsilon, defaulting to 3.0.</value>
    public double Epsilon { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the gradient clipping norm for differential privacy.
    /// </summary>
    /// <value>Clip norm, defaulting to 1.0.</value>
    public double ClipNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the KL divergence weight.
    /// </summary>
    /// <value>KL weight, defaulting to 0.1.</value>
    public double KLWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the adversarial loss weight.
    /// </summary>
    /// <value>Adversarial weight, defaulting to 1.0.</value>
    public double AdversarialWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the constraint violation penalty weight.
    /// </summary>
    /// <value>Constraint weight, defaulting to 5.0.</value>
    public double ConstraintWeight { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 256.</value>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 500.</value>
    public int Epochs { get; set; } = 500;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly deactivates neurons during discriminator training,
    /// preventing it from memorizing specific examples and forcing it to learn generalizable features.
    /// A value of 0.25 means 25% of neurons are randomly turned off each step.</para>
    /// </remarks>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double DiscriminatorDropout { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the number of discriminator training steps per VAE/generator step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The discriminator trains multiple times for each generator step.
    /// This helps it stay ahead and provide meaningful gradient signals to the decoder/generator.</para>
    /// </remarks>
    /// <value>Discriminator steps per generator step, defaulting to 3.</value>
    public int DiscriminatorSteps { get; set; } = 3;
}
