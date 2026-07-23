namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TimeGAN, a generative adversarial network designed specifically
/// for generating realistic time-series data while preserving temporal dynamics.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TimeGAN combines autoencoding and adversarial training in a shared latent space.
/// It uses five components:
/// - <b>Embedder</b>: Maps real data to a latent embedding space
/// - <b>Recovery</b>: Reconstructs data from the latent space
/// - <b>Generator</b>: Produces synthetic latent embeddings
/// - <b>Supervisor</b>: Learns temporal dynamics in the latent space
/// - <b>Discriminator</b>: Distinguishes real from synthetic embeddings
/// </para>
/// <para>
/// <b>For Beginners:</b> TimeGAN generates fake time-series data that looks realistic.
///
/// Regular GANs struggle with sequential data because they don't understand "time."
/// TimeGAN solves this with a clever multi-step training approach:
///
/// 1. <b>Embedding training</b>: Learn to compress real data into a simpler form
/// 2. <b>Supervised training</b>: Learn the rules of how data changes over time
/// 3. <b>Joint training</b>: Train everything together — generator, discriminator,
///    embedder, recovery, and supervisor — all at once
///
/// Example:
/// <code>
/// var options = new TimeGANOptions&lt;double&gt;
/// {
///     SequenceLength = 24,      // Each sample is 24 time steps
///     HiddenDimension = 64,     // Size of hidden states
///     NumFeatures = 5,          // 5 features per time step
///     Epochs = 2000
/// };
/// var timegan = new TimeGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)
/// </para>
/// </remarks>
public class TimeGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the length of each time-series sequence.
    /// </summary>
    /// <value>Sequence length, defaulting to 24.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many time steps are in each training sample.
    /// For example, 24 for hourly data covering one day, or 30 for daily data covering one month.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of features per time step.
    /// </summary>
    /// <value>Number of features, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many measurements at each time step.
    /// For stock data this might be 5: open, high, low, close, volume.
    /// This overrides the inherited NumFeatures with a different default value.
    /// </para>
    /// </remarks>
    public new int NumFeatures { get; set; } = 5;

    /// <summary>
    /// Gets or sets the hidden dimension for the RNN components.
    /// </summary>
    /// <value>Hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The size of the internal memory in each recurrent network.
    /// Larger values capture more complex patterns but train slower.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of RNN layers in each component.
    /// </summary>
    /// <value>Number of layers, defaulting to 3.</value>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 128.</value>
    public int BatchSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the total number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 2000.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeGAN needs many epochs because training has three phases:
    /// embedding pretraining, supervised pretraining, and joint training.
    /// Each phase gets roughly 1/3 of the total epochs.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 5e-4.</value>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets the weight for the supervised loss in the generator objective.
    /// </summary>
    /// <value>Supervised loss weight, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much the generator cares about getting
    /// the temporal dynamics right versus fooling the discriminator.
    /// Higher values = more realistic time patterns.
    /// </para>
    /// </remarks>
    public double SupervisedWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the weight for the reconstruction loss.
    /// </summary>
    /// <value>Reconstruction loss weight, defaulting to 10.0.</value>
    public double ReconstructionWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly deactivates neurons during discriminator training,
    /// preventing it from memorizing specific examples. A value of 0.25 means 25% of neurons are
    /// randomly turned off each training step.</para>
    /// </remarks>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double DiscriminatorDropout { get; set; } = 0.25;
}
