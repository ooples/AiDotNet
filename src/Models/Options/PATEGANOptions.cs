namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PATE-GAN, a differentially private GAN that uses the
/// Private Aggregation of Teacher Ensembles (PATE) framework for privacy-preserving
/// synthetic data generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// PATE-GAN achieves differential privacy through a teacher-student framework:
/// - <b>Teacher ensemble</b>: Multiple discriminators, each trained on a disjoint partition of the data
/// - <b>Noisy aggregation</b>: Teacher votes are aggregated with Laplace noise
/// - <b>Student discriminator</b>: Learns from noisy teacher labels, never sees real data directly
/// - <b>Generator</b>: Trained against the student discriminator
/// </para>
/// <para>
/// <b>For Beginners:</b> PATE-GAN provides privacy by never showing the full dataset to any
/// single component:
///
/// 1. Split real data into N groups, give each to a "teacher"
/// 2. Teachers vote on whether generated data looks real or fake
/// 3. Add random noise to the vote tally (for privacy)
/// 4. A "student" learns from the noisy votes, not the real data
/// 5. Generator tries to fool the student
///
/// Because the student never sees real data, and the votes are noisy,
/// no individual's data can be extracted from the generated output.
///
/// Example:
/// <code>
/// var options = new PATEGANOptions&lt;double&gt;
/// {
///     NumTeachers = 10,        // 10 teacher discriminators
///     LaplaceScale = 0.1,      // Noise for privacy
///     Epochs = 300
/// };
/// var pategan = new PATEGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
/// (Jordon et al., ICLR 2019)
/// </para>
/// </remarks>
public class PATEGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of teacher discriminators in the ensemble.
    /// </summary>
    /// <value>Number of teachers, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More teachers = better privacy (each sees less data)
    /// but requires more data overall. Should divide the dataset evenly.
    /// Typical range: 5-50.
    /// </para>
    /// </remarks>
    public int NumTeachers { get; set; } = 10;

    /// <summary>
    /// Gets or sets the Laplace noise scale for the noisy aggregation mechanism.
    /// </summary>
    /// <value>The noise scale, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls the privacy-utility tradeoff:
    /// - Higher values = more privacy but noisier labels for the student
    /// - Lower values = less privacy but better student learning
    /// The privacy guarantee (epsilon) is inversely proportional to this scale.
    /// </para>
    /// </remarks>
    public double LaplaceScale { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the dimension of the random noise vector for the generator.
    /// </summary>
    /// <value>The embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer sizes for the generator network.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] GeneratorDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for each teacher discriminator.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] TeacherDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the hidden layer sizes for the student discriminator.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] StudentDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 64.</value>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-4.</value>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of student discriminator training steps per teacher query.
    /// </summary>
    /// <value>Number of student steps, defaulting to 5.</value>
    public int StudentSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the dropout rate for the student discriminator hidden layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly deactivates neurons during student training,
    /// preventing the student from memorizing the noisy teacher labels and forcing it to learn
    /// generalizable features of real vs. fake data.</para>
    /// </remarks>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double StudentDropout { get; set; } = 0.25;
}
