namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AutoDiff-Tab, an automated diffusion model for tabular data
/// that searches over diffusion configurations and noise schedules to find optimal settings.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// AutoDiff-Tab automates the design of diffusion models for tabular data by searching over:
/// - <b>Number of diffusion timesteps</b>: How many denoising steps
/// - <b>Beta schedule</b>: Linear, cosine, or learned noise schedule
/// - <b>MLP architecture</b>: Depth, width, and dropout of the denoiser
/// - <b>Noise schedule parameters</b>: Start/end beta values
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoDiff-Tab is like TabDDPM with automatic tuning:
///
/// Instead of manually setting "how noisy" and "how many steps" the diffusion process uses,
/// AutoDiff-Tab tries different settings and picks the best one automatically.
///
/// It explores:
/// 1. Different amounts of noise (noise schedule)
/// 2. Different numbers of denoising steps
/// 3. Different neural network sizes
///
/// Example:
/// <code>
/// var options = new AutoDiffTabOptions&lt;double&gt;
/// {
///     SearchTrials = 5,        // Try 5 different configurations
///     MaxTimesteps = 1000,     // Maximum diffusion steps
///     Epochs = 200
/// };
/// var auto = new AutoDiffTabGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Automated Diffusion Models for Tabular Data" (2024)
/// </para>
/// </remarks>
public class AutoDiffTabOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of search trials to find optimal diffusion configuration.
    /// </summary>
    /// <value>Number of trials, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More trials = better chance of finding a good configuration,
    /// but takes longer. Each trial trains a small model to evaluate performance.
    /// </para>
    /// </remarks>
    public int SearchTrials { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum number of diffusion timesteps to search over.
    /// </summary>
    /// <value>Maximum timesteps, defaulting to 1000.</value>
    public int MaxTimesteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the minimum beta value for the noise schedule.
    /// </summary>
    /// <value>Beta start, defaulting to 1e-4.</value>
    public double BetaStart { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the maximum beta value for the noise schedule.
    /// </summary>
    /// <value>Beta end, defaulting to 0.02.</value>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the type of noise schedule.
    /// </summary>
    /// <value>Schedule type: "linear" or "cosine". Defaults to "linear".</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - "linear": Noise increases evenly over time
    /// - "cosine": Noise increases slowly at first, then faster (often works better)
    /// </para>
    /// </remarks>
    public string BetaSchedule { get; set; } = "linear";

    /// <summary>
    /// Gets or sets the hidden layer dimensions for the denoiser MLP.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] MLPDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the dropout rate for the denoiser.
    /// </summary>
    /// <value>Dropout rate, defaulting to 0.0.</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dimension of timestep embeddings.
    /// </summary>
    /// <value>Timestep embedding dimension, defaulting to 128.</value>
    public int TimestepEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 256.</value>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 200.</value>
    public int Epochs { get; set; } = 200;

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
    /// Gets or sets the number of epochs for each search trial (reduced for speed).
    /// </summary>
    /// <value>Trial epochs, defaulting to 20.</value>
    public int TrialEpochs { get; set; } = 20;
}
