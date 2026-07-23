namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Causal-GAN, a GAN that learns causal graph structure
/// and generates data respecting causal relationships between features.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Causal-GAN discovers and uses causal structure:
/// - <b>DAG structure learning</b>: NOTEARS-style continuous relaxation for learning a directed acyclic graph
/// - <b>Structural equation models</b>: Each feature is generated as a function of its causal parents
/// - <b>Interventional generation</b>: Can simulate "what-if" scenarios by intervening on specific features
/// </para>
/// <para>
/// <b>For Beginners:</b> Causal-GAN learns which features cause other features:
///
/// Instead of just learning correlations (e.g., "Age and Income are related"),
/// it learns causation (e.g., "Education causes higher Income").
///
/// This allows:
/// 1. Generating more realistic data (respecting cause-effect chains)
/// 2. Answering "what if" questions ("what if everyone had a college degree?")
///
/// Example:
/// <code>
/// var options = new CausalGANOptions&lt;double&gt;
/// {
///     DAGPenaltyWeight = 0.5,
///     Epochs = 300
/// };
/// var causalgan = new CausalGANGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class CausalGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the weight for the DAG acyclicity penalty (NOTEARS constraint).
    /// </summary>
    /// <value>DAG penalty weight, defaulting to 0.5.</value>
    public double DAGPenaltyWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the sparsity weight for the learned adjacency matrix.
    /// </summary>
    /// <value>Sparsity weight, defaulting to 0.1.</value>
    public double SparsityWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the noise dimension.
    /// </summary>
    /// <value>Embedding dimension, defaulting to 128.</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer sizes.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256].</value>
    public int[] HiddenDimensions { get; set; } = [256, 256];

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 256.</value>
    public int BatchSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Number of epochs, defaulting to 300.</value>
    public int Epochs { get; set; } = 300;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous column encoding.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for discriminator hidden layers.
    /// </summary>
    /// <value>Dropout probability, defaulting to 0.25.</value>
    public double DiscriminatorDropout { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the weight for the WGAN-GP gradient penalty term.
    /// </summary>
    /// <value>Gradient penalty weight, defaulting to 10.0.</value>
    public double GradientPenaltyWeight { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the number of discriminator training steps per generator step.
    /// </summary>
    /// <value>Discriminator steps per generator step, defaulting to 5.</value>
    public int DiscriminatorSteps { get; set; } = 5;
}
