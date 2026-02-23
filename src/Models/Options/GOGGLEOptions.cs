namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GOGGLE (Generative mOdelling for tabular data by learninG
/// reLational structurE), a graph-based VAE that learns feature dependency structure
/// for generating realistic tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// GOGGLE combines a VAE with a graph neural network (GNN) to learn and exploit
/// feature dependencies:
/// - <b>Structure learning</b>: Learns a soft adjacency matrix representing feature dependencies
/// - <b>GNN encoder</b>: Uses message passing on the learned graph to encode features
/// - <b>VAE framework</b>: Generates data through a latent space with reparameterization
/// - <b>Structure loss</b>: Regularizes the learned graph for sparsity and DAG properties
/// </para>
/// <para>
/// <b>For Beginners:</b> GOGGLE discovers which features depend on each other and uses
/// this knowledge to generate better data:
///
/// 1. Learns a "dependency map" — e.g., "Income depends on Education and Age"
/// 2. Features that are connected share information through graph neural networks
/// 3. This produces data where feature relationships are more realistic
///
/// Example:
/// <code>
/// var options = new GOGGLEOptions&lt;double&gt;
/// {
///     LatentDimension = 32,
///     NumGNNLayers = 2,
///     Epochs = 300
/// };
/// var goggle = new GOGGLEGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure"
/// (Liu et al., ICLR 2023)
/// </para>
/// </remarks>
public class GOGGLEOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the VAE latent space.
    /// </summary>
    /// <value>Latent dimension, defaulting to 32.</value>
    public int LatentDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of GNN message-passing layers.
    /// </summary>
    /// <value>Number of GNN layers, defaulting to 2.</value>
    public int NumGNNLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the hidden dimension for GNN and MLP layers.
    /// </summary>
    /// <value>Hidden dimension, defaulting to 128.</value>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the weight for the KL divergence loss term.
    /// </summary>
    /// <value>KL weight, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls the balance between reconstruction quality and
    /// regularity of the latent space. Lower values prioritize reconstruction;
    /// higher values produce a smoother latent space.
    /// </para>
    /// </remarks>
    public double KLWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the weight for the graph structure learning loss.
    /// </summary>
    /// <value>Structure loss weight, defaulting to 0.01.</value>
    public double StructureWeight { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the sparsity regularization for the adjacency matrix.
    /// </summary>
    /// <value>Sparsity weight, defaulting to 0.1.</value>
    public double SparsityWeight { get; set; } = 0.1;

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
    /// Gets or sets the number of VGM modes for continuous column transformation.
    /// </summary>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;
}
