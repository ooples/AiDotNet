namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the MeshCNN neural network.
/// </summary>
/// <remarks>
/// <para>
/// MeshCNN is a deep learning architecture for processing 3D mesh data. It operates
/// directly on the mesh structure using edge convolutions and mesh pooling operations.
/// </para>
/// <para><b>For Beginners:</b> These options control how the MeshCNN network is configured.
/// The defaults are set to match the original paper and work well for most 3D shape
/// classification and segmentation tasks.
/// </para>
/// </remarks>
public class MeshCNNOptions
{
    /// <summary>
    /// Gets or sets the number of output classes for classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For ModelNet40, this would be 40. For SHREC, this would be 30.
    /// </para>
    /// </remarks>
    public int NumClasses { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of input features per edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default value of 5 corresponds to the standard MeshCNN edge features:
    /// - Dihedral angle
    /// - Two symmetric edge-length ratios
    /// - Two symmetric face angles
    /// </para>
    /// </remarks>
    public int InputFeatures { get; set; } = 5;

    /// <summary>
    /// Gets or sets the channel sizes for each edge convolution block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each value represents the number of output channels for one edge conv layer.
    /// The default [64, 128, 256, 256] matches the original MeshCNN paper.
    /// </para>
    /// </remarks>
    public int[] ConvChannels { get; set; } = [64, 128, 256, 256];

    /// <summary>
    /// Gets or sets the target edge counts after each pooling operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each pooling layer reduces the number of edges to the specified target.
    /// Should have one fewer element than ConvChannels (pooling after each conv except last).
    /// </para>
    /// </remarks>
    public int[] PoolTargets { get; set; } = [1800, 1350, 600];

    /// <summary>
    /// Gets or sets the sizes of fully connected layers before output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After edge convolutions and pooling, the features are aggregated and passed
    /// through fully connected layers for classification.
    /// </para>
    /// </remarks>
    public int[] FullyConnectedSizes { get; set; } = [100];

    /// <summary>
    /// Gets or sets the number of neighboring edges to consider for each edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In a triangular mesh, each edge has 4 neighboring edges (2 from each adjacent face).
    /// This is the standard value and should not be changed unless using a different mesh type.
    /// </para>
    /// </remarks>
    public int NumNeighbors { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to use batch normalization after each conv layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Batch normalization can help training stability and convergence speed.
    /// </para>
    /// </remarks>
    public bool UseBatchNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout is applied to fully connected layers to prevent overfitting.
    /// A value of 0 disables dropout.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the initial learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets whether to use global average pooling before FC layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If false, uses global max pooling instead.
    /// </para>
    /// </remarks>
    public bool UseGlobalAveragePooling { get; set; } = false;
}
