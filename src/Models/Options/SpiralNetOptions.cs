namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SpiralNet++ mesh neural network.
/// </summary>
/// <remarks>
/// <para>
/// SpiralNet++ is a mesh convolution architecture that uses spiral sequences
/// to define consistent neighbor orderings on irregular mesh vertices.
/// </para>
/// <para><b>For Beginners:</b> These options control how the SpiralNet++ network
/// processes 3D mesh data. Key settings include:
/// - SpiralLength: How many neighbors to consider for each vertex
/// - ConvChannels: Feature sizes at each layer
/// - PoolRatios: How much to simplify the mesh at each pooling step
/// </para>
/// </remarks>
public class SpiralNetOptions
{
    /// <summary>
    /// Gets or sets the number of output classes for classification.
    /// </summary>
    /// <value>Default is 40 (ModelNet40 classes).</value>
    public int NumClasses { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of input features per vertex.
    /// </summary>
    /// <value>Default is 3 (x, y, z coordinates).</value>
    public int InputFeatures { get; set; } = 3;

    /// <summary>
    /// Gets or sets the length of the spiral sequence for convolutions.
    /// </summary>
    /// <value>Default is 9 neighbors per spiral.</value>
    public int SpiralLength { get; set; } = 9;

    /// <summary>
    /// Gets or sets the channel sizes for each convolution layer.
    /// </summary>
    /// <value>Default is [32, 64, 128, 256].</value>
    public int[] ConvChannels { get; set; } = [32, 64, 128, 256];

    /// <summary>
    /// Gets or sets the pooling ratios for mesh simplification.
    /// </summary>
    /// <value>Default is [0.5, 0.5] for two pooling layers.</value>
    /// <remarks>
    /// <para>
    /// Each ratio specifies what fraction of vertices to keep after pooling.
    /// A ratio of 0.5 keeps half the vertices.
    /// </para>
    /// </remarks>
    public double[] PoolRatios { get; set; } = [0.5, 0.5];

    /// <summary>
    /// Gets or sets the sizes of fully connected layers before output.
    /// </summary>
    /// <value>Default is [256, 128].</value>
    public int[] FullyConnectedSizes { get; set; } = [256, 128];

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseBatchNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Default is 0.5.</value>
    public double DropoutRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use global average pooling before classification.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseGlobalAveragePooling { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include vertex coordinates as input features.
    /// </summary>
    /// <value>Default is true.</value>
    public bool IncludeCoordinates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include vertex normals as input features.
    /// </summary>
    /// <value>Default is false.</value>
    public bool IncludeNormals { get; set; } = false;
}
