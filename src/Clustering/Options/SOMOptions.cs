using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Self-Organizing Maps (SOM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Self-Organizing Maps are a type of neural network that produces a low-dimensional
/// (typically 2D) discretized representation of the input space. They preserve
/// topological properties of the input, meaning similar inputs map to nearby neurons.
/// </para>
/// <para><b>For Beginners:</b> SOM creates a "map" of your data.
///
/// Imagine compressing a 3D world onto a 2D map:
/// - Nearby countries on the map should be nearby in reality
/// - The map preserves relationships even though it's lower dimensional
///
/// SOM does this for high-dimensional data:
/// - Creates a 2D grid of "neurons"
/// - Each neuron represents a prototype pattern
/// - Similar data points activate nearby neurons
///
/// Uses:
/// - Visualization of high-dimensional data
/// - Dimensionality reduction that preserves topology
/// - Finding natural groupings in data
/// </para>
/// </remarks>
public class SOMOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes SOMOptions with appropriate defaults.
    /// </summary>
    public SOMOptions()
    {
        MaxIterations = 1000;
    }

    /// <summary>
    /// Gets or sets the width of the SOM grid.
    /// </summary>
    /// <value>Grid width. Default is 10.</value>
    public int GridWidth { get; set; } = 10;

    /// <summary>
    /// Gets or sets the height of the SOM grid.
    /// </summary>
    /// <value>Grid height. Default is 10.</value>
    public int GridHeight { get; set; } = 10;

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    /// <value>Initial learning rate. Default is 0.5.</value>
    /// <remarks>
    /// <para>
    /// Learning rate controls how much neurons adjust toward input patterns.
    /// It typically decreases over training time.
    /// </para>
    /// </remarks>
    public double InitialLearningRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the initial neighborhood radius.
    /// </summary>
    /// <value>Initial radius. Default is -1 (auto-calculated).</value>
    /// <remarks>
    /// <para>
    /// If -1, the initial radius is set to max(GridWidth, GridHeight) / 2.
    /// The neighborhood radius determines how many nearby neurons are
    /// updated when a neuron "wins" for an input.
    /// </para>
    /// </remarks>
    public double InitialNeighborhoodRadius { get; set; } = -1;

    /// <summary>
    /// Gets or sets the neighborhood function type.
    /// </summary>
    /// <value>Neighborhood function. Default is Gaussian.</value>
    public NeighborhoodFunction NeighborhoodType { get; set; } = NeighborhoodFunction.Gaussian;

    /// <summary>
    /// Gets or sets the topology of the SOM grid.
    /// </summary>
    /// <value>Grid topology. Default is Rectangular.</value>
    public SOMTopology Topology { get; set; } = SOMTopology.Rectangular;

    /// <summary>
    /// Gets or sets the distance metric for input space.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Neighborhood function types for SOM.
/// </summary>
public enum NeighborhoodFunction
{
    /// <summary>
    /// Gaussian neighborhood function with smooth decay.
    /// </summary>
    Gaussian,

    /// <summary>
    /// Bubble neighborhood with sharp cutoff at radius.
    /// </summary>
    Bubble,

    /// <summary>
    /// Mexican hat (difference of Gaussians) for lateral inhibition.
    /// </summary>
    MexicanHat
}

/// <summary>
/// SOM grid topology types.
/// </summary>
public enum SOMTopology
{
    /// <summary>
    /// Standard rectangular grid.
    /// </summary>
    Rectangular,

    /// <summary>
    /// Hexagonal grid (6 neighbors per neuron).
    /// </summary>
    Hexagonal,

    /// <summary>
    /// Toroidal topology (wraps at edges).
    /// </summary>
    Toroidal
}
