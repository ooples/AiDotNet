using AiDotNet.Clustering.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Base configuration options for clustering algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class provides common configuration options shared by most clustering algorithms.
/// Specific clustering implementations may extend this with algorithm-specific options.
/// Inherits from ModelOptions to provide the standard Seed property for reproducibility.
/// </para>
/// <para><b>For Beginners:</b> These are the settings you can adjust to control
/// how the clustering algorithm works.
///
/// Common options include:
/// - How many iterations to run
/// - When to stop (convergence threshold)
/// - Random seed for reproducibility (inherited Seed property)
/// - Which distance metric to use
/// </para>
/// </remarks>
public class ClusteringOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    /// <value>The maximum iterations. Default is 300.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Clustering algorithms iterate (repeat steps) until they
    /// find a good solution or reach this limit. Higher values allow more refinement
    /// but take longer to run.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 300;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>The tolerance for declaring convergence. Default is 1e-4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When changes between iterations become smaller than
    /// this value, the algorithm stops. Smaller values mean more precise results
    /// but potentially more iterations.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the number of times to run with different random initializations.
    /// </summary>
    /// <value>Number of initializations. Default is 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since random starting points can lead to different
    /// results, running multiple times and keeping the best result helps find
    /// a better solution. Higher values improve quality but take longer.
    /// </para>
    /// </remarks>
    public int NumInitializations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the verbosity level for logging.
    /// </summary>
    /// <value>0 for silent, 1 for progress, 2 for detailed. Default is 0.</value>
    public int Verbose { get; set; } = 0;

    /// <summary>
    /// Gets or sets the distance metric to use.
    /// </summary>
    /// <value>The distance metric, or null to use algorithm default (typically Euclidean).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different distance metrics are better for different
    /// types of data. Euclidean is the most common, but Cosine is better for text,
    /// and Manhattan can be better for high-dimensional data.
    /// </para>
    /// </remarks>
    public IDistanceMetric<T>? DistanceMetric { get; set; }
}
