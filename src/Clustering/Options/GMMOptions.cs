using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Gaussian Mixture Model clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GMM assumes data is generated from a mixture of several Gaussian distributions.
/// It finds the parameters of these distributions using the Expectation-Maximization (EM) algorithm.
/// </para>
/// <para><b>For Beginners:</b> GMM is like finding overlapping groups in data.
///
/// Imagine drops of different colored paint on paper:
/// - K-Means: Draws hard boundaries between colors
/// - GMM: Allows colors to blend, giving probability of belonging to each
///
/// Key features:
/// - Soft clustering: Each point has probabilities for all clusters
/// - Captures different cluster shapes and sizes
/// - Based on statistical modeling
///
/// When to use GMM:
/// - Clusters have different sizes or shapes
/// - You need probability of cluster membership
/// - Data might have overlapping clusters
/// </para>
/// </remarks>
public class GMMOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes a new instance of GMMOptions with GMM-appropriate defaults.
    /// </summary>
    public GMMOptions()
    {
        // Override base defaults with GMM-specific values
        MaxIterations = 100;
        NumInitializations = 1;
    }

    /// <summary>
    /// Gets or sets the number of mixture components.
    /// </summary>
    /// <value>The number of Gaussian components. Default is 2.</value>
    public int NumComponents { get; set; } = 2;

    /// <summary>
    /// Gets or sets the type of covariance parameters to use.
    /// </summary>
    /// <value>The covariance type. Default is Full.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls cluster shape flexibility:
    /// - Full: Each cluster can be any ellipse shape (most flexible)
    /// - Tied: All clusters share the same ellipse shape
    /// - Diagonal: Clusters are axis-aligned ellipses
    /// - Spherical: All clusters are spheres (simplest)
    /// </para>
    /// </remarks>
    public CovarianceType CovarianceType { get; set; } = CovarianceType.Full;

    /// <summary>
    /// Gets or sets the regularization added to the diagonal of covariance.
    /// </summary>
    /// <value>Regularization value. Default is 1e-6.</value>
    /// <remarks>
    /// Prevents singular covariance matrices that can occur with too few samples.
    /// </remarks>
    public double RegularizationCovariance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the initialization method.
    /// </summary>
    /// <value>The initialization method. Default is KMeans.</value>
    public GMMInitMethod InitMethod { get; set; } = GMMInitMethod.KMeans;

    /// <summary>
    /// Gets or sets the weight concentration prior (for Dirichlet process).
    /// </summary>
    /// <value>Weight concentration prior, or null for maximum likelihood.</value>
    public double? WeightConcentrationPrior { get; set; }

    /// <summary>
    /// Gets or sets whether to compute the lower bound during training.
    /// </summary>
    /// <value>True to compute lower bound. Default is true.</value>
    public bool ComputeLowerBound { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to allow components with very low weights.
    /// </summary>
    /// <value>True to allow low weights. Default is false.</value>
    public bool AllowLowWeights { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum weight threshold for components.
    /// </summary>
    /// <value>Minimum weight. Default is 1e-3.</value>
    public double MinWeight { get; set; } = 1e-3;
}

/// <summary>
/// Type of covariance parameters to estimate.
/// </summary>
public enum CovarianceType
{
    /// <summary>
    /// Each component has its own general covariance matrix.
    /// Most flexible, captures arbitrary ellipse shapes.
    /// </summary>
    Full,

    /// <summary>
    /// All components share the same general covariance matrix.
    /// Reduces parameters while allowing ellipse shapes.
    /// </summary>
    Tied,

    /// <summary>
    /// Each component has its own diagonal covariance matrix.
    /// Axis-aligned ellipses, fewer parameters than full.
    /// </summary>
    Diagonal,

    /// <summary>
    /// Each component has its own single variance.
    /// Spherical clusters only, fewest parameters.
    /// </summary>
    Spherical
}

/// <summary>
/// Initialization methods for GMM.
/// </summary>
public enum GMMInitMethod
{
    /// <summary>
    /// Initialize using K-Means clustering.
    /// Most common and usually works well.
    /// </summary>
    KMeans,

    /// <summary>
    /// Random initialization.
    /// May require more restarts but is faster per initialization.
    /// </summary>
    Random,

    /// <summary>
    /// K-Means++ style initialization for means.
    /// Better spread of initial centers.
    /// </summary>
    KMeansPlusPlus
}
