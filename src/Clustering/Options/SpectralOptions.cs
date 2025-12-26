using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Spectral Clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spectral clustering uses the eigenvalues of a similarity matrix to perform
/// dimensionality reduction before clustering. It can find non-convex clusters
/// that K-Means cannot.
/// </para>
/// <para><b>For Beginners:</b> Spectral clustering finds clusters by analyzing connections.
///
/// Think of your data as a graph where:
/// - Each point is a node
/// - Similar points are connected by edges
/// - Clusters are groups of densely connected nodes
///
/// The algorithm:
/// 1. Build a similarity graph (like a social network)
/// 2. Find the "natural cuts" in this graph
/// 3. These cuts define your clusters
///
/// When to use:
/// - Clusters have unusual shapes (crescents, spirals)
/// - You can define a good similarity measure
/// - Data has clear connectivity patterns
/// </para>
/// </remarks>
public class SpectralOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes a new instance of SpectralOptions.
    /// </summary>
    public SpectralOptions()
    {
        MaxIterations = 300; // For internal KMeans
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 8.</value>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the affinity/similarity method.
    /// </summary>
    /// <value>Affinity method. Default is RBF.</value>
    public AffinityType Affinity { get; set; } = AffinityType.RBF;

    /// <summary>
    /// Gets or sets the gamma parameter for RBF kernel.
    /// </summary>
    /// <value>Gamma value, or null for automatic (1/n_features).</value>
    public double? Gamma { get; set; }

    /// <summary>
    /// Gets or sets the number of neighbors for nearest neighbors affinity.
    /// </summary>
    /// <value>Number of neighbors. Default is 10.</value>
    public int NumNeighbors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the eigenvalue solver method.
    /// </summary>
    /// <value>Eigen solver. Default is Arpack.</value>
    public EigenSolver EigenSolver { get; set; } = EigenSolver.Arpack;

    /// <summary>
    /// Gets or sets the type of Laplacian normalization.
    /// </summary>
    /// <value>Normalization type. Default is Normalized.</value>
    public LaplacianNormalization Normalization { get; set; } = LaplacianNormalization.Normalized;

    /// <summary>
    /// Gets or sets the assignment strategy after spectral embedding.
    /// </summary>
    /// <value>Assignment method. Default is KMeans.</value>
    public SpectralAssignment AssignLabels { get; set; } = SpectralAssignment.KMeans;

    /// <summary>
    /// Gets or sets the distance metric for building affinity matrix.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}

/// <summary>
/// Types of affinity/similarity computation.
/// </summary>
public enum AffinityType
{
    /// <summary>
    /// RBF (Radial Basis Function) kernel: exp(-gamma * ||x-y||²).
    /// </summary>
    RBF,

    /// <summary>
    /// Nearest neighbors: Points are similar if they're nearest neighbors.
    /// </summary>
    NearestNeighbors,

    /// <summary>
    /// Precomputed: Affinity matrix is provided directly.
    /// </summary>
    Precomputed,

    /// <summary>
    /// Polynomial kernel.
    /// </summary>
    Polynomial,

    /// <summary>
    /// Sigmoid kernel.
    /// </summary>
    Sigmoid
}

/// <summary>
/// Eigenvalue solver methods.
/// </summary>
public enum EigenSolver
{
    /// <summary>
    /// Use ARPACK for sparse matrices (default for large datasets).
    /// </summary>
    Arpack,

    /// <summary>
    /// Use LOBPCG for sparse matrices with preconditioner.
    /// </summary>
    Lobpcg,

    /// <summary>
    /// Use AMG for algebraic multigrid preconditioned LOBPCG.
    /// </summary>
    Amg,

    /// <summary>
    /// Full eigenvalue decomposition (only for small datasets).
    /// </summary>
    Full
}

/// <summary>
/// Laplacian matrix normalization types.
/// </summary>
public enum LaplacianNormalization
{
    /// <summary>
    /// Unnormalized Laplacian: L = D - W.
    /// </summary>
    Unnormalized,

    /// <summary>
    /// Symmetric normalized: L = D^(-1/2) * (D-W) * D^(-1/2).
    /// Also called Normalized Cut.
    /// </summary>
    Normalized,

    /// <summary>
    /// Random walk normalization: L = D^(-1) * (D-W).
    /// </summary>
    RandomWalk
}

/// <summary>
/// Label assignment methods after spectral embedding.
/// </summary>
public enum SpectralAssignment
{
    /// <summary>
    /// Use K-Means on the embedded space.
    /// </summary>
    KMeans,

    /// <summary>
    /// Use discretization to find closest discrete solution.
    /// </summary>
    Discretize
}
