using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Interfaces;

/// <summary>
/// Defines the common interface for all clustering algorithms in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Clustering is an unsupervised machine learning technique that groups similar data points
/// together without prior knowledge of the categories. This interface extends IFullModel with
/// clustering-specific functionality.
/// </para>
/// <para><b>For Beginners:</b> Clustering is about finding natural groups in data.
///
/// Unlike classification (where you have labeled examples to learn from), clustering
/// discovers patterns on its own. For example:
/// - Grouping customers by purchasing behavior
/// - Identifying topics in documents
/// - Segmenting images into regions
/// - Detecting anomalies (points that don't belong to any cluster)
///
/// The algorithm decides how many groups exist and which points belong together.
/// </para>
/// </remarks>
public interface IClustering<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Gets the number of clusters found or specified.
    /// </summary>
    /// <value>
    /// The total number of distinct clusters. For algorithms like K-Means, this is
    /// specified before training. For algorithms like DBSCAN, this is determined during training.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many groups were found in the data.
    ///
    /// Some algorithms require you to specify this (like K-Means where you say "find 5 groups"),
    /// while others figure it out automatically (like DBSCAN which finds natural groupings).
    /// </para>
    /// </remarks>
    int NumClusters { get; }

    /// <summary>
    /// Gets the cluster centers (centroids) for centroid-based clustering algorithms.
    /// </summary>
    /// <value>
    /// A matrix where each row represents a cluster center, or null for non-centroid-based
    /// algorithms like DBSCAN.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A centroid is the "center point" of a cluster.
    ///
    /// For algorithms like K-Means, each cluster has a centroid that represents its average position.
    /// New points are assigned to the cluster whose centroid is closest. Not all clustering
    /// algorithms use centroids (e.g., density-based methods).
    /// </para>
    /// </remarks>
    Matrix<T>? ClusterCenters { get; }

    /// <summary>
    /// Gets the cluster labels assigned to each training sample after fitting.
    /// </summary>
    /// <value>
    /// A vector where each element is the cluster index (0 to NumClusters-1) for the
    /// corresponding training sample. May contain -1 for noise points in density-based algorithms.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> After training, each data point is assigned to a cluster.
    ///
    /// This property shows which cluster each training point belongs to. For example:
    /// [0, 0, 1, 1, 2, 0] means the first two points are in cluster 0, the next two in cluster 1, etc.
    ///
    /// Some algorithms (like DBSCAN) use -1 to indicate "noise" points that don't belong to any cluster.
    /// </para>
    /// </remarks>
    Vector<T>? Labels { get; }

    /// <summary>
    /// Gets the inertia (within-cluster sum of squares) for centroid-based algorithms.
    /// </summary>
    /// <value>
    /// The sum of squared distances from each point to its assigned cluster center,
    /// or null for algorithms that don't compute this metric.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Inertia measures how tightly packed the clusters are.
    ///
    /// Lower inertia means points are closer to their cluster centers (tighter clusters).
    /// This is useful for choosing the number of clusters - you look for the "elbow" point
    /// where adding more clusters stops significantly reducing inertia.
    /// </para>
    /// </remarks>
    T? Inertia { get; }

    /// <summary>
    /// Fits the clustering model and returns cluster labels in one operation.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample.</param>
    /// <returns>A vector of cluster labels for each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method that combines Train (Fit) and Predict operations.
    /// It's more efficient than calling them separately for algorithms that naturally
    /// compute labels during fitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method does two things at once:
    /// 1. Learns the cluster structure from your data
    /// 2. Tells you which cluster each data point belongs to
    ///
    /// Use this when you want to cluster the same data you're learning from.
    /// </para>
    /// </remarks>
    Vector<T> FitPredict(Matrix<T> x);

    /// <summary>
    /// Transforms data into cluster-distance space.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample.</param>
    /// <returns>
    /// A matrix where each row contains the distance from the corresponding sample
    /// to each cluster center. Shape is [n_samples, n_clusters].
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is useful for soft clustering or when you need the distances
    /// to all clusters rather than just the closest one.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of just saying "this point belongs to cluster 2",
    /// this tells you how close the point is to every cluster.
    ///
    /// This is useful when:
    /// - A point is close to multiple clusters (ambiguous)
    /// - You want to understand the confidence of cluster assignments
    /// - You're building more complex systems on top of clustering
    /// </para>
    /// </remarks>
    Matrix<T> Transform(Matrix<T> x);
}
