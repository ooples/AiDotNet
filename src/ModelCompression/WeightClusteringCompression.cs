using AiDotNet.Helpers;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements weight clustering compression using K-means clustering to group similar weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Weight clustering reduces model size by identifying groups of similar weight values and replacing
/// them with their cluster representatives. This technique can achieve significant compression ratios
/// (10-50x) while maintaining model accuracy.
/// </para>
/// <para><b>For Beginners:</b> Weight clustering is like organizing a messy toolbox.
///
/// Imagine you have thousands of screws that are almost the same size:
/// - Some are 2.01mm, some 2.02mm, some 2.03mm, etc.
/// - Instead of keeping track of each exact size, you group similar sizes together
/// - You replace all sizes in a group with one representative size (like 2.0mm)
///
/// For neural networks:
/// - Instead of storing millions of slightly different weight values
/// - We group similar weights into clusters (like 256 or 512 groups)
/// - Each weight is replaced with its cluster center
/// - Instead of storing millions of unique values, we store which cluster each weight belongs to
///
/// This dramatically reduces storage because:
/// - Cluster IDs are much smaller than full weight values (8 bits vs 32 bits)
/// - We only need to store the cluster centers once
///
/// The result is a much smaller model that performs almost the same as the original!
/// </para>
/// </remarks>
public class WeightClusteringCompression<T> : ModelCompressionBase<T>
{
    private readonly int _numClusters;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the WeightClusteringCompression class.
    /// </summary>
    /// <param name="numClusters">The number of clusters to use (default: 256 for 8-bit quantization).</param>
    /// <param name="maxIterations">The maximum number of K-means iterations (default: 100).</param>
    /// <param name="tolerance">The convergence tolerance for K-means (default: 1e-6).</param>
    /// <param name="randomSeed">Random seed for reproducibility (default: null for random).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> These parameters control the compression behavior:
    ///
    /// - numClusters: How many groups to create
    ///   * 256 clusters = 8-bit compression (very common, good balance)
    ///   * 128 clusters = 7-bit compression (more aggressive)
    ///   * 512 clusters = 9-bit compression (less aggressive, higher quality)
    ///
    /// - maxIterations: How hard to try finding the best clusters
    ///   * Higher = better clusters but slower compression
    ///   * 100 is usually plenty
    ///
    /// - tolerance: When to stop improving clusters
    ///   * Smaller = more precise but takes longer
    ///   * 1e-6 is a good default
    ///
    /// - randomSeed: For getting the same results each time (useful for testing)
    /// </para>
    /// </remarks>
    public WeightClusteringCompression(
        int numClusters = 256,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int? randomSeed = null)
    {
        if (numClusters <= 0)
        {
            throw new ArgumentException("Number of clusters must be positive.", nameof(numClusters));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentException("Max iterations must be positive.", nameof(maxIterations));
        }

        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
    }

    /// <summary>
    /// Compresses weights using K-means clustering.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed weights and metadata containing cluster centers and assignments.</returns>
    public override (T[] compressedWeights, object metadata) Compress(T[] weights)
    {
        if (weights == null || weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be null or empty.", nameof(weights));
        }

        // Adjust number of clusters if we have fewer weights
        int effectiveClusters = Math.Min(_numClusters, weights.Length);

        // Run K-means clustering
        var (clusterCenters, assignments) = PerformKMeansClustering(weights, effectiveClusters);

        // Create metadata
        var metadata = new WeightClusteringMetadata
        {
            ClusterCenters = clusterCenters,
            NumClusters = effectiveClusters,
            OriginalLength = weights.Length
        };

        // Compressed weights are the cluster assignments (as indices)
        var compressedWeights = new T[assignments.Length];
        for (int i = 0; i < assignments.Length; i++)
        {
            compressedWeights[i] = NumOps.FromDouble(assignments[i]);
        }

        return (compressedWeights, metadata);
    }

    /// <summary>
    /// Decompresses weights by mapping cluster assignments back to cluster centers.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights (cluster assignments).</param>
    /// <param name="metadata">The metadata containing cluster centers.</param>
    /// <returns>The decompressed weights.</returns>
    public override T[] Decompress(T[] compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata is not WeightClusteringMetadata clusterMetadata)
        {
            throw new ArgumentException("Invalid metadata type for weight clustering.", nameof(metadata));
        }

        var decompressedWeights = new T[compressedWeights.Length];
        for (int i = 0; i < compressedWeights.Length; i++)
        {
            int clusterIndex = (int)NumOps.ToDouble(compressedWeights[i]);
            decompressedWeights[i] = clusterMetadata.ClusterCenters[clusterIndex];
        }

        return decompressedWeights;
    }

    /// <summary>
    /// Gets the compressed size including cluster centers and assignments.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights.</param>
    /// <param name="metadata">The compression metadata.</param>
    /// <returns>The total size in bytes.</returns>
    public override long GetCompressedSize(T[] compressedWeights, object metadata)
    {
        if (metadata is not WeightClusteringMetadata clusterMetadata)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size of cluster centers (full precision)
        long clusterCentersSize = clusterMetadata.NumClusters * GetElementSize();

        // Size of cluster assignments (can use fewer bits, but we'll use int for simplicity)
        long assignmentsSize = compressedWeights.Length * sizeof(int);

        // Metadata overhead (NumClusters, OriginalLength)
        long metadataSize = sizeof(int) * 2;

        return clusterCentersSize + assignmentsSize + metadataSize;
    }

    /// <summary>
    /// Performs K-means clustering on the weights.
    /// </summary>
    /// <param name="weights">The weights to cluster.</param>
    /// <param name="k">The number of clusters.</param>
    /// <returns>Cluster centers and assignments for each weight.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> K-means clustering finds groups of similar values.
    ///
    /// The algorithm works like this:
    /// 1. Start with random cluster centers
    /// 2. Assign each weight to its nearest cluster center
    /// 3. Update cluster centers to be the average of their assigned weights
    /// 4. Repeat steps 2-3 until centers stop moving
    ///
    /// It's like organizing students into groups by height:
    /// - First, pick some random heights as group centers
    /// - Assign each student to the nearest height
    /// - Calculate the average height of each group (new centers)
    /// - Repeat until groups stabilize
    /// </para>
    /// </remarks>
    private (T[] clusterCenters, int[] assignments) PerformKMeansClustering(T[] weights, int k)
    {
        // Initialize cluster centers using k-means++ for better initial placement
        var clusterCenters = InitializeClusterCentersKMeansPlusPlus(weights, k);
        var assignments = new int[weights.Length];
        var clusterSums = new double[k];
        var clusterCounts = new int[k];

        double previousInertia = double.MaxValue;

        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            // Assign each weight to nearest cluster
            Array.Clear(clusterSums, 0, k);
            Array.Clear(clusterCounts, 0, k);

            for (int i = 0; i < weights.Length; i++)
            {
                double weightValue = NumOps.ToDouble(weights[i]);
                int nearestCluster = FindNearestCluster(weightValue, clusterCenters);
                assignments[i] = nearestCluster;
                clusterSums[nearestCluster] += weightValue;
                clusterCounts[nearestCluster]++;
            }

            // Update cluster centers
            for (int j = 0; j < k; j++)
            {
                if (clusterCounts[j] > 0)
                {
                    clusterCenters[j] = NumOps.FromDouble(clusterSums[j] / clusterCounts[j]);
                }
            }

            // Check convergence
            double inertia = CalculateInertia(weights, clusterCenters, assignments);
            if (Math.Abs(previousInertia - inertia) < _tolerance)
            {
                break;
            }
            previousInertia = inertia;
        }

        return (clusterCenters, assignments);
    }

    /// <summary>
    /// Initializes cluster centers using the K-means++ algorithm for better initial placement.
    /// </summary>
    private T[] InitializeClusterCentersKMeansPlusPlus(T[] weights, int k)
    {
        var centers = new T[k];
        var distances = new double[weights.Length];

        // Choose first center randomly
        centers[0] = weights[_random.Next(weights.Length)];

        for (int i = 1; i < k; i++)
        {
            // Calculate distances to nearest center for each point
            double totalDistance = 0;
            for (int j = 0; j < weights.Length; j++)
            {
                double minDist = double.MaxValue;
                for (int c = 0; c < i; c++)
                {
                    double dist = Math.Abs(NumOps.ToDouble(weights[j]) - NumOps.ToDouble(centers[c]));
                    minDist = Math.Min(minDist, dist);
                }
                distances[j] = minDist * minDist;
                totalDistance += distances[j];
            }

            // Choose next center with probability proportional to distance squared
            double randomValue = _random.NextDouble() * totalDistance;
            double cumulative = 0;
            for (int j = 0; j < weights.Length; j++)
            {
                cumulative += distances[j];
                if (cumulative >= randomValue)
                {
                    centers[i] = weights[j];
                    break;
                }
            }
        }

        return centers;
    }

    /// <summary>
    /// Finds the nearest cluster center for a given weight value.
    /// </summary>
    private int FindNearestCluster(double weight, T[] clusterCenters)
    {
        int nearestCluster = 0;
        double minDistance = double.MaxValue;

        for (int i = 0; i < clusterCenters.Length; i++)
        {
            double distance = Math.Abs(weight - NumOps.ToDouble(clusterCenters[i]));
            if (distance < minDistance)
            {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    /// <summary>
    /// Calculates the total inertia (sum of squared distances to cluster centers).
    /// </summary>
    private double CalculateInertia(T[] weights, T[] clusterCenters, int[] assignments)
    {
        double inertia = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double weight = NumOps.ToDouble(weights[i]);
            double center = NumOps.ToDouble(clusterCenters[assignments[i]]);
            double diff = weight - center;
            inertia += diff * diff;
        }
        return inertia;
    }
}

/// <summary>
/// Metadata for weight clustering compression.
/// </summary>
public class WeightClusteringMetadata
{
    /// <summary>
    /// The cluster centers.
    /// </summary>
    public required T[] ClusterCenters { get; init; }

    /// <summary>
    /// The number of clusters.
    /// </summary>
    public required int NumClusters { get; init; }

    /// <summary>
    /// The original length of the weights array.
    /// </summary>
    public required int OriginalLength { get; init; }
}
