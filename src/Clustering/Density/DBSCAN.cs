using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.SpatialIndex;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DBSCAN is a density-based clustering algorithm that groups together points that are
/// closely packed together, marking points in low-density regions as outliers.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. For each point, find all neighbors within epsilon distance
/// 2. If a point has at least MinPoints neighbors, it's a core point
/// 3. Core points form cluster seeds
/// 4. Expand clusters by adding density-reachable points
/// 5. Mark remaining points as noise (-1)
/// </para>
/// <para><b>For Beginners:</b> DBSCAN finds clusters by looking for dense regions.
///
/// Imagine you're at a party:
/// - Dense groups of people talking are clusters
/// - People standing alone are noise/outliers
/// - DBSCAN finds these groups automatically
///
/// Key advantages:
/// - Discovers clusters of any shape (not just circles)
/// - Doesn't require knowing the number of clusters
/// - Identifies outliers as noise
/// - Handles clusters of different sizes and densities
///
/// Limitations:
/// - Sensitive to epsilon and MinPoints parameters
/// - May struggle with clusters of varying density
/// - O(n²) complexity without spatial indexing
/// </para>
/// </remarks>
public class DBSCAN<T> : ClusteringBase<T>
{
    private readonly DBSCANOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private bool[]? _corePointMask;

    /// <summary>
    /// Noise label (points not assigned to any cluster).
    /// </summary>
    public const int NoiseLabel = -1;

    /// <summary>
    /// Undefined label (not yet processed).
    /// </summary>
    private const int UndefinedLabel = -2;

    /// <summary>
    /// Initializes a new DBSCAN instance with the specified options.
    /// </summary>
    /// <param name="options">The DBSCAN configuration options.</param>
    public DBSCAN(DBSCANOptions<T>? options = null)
        : base(options ?? new DBSCANOptions<T>())
    {
        _options = options ?? new DBSCANOptions<T>();

        if (_options.DistanceMetric is null)
        {
            _options.DistanceMetric = new EuclideanDistance<T>();
        }
    }

    /// <summary>
    /// Gets the epsilon neighborhood radius.
    /// </summary>
    public double Epsilon => _options.Epsilon;

    /// <summary>
    /// Gets the minimum points for core point classification.
    /// </summary>
    public int MinPoints => _options.MinPoints;

    /// <summary>
    /// Gets the mask indicating which points are core points.
    /// </summary>
    public bool[]? CorePointMask => _corePointMask;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DBSCAN<T>(new DBSCANOptions<T>
        {
            Epsilon = _options.Epsilon,
            MinPoints = _options.MinPoints,
            Algorithm = _options.Algorithm,
            LeafSize = _options.LeafSize,
            P = _options.P,
            DistanceMetric = _options.DistanceMetric,
            NumJobs = _options.NumJobs
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (DBSCAN<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        int n = x.Rows;
        var labels = new int[n];
        _corePointMask = new bool[n];

        // Initialize all points as undefined
        for (int i = 0; i < n; i++)
        {
            labels[i] = UndefinedLabel;
        }

        // Build spatial index for efficient neighbor queries
        var neighborFinder = CreateNeighborFinder(x);

        // Find neighbors for each point and identify core points
        var neighbors = new List<int>[n];
        T epsilonT = NumOps.FromDouble(_options.Epsilon);

        for (int i = 0; i < n; i++)
        {
            var query = GetRow(x, i);
            neighbors[i] = FindNeighbors(neighborFinder, x, query, epsilonT);
            _corePointMask[i] = neighbors[i].Count >= _options.MinPoints;
        }

        // Cluster expansion
        int currentCluster = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != UndefinedLabel)
            {
                continue; // Already processed
            }

            if (!_corePointMask[i])
            {
                labels[i] = NoiseLabel; // Mark as noise for now
                continue;
            }

            // Start a new cluster
            ExpandCluster(x, i, neighbors, labels, currentCluster, epsilonT, neighborFinder);
            currentCluster++;
        }

        // Convert to Vector<T>
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        NumClusters = currentCluster;

        // Compute cluster centers for clusters with at least one point
        if (currentCluster > 0)
        {
            ComputeClusterCenters(x, labels, currentCluster);
        }

        IsTrained = true;
    }

    private void ExpandCluster(
        Matrix<T> x,
        int seedIndex,
        List<int>[] neighbors,
        int[] labels,
        int clusterId,
        T epsilon,
        object neighborFinder)
    {
        // Use a queue for BFS expansion
        var queue = new Queue<int>();
        queue.Enqueue(seedIndex);
        labels[seedIndex] = clusterId;

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            // If current is a core point, expand to its neighbors
            if (_corePointMask![current])
            {
                foreach (int neighborIdx in neighbors[current])
                {
                    if (labels[neighborIdx] == UndefinedLabel || labels[neighborIdx] == NoiseLabel)
                    {
                        if (labels[neighborIdx] == UndefinedLabel)
                        {
                            queue.Enqueue(neighborIdx);
                        }
                        labels[neighborIdx] = clusterId;
                    }
                }
            }
        }
    }

    private void ComputeClusterCenters(Matrix<T> x, int[] labels, int numClusters)
    {
        ClusterCenters = new Matrix<T>(numClusters, x.Columns);
        var counts = new int[numClusters];

        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = labels[i];
            if (cluster >= 0 && cluster < numClusters)
            {
                counts[cluster]++;
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[cluster, j] = NumOps.Add(ClusterCenters[cluster, j], x[i, j]);
                }
            }
        }

        // Compute mean
        for (int k = 0; k < numClusters; k++)
        {
            if (counts[k] > 0)
            {
                T countT = NumOps.FromDouble(counts[k]);
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], countT);
                }
            }
        }
    }

    private object CreateNeighborFinder(Matrix<T> x)
    {
        var algorithm = _options.Algorithm;

        // Auto-select algorithm based on data characteristics
        if (algorithm == NeighborAlgorithm.Auto)
        {
            if (x.Columns > 20)
            {
                algorithm = NeighborAlgorithm.BallTree;
            }
            else if (x.Rows < 50)
            {
                algorithm = NeighborAlgorithm.BruteForce;
            }
            else
            {
                algorithm = NeighborAlgorithm.KDTree;
            }
        }

        switch (algorithm)
        {
            case NeighborAlgorithm.KDTree:
                var kdTree = new KDTree<T>(_options.DistanceMetric, _options.LeafSize);
                kdTree.Build(x);
                return kdTree;

            case NeighborAlgorithm.BallTree:
                var ballTree = new BallTree<T>(_options.DistanceMetric, _options.LeafSize);
                ballTree.Build(x);
                return ballTree;

            case NeighborAlgorithm.BruteForce:
            default:
                return x; // Just return the data for brute force
        }
    }

    private List<int> FindNeighbors(object neighborFinder, Matrix<T> data, Vector<T> query, T epsilon)
    {
        if (neighborFinder is KDTree<T> kdTree)
        {
            var results = kdTree.QueryRadius(query, epsilon);
            return results.Select(r => r.Index).ToList();
        }
        else if (neighborFinder is BallTree<T> ballTree)
        {
            var results = ballTree.QueryRadius(query, epsilon);
            return results.Select(r => r.Index).ToList();
        }
        else
        {
            // Brute force
            return FindNeighborsBruteForce(data, query, epsilon);
        }
    }

    private List<int> FindNeighborsBruteForce(Matrix<T> data, Vector<T> query, T epsilon)
    {
        var neighbors = new List<int>();
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        double epsilonDouble = NumOps.ToDouble(epsilon);

        for (int i = 0; i < data.Rows; i++)
        {
            var point = GetRow(data, i);
            T dist = distanceMetric.Compute(query, point);
            if (NumOps.ToDouble(dist) <= epsilonDouble)
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // For new data, assign to nearest cluster or noise
        var labels = new Vector<T>(x.Rows);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearestCluster = NoiseLabel;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    T dist = distanceMetric.Compute(point, center);
                    double distDouble = NumOps.ToDouble(dist);

                    if (distDouble < minDist)
                    {
                        minDist = distDouble;
                        nearestCluster = k;
                    }
                }

                // Only assign to cluster if within epsilon of the center
                if (minDist > _options.Epsilon * 2)
                {
                    nearestCluster = NoiseLabel;
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            // No clusters found, return single column of zeros
            return new Matrix<T>(x.Rows, 1);
        }

        return base.Transform(x);
    }

    /// <summary>
    /// Gets the indices of core samples.
    /// </summary>
    /// <returns>Array of indices that are core points.</returns>
    public int[] GetCoreSampleIndices()
    {
        if (_corePointMask is null)
        {
            return Array.Empty<int>();
        }

        var indices = new List<int>();
        for (int i = 0; i < _corePointMask.Length; i++)
        {
            if (_corePointMask[i])
            {
                indices.Add(i);
            }
        }

        return indices.ToArray();
    }

    /// <summary>
    /// Gets the number of noise points (outliers).
    /// </summary>
    /// <returns>Count of points labeled as noise.</returns>
    public int GetNoiseCount()
    {
        if (Labels is null)
        {
            return 0;
        }

        int count = 0;
        for (int i = 0; i < Labels.Length; i++)
        {
            if ((int)NumOps.ToDouble(Labels[i]) == NoiseLabel)
            {
                count++;
            }
        }

        return count;
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows < 1)
        {
            throw new ArgumentException("Data must have at least one sample.");
        }

        if (_options.Epsilon <= 0)
        {
            throw new ArgumentException("Epsilon must be positive.");
        }

        if (_options.MinPoints < 1)
        {
            throw new ArgumentException("MinPoints must be at least 1.");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (ClusterCenters is not null && x.Columns != ClusterCenters.Columns)
        {
            throw new ArgumentException(
                $"Input columns ({x.Columns}) must match training data columns ({ClusterCenters.Columns}).");
        }
    }
}
