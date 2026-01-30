using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using HDBSCAN (Hierarchical Density-Based Spatial Clustering).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> HDBSCAN improves on DBSCAN by automatically finding clusters of varying
/// densities. It builds a hierarchy of clusters and extracts the most stable ones. Points that
/// don't belong to any stable cluster are considered anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute core distances for all points
/// 2. Build mutual reachability graph
/// 3. Construct minimum spanning tree
/// 4. Build cluster hierarchy and extract stable clusters
/// 5. Points not in any cluster (noise) are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Clusters of varying densities
/// - Unknown number of clusters
/// - When DBSCAN's fixed epsilon is too restrictive
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Min cluster size: 5
/// - Min samples: 5
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Campello, R.J.G.B., et al. (2013). "Density-Based Clustering Based on
/// Hierarchical Density Estimates." PAKDD.
/// </para>
/// </remarks>
public class HDBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _minClusterSize;
    private readonly int _minSamples;
    private double[][]? _trainingData;
    private int[]? _labels;
    private double[]? _outlierScores;
    private double _maxCoreDistance;
    private int _nFeatures;

    /// <summary>
    /// Gets the minimum cluster size.
    /// </summary>
    public int MinClusterSize => _minClusterSize;

    /// <summary>
    /// Gets the minimum number of samples for core points.
    /// </summary>
    public int MinSamples => _minSamples;

    /// <summary>
    /// Creates a new HDBSCAN anomaly detector.
    /// </summary>
    /// <param name="minClusterSize">Minimum size of clusters. Default is 5.</param>
    /// <param name="minSamples">
    /// Minimum samples for a point to be a core point. Default is 5.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HDBSCANDetector(int minClusterSize = 5, int minSamples = 5,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (minClusterSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(minClusterSize),
                "MinClusterSize must be at least 2. Recommended is 5.");
        }

        if (minSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minSamples),
                "MinSamples must be at least 1. Recommended is 5.");
        }

        _minClusterSize = minClusterSize;
        _minSamples = minSamples;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;
        _nFeatures = d;

        // Convert to double array
        _trainingData = new double[n][];
        for (int i = 0; i < n; i++)
        {
            _trainingData[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                _trainingData[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Run simplified HDBSCAN
        RunHDBSCAN();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void RunHDBSCAN()
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Length;

        // Step 1: Compute core distances
        var coreDistances = ComputeCoreDistances();

        // Step 2: Compute mutual reachability distances
        var mutualReach = ComputeMutualReachability(coreDistances);

        // Step 3: Build minimum spanning tree using Prim's algorithm
        var mst = BuildMST(mutualReach);

        // Step 4: Build cluster hierarchy and extract clusters
        ExtractClusters(mst, coreDistances);
    }

    private double[] ComputeCoreDistances()
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Length;
        var coreDistances = new double[n];

        for (int i = 0; i < n; i++)
        {
            // Find k-th nearest neighbor distance
            var distances = new double[n];
            for (int j = 0; j < n; j++)
            {
                distances[j] = EuclideanDistance(trainingData[i], trainingData[j]);
            }

            Array.Sort(distances);
            // Core distance is distance to _minSamples-th neighbor (0-indexed, skip self)
            int k = Math.Min(_minSamples, n - 1);
            coreDistances[i] = distances[k];
        }

        return coreDistances;
    }

    private double[,] ComputeMutualReachability(double[] coreDistances)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Length;
        var mutualReach = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                if (i == j)
                {
                    mutualReach[i, j] = 0;
                    continue;
                }

                double dist = EuclideanDistance(trainingData[i], trainingData[j]);
                // Mutual reachability = max(core_i, core_j, dist)
                double mr = Math.Max(coreDistances[i], Math.Max(coreDistances[j], dist));
                mutualReach[i, j] = mr;
                mutualReach[j, i] = mr;
            }
        }

        return mutualReach;
    }

    private (int from, int to, double weight)[] BuildMST(double[,] distances)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Length;
        var mst = new List<(int from, int to, double weight)>();
        var inMST = new bool[n];
        var minEdge = new double[n];
        var minEdgeFrom = new int[n];

        for (int i = 0; i < n; i++)
        {
            minEdge[i] = double.MaxValue;
        }

        minEdge[0] = 0;

        for (int count = 0; count < n; count++)
        {
            // Find minimum edge
            int u = -1;
            double minDist = double.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (!inMST[i] && minEdge[i] < minDist)
                {
                    minDist = minEdge[i];
                    u = i;
                }
            }

            if (u == -1) break;

            inMST[u] = true;
            if (count > 0)
            {
                mst.Add((minEdgeFrom[u], u, minEdge[u]));
            }

            // Update adjacent vertices
            for (int v = 0; v < n; v++)
            {
                if (!inMST[v] && distances[u, v] < minEdge[v])
                {
                    minEdge[v] = distances[u, v];
                    minEdgeFrom[v] = u;
                }
            }
        }

        return mst.ToArray();
    }

    private void ExtractClusters((int from, int to, double weight)[] mst, double[] coreDistances)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Length;
        _labels = new int[n];
        _outlierScores = new double[n];

        for (int i = 0; i < n; i++)
        {
            _labels[i] = -1; // Initialize as noise
        }

        // Sort MST edges by weight (descending) for hierarchical processing
        var sortedEdges = mst.OrderByDescending(e => e.weight).ToArray();

        // Use Union-Find for clustering
        var parent = new int[n];
        var rank = new int[n];
        var clusterSize = new int[n];
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            clusterSize[i] = 1;
        }

        // Build clusters by removing long edges
        // Ensure at least 1 edge is taken to avoid empty sequence from Take(0)
        int takeCount = Math.Max(1, (int)(sortedEdges.Length * _contamination));
        double cutoff = sortedEdges.Length > 0
            ? sortedEdges.Take(takeCount).Last().weight
            : double.MaxValue;

        // Process edges in ascending order to build clusters
        var ascendingEdges = mst.OrderBy(e => e.weight).ToArray();
        foreach (var edge in ascendingEdges)
        {
            if (edge.weight > cutoff) continue;

            int root1 = Find(parent, edge.from);
            int root2 = Find(parent, edge.to);

            if (root1 != root2)
            {
                Union(parent, rank, clusterSize, root1, root2);
            }
        }

        // Assign cluster labels
        var rootToLabel = new Dictionary<int, int>();
        int nextLabel = 0;

        for (int i = 0; i < n; i++)
        {
            int root = Find(parent, i);
            if (clusterSize[root] >= _minClusterSize)
            {
                if (!rootToLabel.ContainsKey(root))
                {
                    rootToLabel[root] = nextLabel++;
                }
                _labels[i] = rootToLabel[root];
            }
            // Else remains -1 (noise)
        }

        // Store max core distance for proper normalization when scoring new data
        _maxCoreDistance = coreDistances.Max();
        if (_maxCoreDistance < 1e-10) _maxCoreDistance = 1.0;

        // Compute outlier scores based on core distances and cluster membership
        for (int i = 0; i < n; i++)
        {
            if (_labels[i] == -1)
            {
                // Noise points get high outlier score
                _outlierScores[i] = 1.0;
            }
            else
            {
                // Cluster members: score based on core distance relative to cluster
                _outlierScores[i] = coreDistances[i] / _maxCoreDistance;
            }
        }
    }

    private int Find(int[] parent, int i)
    {
        if (parent[i] != i)
            parent[i] = Find(parent, parent[i]);
        return parent[i];
    }

    private void Union(int[] parent, int[] rank, int[] size, int x, int y)
    {
        int rootX = Find(parent, x);
        int rootY = Find(parent, y);

        if (rank[rootX] < rank[rootY])
        {
            parent[rootX] = rootY;
            size[rootY] += size[rootX];
        }
        else if (rank[rootX] > rank[rootY])
        {
            parent[rootY] = rootX;
            size[rootX] += size[rootY];
        }
        else
        {
            parent[rootY] = rootX;
            size[rootX] += size[rootY];
            rank[rootX]++;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var trainingData = _trainingData;
        var labels = _labels;
        var outlierScores = _outlierScores;

        if (trainingData == null || labels == null || outlierScores == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // For new points, compute distance to nearest training point
            // and use mutual reachability concept
            double minDist = double.MaxValue;
            int nearestIdx = 0;

            for (int t = 0; t < trainingData.Length; t++)
            {
                double dist = EuclideanDistance(point, trainingData[t]);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestIdx = t;
                }
            }

            // Score based on distance and whether nearest neighbor is noise
            double score;
            if (labels[nearestIdx] == -1)
            {
                // Nearest neighbor is noise - high score
                score = 1.0;
            }
            else
            {
                // Score based on distance normalized by max core distance from training
                score = minDist / _maxCoreDistance;
            }

            scores[i] = NumOps.FromDouble(Math.Min(score, 1.0));
        }

        return scores;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
