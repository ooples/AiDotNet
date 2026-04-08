using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using HDBSCAN (Hierarchical DBSCAN).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> HDBSCAN is an improved version of DBSCAN that automatically
/// finds clusters of varying densities without requiring an epsilon parameter.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute core distances for each point
/// 2. Build a mutual reachability graph
/// 3. Construct a minimum spanning tree
/// 4. Extract a cluster hierarchy
/// 5. Points not in any cluster are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When clusters have varying densities
/// - When you don't want to tune epsilon
/// - Complex data structures with hierarchical cluster patterns
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Min cluster size: 5
/// - Min samples: 5
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Campello, R., et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates." PAKDD.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Density-Based Clustering Based on Hierarchical Density Estimates", "https://doi.org/10.1007/978-3-642-37456-2_14", Year = 2013, Authors = "Ricardo J. G. B. Campello, Davide Moulavi, Jörg Sander")]
public class HDBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _minClusterSize;
    private readonly int _minSamples;
    private Matrix<T>? _trainingData;
    private int[]? _labels;
    private T[]? _outlierScores;
    private T _maxCoreDistance;
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
    /// <param name="minSamples">Minimum number of samples for core points. Default is 5.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HDBSCANDetector(int minClusterSize = 5, int minSamples = 5, double contamination = 0.1, int randomSeed = 42)
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
        _maxCoreDistance = NumOps.One;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _trainingData = X;

        // Run simplified HDBSCAN directly on Matrix<T>
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

        int n = trainingData.Rows;

        // Step 1: Compute core distances
        var coreDistances = ComputeCoreDistances();

        // Step 2: Compute mutual reachability distances
        var mutualReach = ComputeMutualReachability(coreDistances);

        // Step 3: Build minimum spanning tree using Prim's algorithm
        var mst = BuildMST(mutualReach, n);

        // Step 4: Build cluster hierarchy and extract clusters
        ExtractClusters(mst, coreDistances, n);
    }

    private T[] ComputeCoreDistances()
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Rows;
        var coreDistances = new T[n];

        for (int i = 0; i < n; i++)
        {
            var pointI = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());

            // Compute distances to all points using vectorized ops
            var distances = new T[n];
            for (int j = 0; j < n; j++)
            {
                var pointJ = new Vector<T>(trainingData.GetRowReadOnlySpan(j).ToArray());
                var diff = Engine.Subtract(pointI, pointJ);
                distances[j] = NumOps.Sqrt(Engine.DotProduct(diff, diff));
            }

            Array.Sort(distances, (a, b) => NumOps.Compare(a, b));
            int k = Math.Min(_minSamples, n - 1);
            coreDistances[i] = distances[k];
        }

        return coreDistances;
    }

    private T[,] ComputeMutualReachability(T[] coreDistances)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Training data not initialized.");
        }

        int n = trainingData.Rows;
        var mutualReach = new T[n, n];

        for (int i = 0; i < n; i++)
        {
            var pointI = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());

            for (int j = i; j < n; j++)
            {
                if (i == j)
                {
                    mutualReach[i, j] = NumOps.Zero;
                    continue;
                }

                var pointJ = new Vector<T>(trainingData.GetRowReadOnlySpan(j).ToArray());
                var diff = Engine.Subtract(pointI, pointJ);
                T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));

                // Mutual reachability = max(core_i, core_j, dist)
                T mr = dist;
                if (NumOps.GreaterThan(coreDistances[i], mr)) mr = coreDistances[i];
                if (NumOps.GreaterThan(coreDistances[j], mr)) mr = coreDistances[j];

                mutualReach[i, j] = mr;
                mutualReach[j, i] = mr;
            }
        }

        return mutualReach;
    }

    private (int from, int to, T weight)[] BuildMST(T[,] distances, int n)
    {
        var mst = new List<(int from, int to, T weight)>();
        var inMST = new bool[n];
        var minEdge = new T[n];
        var minEdgeFrom = new int[n];

        for (int i = 0; i < n; i++)
        {
            minEdge[i] = NumOps.MaxValue;
        }

        minEdge[0] = NumOps.Zero;

        for (int count = 0; count < n; count++)
        {
            int u = -1;
            T minDist = NumOps.MaxValue;
            for (int i = 0; i < n; i++)
            {
                if (!inMST[i] && NumOps.LessThan(minEdge[i], minDist))
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

            for (int v = 0; v < n; v++)
            {
                if (!inMST[v] && NumOps.LessThan(distances[u, v], minEdge[v]))
                {
                    minEdge[v] = distances[u, v];
                    minEdgeFrom[v] = u;
                }
            }
        }

        return mst.ToArray();
    }

    private void ExtractClusters((int from, int to, T weight)[] mst, T[] coreDistances, int n)
    {
        _labels = new int[n];
        _outlierScores = new T[n];

        for (int i = 0; i < n; i++)
        {
            _labels[i] = -1; // Initialize as noise
        }

        // Sort MST edges by weight (descending) for hierarchical processing
        var sortedEdges = mst.OrderByDescending(e => NumOps.ToDouble(e.weight)).ToArray();

        var parent = new int[n];
        var rank = new int[n];
        var clusterSize = new int[n];
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            clusterSize[i] = 1;
        }

        int takeCount = Math.Max(1, (int)(sortedEdges.Length * _contamination));
        T cutoff = sortedEdges.Length > 0
            ? sortedEdges.Take(takeCount).Last().weight
            : NumOps.MaxValue;

        var ascendingEdges = mst.OrderBy(e => NumOps.ToDouble(e.weight)).ToArray();
        foreach (var edge in ascendingEdges)
        {
            if (NumOps.GreaterThan(edge.weight, cutoff)) continue;

            int root1 = Find(parent, edge.from);
            int root2 = Find(parent, edge.to);

            if (root1 != root2)
            {
                Union(parent, rank, clusterSize, root1, root2);
            }
        }

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
        }

        // Store max core distance for normalization
        _maxCoreDistance = coreDistances[0];
        for (int i = 1; i < coreDistances.Length; i++)
        {
            if (NumOps.GreaterThan(coreDistances[i], _maxCoreDistance))
                _maxCoreDistance = coreDistances[i];
        }
        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(_maxCoreDistance, epsilon)) _maxCoreDistance = NumOps.One;

        for (int i = 0; i < n; i++)
        {
            if (_labels[i] == -1)
            {
                _outlierScores[i] = NumOps.One;
            }
            else
            {
                _outlierScores[i] = NumOps.Divide(coreDistances[i], _maxCoreDistance);
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
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Find nearest training point via vectorized distance
            T minDist = NumOps.MaxValue;
            int nearestIdx = 0;

            for (int t = 0; t < trainingData.Rows; t++)
            {
                var trainPoint = new Vector<T>(trainingData.GetRowReadOnlySpan(t).ToArray());
                var diff = Engine.Subtract(point, trainPoint);
                T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));

                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                    nearestIdx = t;
                }
            }

            T score;
            if (labels[nearestIdx] == -1)
            {
                score = NumOps.One;
            }
            else
            {
                score = NumOps.Divide(minDist, _maxCoreDistance);
            }

            // Clamp to [0, 1]
            if (NumOps.GreaterThan(score, NumOps.One)) score = NumOps.One;
            scores[i] = score;
        }

        return scores;
    }
}
