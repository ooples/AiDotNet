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
public partial class HDBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _minClusterSize;
    private readonly int _minSamples;
    [Buffer]
    private Matrix<T>? _trainingData;
    private int[]? _labels;
    private Vector<T>? _outlierScores;
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

        // Set the contamination threshold from the fit-time per-point outlier scores
        // (core-distance / noise-membership based), NOT ScoreAnomaliesInternal(X).
        // ScoreAnomaliesInternal scores each training point against the training set
        // INCLUDING ITSELF, so its nearest neighbour is the point itself at distance 0 and
        // every training score collapses to ~0. The contamination threshold then degenerates
        // to 0, and every subsequent query — normal or genuine outlier — scores above it and
        // is flagged an anomaly (Outliers_ShouldHaveHigherScores saw normal and outlier both
        // predicted -1). The fit-time _outlierScores give a real score distribution to
        // percentile against.
        var outlierScores = _outlierScores
            ?? throw new InvalidOperationException("HDBSCAN outlier scores were not computed during fit.");
        SetThresholdFromContamination(outlierScores);

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
        var points = AnomalyTensorOps.ToTensor(trainingData);
        var distances = AnomalyTensorOps.PairwiseDistances(points, points);

        // Step 1: Compute core distances
        var coreDistances = ComputeCoreDistances(distances);

        // Step 2: Compute mutual reachability distances
        var mutualReach = ComputeMutualReachability(distances, coreDistances);

        // Step 3: Build minimum spanning tree using Prim's algorithm
        var (from, to, weights) = BuildMST(mutualReach, n);

        // Step 4: Build cluster hierarchy and extract clusters
        ExtractClusters(from, to, weights, coreDistances, n);
    }

    // Distance to the min(minSamples, n - 1)-th nearest point (the point itself, at 0, being the 0-th).
    private Tensor<T> ComputeCoreDistances(Tensor<T> distances)
    {
        var engine = Engine;
        int n = distances.Shape[0];
        var (sorted, _) = engine.TensorSort(distances, 1, false);
        return engine.TensorSliceAxis(sorted, 1, Math.Min(_minSamples, n - 1));
    }

    // Mutual reachability = max(core_i, core_j, dist), 0 on the diagonal.
    private Tensor<T> ComputeMutualReachability(Tensor<T> distances, Tensor<T> coreDistances)
    {
        var engine = Engine;
        int n = distances.Shape[0];
        var mutualReach = engine.TensorMax(engine.TensorMax(distances, AnomalyTensorOps.BroadcastColumn(coreDistances, n)),
            AnomalyTensorOps.BroadcastRow(coreDistances, n));
        return engine.TensorMaskedFill(mutualReach, engine.TensorEqScalar(engine.TensorEye<T>(n), NumOps.One), NumOps.Zero);
    }

    // Prim's algorithm from vertex 0: the edges (from, to) in the order they join the tree and their weights.
    private (int[] from, int[] to, Tensor<T> weights) BuildMST(Tensor<T> distances, int n)
    {
        var engine = Engine;
        T maxValue = NumOps.MaxValue;
        T one = NumOps.One;
        var minEdge = engine.TensorIndexFill(engine.Fill(n, maxValue).AsTensor(), 0, AnomalyTensorOps.Range(1), NumOps.Zero);
        var inMST = Tensor<T>.CreateZeros(new[] { n });
        var minEdgeFrom = new int[n];
        var from = new List<int>();
        var to = new List<int>();
        var weights = new List<Tensor<T>>();
        var unavailable = engine.Fill(n, maxValue).AsTensor();

        for (int count = 0; count < n; count++)
        {
            // The vertex outside the tree with the smallest edge to it (first on ties); none below MaxValue ends
            var candidates = engine.TensorWhere(inMST, unavailable, minEdge);
            var uTensor = engine.TensorArgMin(candidates, 0);
            int u = uTensor.ToArray()[0];
            var edge = engine.TensorSlice(candidates, new[] { u }, new[] { 1 });
            if (engine.TensorLessThan(edge, maxValue).Cast<int>().ToArray()[0] == 0) break;

            inMST = engine.TensorIndexFill(inMST, 0, engine.Reshape(uTensor, new[] { 1 }), one);
            if (count > 0)
            {
                from.Add(minEdgeFrom[u]);
                to.Add(u);
                weights.Add(edge);
            }

            // Edges through u that improve on the current best for vertices outside the tree
            var row = engine.TensorSliceAxis(distances, 0, u);
            var improves = engine.TensorMultiply(engine.ScalarMinusTensor(one, inMST), engine.TensorLessThan(row, minEdge));
            minEdge = engine.TensorWhere(improves, row, minEdge);
            var improved = improves.Cast<int>().ToArray();
            for (int v = 0; v < n; v++)
            {
                if (improved[v] != 0)
                {
                    minEdgeFrom[v] = u;
                }
            }
        }

        var weightTensor = weights.Count > 0 ? engine.TensorConcatenate(weights.ToArray(), 0) : Tensor<T>.CreateZeros(new[] { 0 });
        return (from.ToArray(), to.ToArray(), weightTensor);
    }

    private void ExtractClusters(int[] from, int[] to, Tensor<T> weights, Tensor<T> coreDistances, int n)
    {
        var engine = Engine;
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1; // Initialize as noise
        }

        var parent = new int[n];
        var rank = new int[n];
        var clusterSize = new int[n];
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            clusterSize[i] = 1;
        }

        // Cutoff = the takeCount-th largest MST edge weight; edges above it are cut
        int edgeCount = from.Length;
        if (edgeCount > 0)
        {
            int takeCount = Math.Min(Math.Max(1, (int)(edgeCount * _contamination)), edgeCount);
            var (descending, _) = engine.TensorSort(weights, 0, true);
            T cutoff = engine.TensorSlice(descending, new[] { takeCount - 1 }, new[] { 1 }).ToVector()[0];

            // Join the kept edges in ascending weight order (stable)
            var ascending = engine.TensorArgsort(weights, 0, false).ToArray();
            var kept = engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(weights, cutoff)).Cast<int>().ToArray();
            foreach (int e in ascending)
            {
                if (kept[e] == 0) continue;

                int root1 = Find(parent, from[e]);
                int root2 = Find(parent, to[e]);

                if (root1 != root2)
                {
                    Union(parent, rank, clusterSize, root1, root2);
                }
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
                labels[i] = rootToLabel[root];
            }
        }

        _labels = labels;

        // Store max core distance for normalization
        _maxCoreDistance = engine.TensorMaxValue(coreDistances);
        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(_maxCoreDistance, epsilon)) _maxCoreDistance = NumOps.One;

        // Noise scores 1, cluster members core distance / max core distance
        _outlierScores = engine.TensorWhere(NoiseMask(labels), Tensor<T>.CreateOnes(new[] { n }),
            engine.TensorDivideScalar(coreDistances, _maxCoreDistance)).ToVector();
    }

    // 1 for noise points (label -1), else 0.
    private static Tensor<T> NoiseMask(int[] labels)
    {
        var noise = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            noise[i] = labels[i] == -1 ? 1 : 0;
        }

        return Tensor<int>.FromVector(Vector<int>.Wrap(noise), new[] { labels.Length }).Cast<T>();
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

        // Nearest training point (first on ties): noise scores 1, otherwise distance / max core distance,
        // clamped to at most 1
        var engine = Engine;
        var distances = AnomalyTensorOps.PairwiseDistances(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(trainingData));
        var nearestIdx = engine.TensorArgMin(distances, 1);
        var minDist = engine.TensorAmin(distances, new[] { 1 }, false);
        var score = engine.TensorWhere(engine.TensorGather(NoiseMask(labels), nearestIdx, 0),
            Tensor<T>.CreateOnes(new[] { X.Rows }), engine.TensorDivideScalar(minDist, _maxCoreDistance));
        return engine.TensorMin(score, NumOps.One).ToVector();
    }
}
