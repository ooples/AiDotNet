using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.SpatialIndex;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// OPTICS (Ordering Points To Identify the Clustering Structure) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OPTICS is a density-based clustering algorithm that creates an ordering of points
/// based on their density-reachability. Unlike DBSCAN, it doesn't require a fixed
/// epsilon and can reveal hierarchical clustering structure.
/// </para>
/// <para>
/// Key concepts:
/// - Core distance: Minimum radius to include MinSamples points
/// - Reachability distance: max(core distance, distance to point)
/// - Cluster ordering: Sequence of points with reachability distances
/// </para>
/// <para><b>For Beginners:</b> OPTICS creates a "profile" of your data's cluster structure.
///
/// Imagine walking through a landscape:
/// - In dense areas (clusters), reachability is low
/// - At cluster boundaries, reachability spikes up
/// - This creates a "reachability plot" showing cluster structure
///
/// Benefits:
/// - No need to choose epsilon beforehand
/// - Can find clusters at multiple scales
/// - Better for clusters with varying densities
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OPTICS&lt;double&gt;(new OPTICSOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("OPTICS: Ordering Points To Identify the Clustering Structure", "https://doi.org/10.1145/304181.304187", Year = 1999, Authors = "Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jorg Sander")]
public partial class OPTICS<T> : ClusteringBase<T>
{
    private readonly OPTICSOptions<T> _options;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureMeans;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureStds;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _normalizedClusterCenters;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _reachabilityDistances = new Vector<T>(0);
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _coreDistances = new Vector<T>(0);
    private int[] _ordering = Array.Empty<int>();
    private int[] _predecessor = Array.Empty<int>();

    /// <summary>
    /// Noise label for points not in any cluster.
    /// </summary>
    public const int NoiseLabel = -1;

    /// <summary>
    /// Initializes a new OPTICS instance.
    /// </summary>
    /// <param name="options">The OPTICS options.</param>
    public OPTICS(OPTICSOptions<T>? options = null)
        : base(options ??= new OPTICSOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the reachability distances in ordering.
    /// </summary>
    public Vector<T> ReachabilityDistances => _reachabilityDistances;

    /// <summary>
    /// Gets the core distances for each point.
    /// </summary>
    public Vector<T> CoreDistances => _coreDistances;

    /// <summary>
    /// Gets the cluster ordering.
    /// </summary>
    public int[] Ordering => _ordering;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (OPTICS<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        if (n < _options.MinSamples)
        {
            throw new ArgumentException($"Need at least {_options.MinSamples} samples.");
        }

        // Normalize features for scale-invariant density ordering
        x = ClusteringTensorOps.StandardizeColumns(x, out var featureMeans, out var featureStds);
        _featureMeans = featureMeans;
        _featureStds = featureStds;

        // Initialize arrays
        T infinity = NumOps.MaxValue;
        _reachabilityDistances = Engine.Fill(n, infinity);
        _ordering = new int[n];
        _predecessor = new int[n];
        var processed = new bool[n];

        for (int i = 0; i < n; i++)
        {
            _predecessor[i] = -1;
        }

        // Build spatial index
        var neighborFinder = CreateNeighborFinder(x);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Compute core distances
        T maxEpsT = NumOps.FromDouble(_options.MaxEpsilon);
        var coreDistances = new List<T>(n);
        for (int i = 0; i < n; i++)
        {
            var query = GetRow(x, i);
            var neighbors = FindNeighbors(neighborFinder, x, query, maxEpsT);
            coreDistances.Add(neighbors.Count >= _options.MinSamples
                ? CoreDistance(x, query, neighbors, metric)
                : infinity);
        }

        _coreDistances = Vector<T>.FromList(coreDistances);
        var isCore = ClusteringTensorOps.LessThanFlags(_coreDistances, infinity);

        // OPTICS ordering algorithm (seeds ordered by reachability, then index)
        int orderIndex = 0;
        var orderSeeds = new SortedSet<(T Reachability, int Index)>(
            Comparer<(T, int)>.Create(CompareSeeds));

        for (int i = 0; i < n; i++)
        {
            if (processed[i]) continue;

            processed[i] = true;
            _ordering[orderIndex++] = i;

            if (isCore[i])
            {
                UpdateSeeds(x, i, orderSeeds, processed, neighborFinder, maxEpsT);

                while (orderSeeds.Count > 0)
                {
                    var (reachability, currentIdx) = orderSeeds.Min;
                    orderSeeds.Remove(orderSeeds.Min);

                    if (processed[currentIdx]) continue;

                    processed[currentIdx] = true;
                    _ordering[orderIndex++] = currentIdx;

                    if (isCore[currentIdx])
                    {
                        UpdateSeeds(x, currentIdx, orderSeeds, processed, neighborFinder, maxEpsT);
                    }
                }
            }
        }

        // Extract clusters
        ExtractClusters(n);

        // Compute cluster centers
        ComputeClusterCenters(x);

        // Save normalized centers for Predict (where input is also normalized)
        if (_featureMeans is not null && _featureStds is not null && ClusterCenters is not null)
        {
            _normalizedClusterCenters = ClusterCenters.Clone();

            // De-normalize cluster centers back to original space for user consumption
            ClusterCenters = ClusteringTensorOps.Destandardize(ClusterCenters, _featureMeans, _featureStds);
        }

        IsTrained = true;
    }

    /// <summary>Distance from the query to its MinSamples-th nearest neighbor (or the farthest one when there are fewer).</summary>
    private T CoreDistance(Matrix<T> x, Vector<T> query, List<int> neighbors, IDistanceMetric<T> metric)
    {
        var rows = ClusteringTensorOps.SelectRows(x, neighbors.ToArray());
        var sorted = Engine.TensorMsort(ClusteringTensorOps.DistancesToPoint(metric, rows, query).AsTensor()).ToVector();
        return sorted[Math.Min(_options.MinSamples - 1, sorted.Length - 1)];
    }

    /// <summary>Seed order: reachability, then index.</summary>
    private int CompareSeeds((T Reachability, int Index) a, (T Reachability, int Index) b)
    {
        int cmp = NumOps.Compare(a.Reachability, b.Reachability);
        return cmp != 0 ? cmp : a.Index.CompareTo(b.Index);
    }

    private void UpdateSeeds(
        Matrix<T> x,
        int centerIdx,
        SortedSet<(T, int)> seeds,
        bool[] processed,
        object neighborFinder,
        T maxEps)
    {
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centerPoint = GetRow(x, centerIdx);
        var candidates = FindNeighbors(neighborFinder, x, centerPoint, maxEps)
            .Where(neighborIdx => !processed[neighborIdx])
            .ToArray();
        if (candidates.Length == 0)
        {
            return;
        }

        // reach(o) = max(core(center), d(center, o)); keep the smaller of the old and new reachability
        T coreDist = _coreDistances[centerIdx];
        var distances = ClusteringTensorOps.DistancesToPoint(metric, ClusteringTensorOps.SelectRows(x, candidates), centerPoint);
        var reach = Engine.TensorMax(distances.AsTensor(), coreDist);
        var current = ClusteringTensorOps.SelectElements(_reachabilityDistances, candidates).AsTensor();
        var improved = ClusteringTensorOps.ToBoolArray(Engine.TensorLessThan(reach, current).ToVector());

        var oldKeys = current.ToArray();
        var newKeys = reach.ToArray();
        for (int j = 0; j < candidates.Length; j++)
        {
            if (!improved[j]) continue;

            // Re-key the seed
            seeds.Remove((oldKeys[j], candidates[j]));
            _predecessor[candidates[j]] = centerIdx;
            seeds.Add((newKeys[j], candidates[j]));
        }

        _reachabilityDistances = Engine.Scatter(
            _reachabilityDistances.AsTensor(),
            Vector<int>.Wrap(candidates).AsTensor(),
            Engine.TensorWhere(Engine.TensorLessThan(reach, current), reach, current),
            0).ToVector();
    }

    private void ExtractClusters(int n)
    {
        int[] labels;
        int numClusters;
        if (_options.ExtractionMethod == OPTICSExtractionMethod.DbscanStyle)
        {
            labels = DbscanStyleLabels(NumOps.FromDouble(_options.ClusterEpsilon ?? _options.MaxEpsilon), out numClusters);
        }
        else
        {
            labels = XiLabels(n, out numClusters);
        }

        Labels = ClusteringTensorOps.FromIntArray<T>(labels);
        NumClusters = numClusters;
    }

    /// <summary>
    /// DBSCAN-style extraction along the ordering: a reachability jump above eps starts a cluster at a core point
    /// (core distance at most eps) and otherwise leaves noise; smaller reachabilities join the current cluster.
    /// </summary>
    private int[] DbscanStyleLabels(T eps, out int numClusters)
    {
        int n = _ordering.Length;
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = NoiseLabel;
        }

        if (n == 0)
        {
            numClusters = 0;
            return labels;
        }

        var reach = ClusteringTensorOps.SelectElements(_reachabilityDistances, _ordering).AsTensor();
        var core = ClusteringTensorOps.SelectElements(_coreDistances, _ordering).AsTensor();
        var jumps = ClusteringTensorOps.ToBoolArray(Engine.TensorGreaterThan(reach, eps).ToVector());
        var corePoints = ClusteringTensorOps.ToBoolArray(Engine.TensorMultiply(
            Engine.TensorLessThan(core, NumOps.MaxValue),
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorGreaterThan(core, eps))).ToVector());

        int currentCluster = 0;
        for (int i = 0; i < n; i++)
        {
            int pointIdx = _ordering[i];

            if (jumps[i])
            {
                // Start new cluster or noise
                if (corePoints[i])
                {
                    labels[pointIdx] = currentCluster;
                    currentCluster++;
                }
            }
            else if (currentCluster > 0)
            {
                // Assign to current cluster
                labels[pointIdx] = currentCluster - 1;
            }
        }

        numClusters = currentCluster;
        return labels;
    }

    /// <summary>Reachabilities in ordering order with undefined (infinite) values replaced by MaxEpsilon.</summary>
    private Tensor<T> OrderedReachability()
    {
        var reach = ClusteringTensorOps.SelectElements(_reachabilityDistances, _ordering).AsTensor();
        var defined = Engine.TensorLessThan(reach, NumOps.MaxValue);
        return Engine.TensorWhere(defined, reach, Engine.TensorFullLike(reach, NumOps.FromDouble(_options.MaxEpsilon)));
    }

    private int[] XiLabels(int n, out int numClusters)
    {
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = NoiseLabel;
        }

        int minClusterSize = Math.Max(2, (int)(n * _options.MinClusterSizeFraction));

        // Steepness of every consecutive pair (p, p + 1) of the ordering
        var steepUpPairs = new bool[Math.Max(0, n - 1)];
        var steepDownPairs = new bool[Math.Max(0, n - 1)];
        if (n > 1)
        {
            T xi = NumOps.FromDouble(_options.Xi);
            var ordered = OrderedReachability();
            var prev = Engine.TensorSlice(ordered, new[] { 0 }, new[] { n - 1 });
            var curr = Engine.TensorSlice(ordered, new[] { 1 }, new[] { n - 1 });

            // steep up: curr >= prev (1 + xi); steep down: curr <= prev (1 - xi)
            var upThreshold = Engine.TensorMultiplyScalar(prev, NumOps.Add(NumOps.One, xi));
            var downThreshold = Engine.TensorMultiplyScalar(prev, NumOps.Subtract(NumOps.One, xi));
            steepUpPairs = ClusteringTensorOps.ToBoolArray(Engine.TensorLessThan(curr, upThreshold).ToVector(), invert: true);
            steepDownPairs = ClusteringTensorOps.ToBoolArray(Engine.TensorGreaterThan(curr, downThreshold).ToVector(), invert: true);
        }

        var steepAreas = new List<(int Start, int End, bool IsUp)>();

        // Find steep up and down areas
        for (int i = 1; i < n; i++)
        {
            bool steepUp = steepUpPairs[i - 1];
            bool steepDown = steepDownPairs[i - 1];

            if (steepUp || steepDown)
            {
                int start = i - 1;
                int end = i;

                // Extend the steep area
                while (end < n - 1)
                {
                    bool stillSteep = steepUp ? steepUpPairs[end] : steepDownPairs[end];
                    if (!stillSteep) break;
                    end++;
                }

                if (end - start >= 1)
                {
                    steepAreas.Add((start, end, steepUp));
                }

                i = end;
            }
        }

        // Match steep down areas with steep up areas to form clusters
        var clusters = new List<(int Start, int End)>();
        var downAreas = steepAreas.Where(a => !a.IsUp).ToList();
        var upAreas = steepAreas.Where(a => a.IsUp).ToList();

        foreach (var down in downAreas)
        {
            foreach (var up in upAreas)
            {
                if (up.Start > down.End)
                {
                    int clusterStart = down.Start;
                    int clusterEnd = up.End;

                    if (clusterEnd - clusterStart >= minClusterSize)
                    {
                        clusters.Add((clusterStart, clusterEnd));
                    }
                    break;
                }
            }
        }

        // Assign labels based on clusters (earlier clusters keep their points)
        int currentCluster = 0;
        foreach (var (start, end) in clusters)
        {
            for (int i = start; i <= end && i < n; i++)
            {
                int pointIdx = _ordering[i];
                if (labels[pointIdx] < 0)
                {
                    labels[pointIdx] = currentCluster;
                }
            }
            currentCluster++;
        }

        numClusters = currentCluster;
        return labels;
    }

    private object CreateNeighborFinder(Matrix<T> x)
    {
        var algorithm = _options.Algorithm;

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
                return x;
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
            return FindNeighborsBruteForce(data, query, epsilon);
        }
    }

    private List<int> FindNeighborsBruteForce(Matrix<T> data, Vector<T> query, T epsilon)
    {
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var within = ClusteringTensorOps.LessOrEqualFlags(ClusteringTensorOps.DistancesToPoint(metric, data, query), epsilon);

        var neighbors = new List<int>();
        for (int i = 0; i < within.Length; i++)
        {
            if (within[i])
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters <= 0 || Labels is null)
        {
            return;
        }

        // Mean of each cluster's members (noise is ignored; empty clusters stay at zero)
        ClusterCenters = ClusteringTensorOps.ClusterMeans(x, Labels, NumClusters, out _);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Normalize input using saved parameters from training
        if (_featureMeans is not null && _featureStds is not null)
        {
            x = ClusteringTensorOps.Standardize(x, _featureMeans, _featureStds);
        }

        // Use normalized centers when input was normalized
        var centersForPrediction = _normalizedClusterCenters ?? ClusterCenters;
        if (centersForPrediction is null || NumClusters <= 0 || x.Rows == 0)
        {
            var noise = new Vector<T>(x.Rows);
            return x.Rows == 0
                ? noise
                : Engine.TensorFullLike(noise.AsTensor(), NumOps.FromDouble(NoiseLabel)).ToVector();
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = centersForPrediction.Rows == NumClusters
            ? centersForPrediction
            : centersForPrediction.GetSubMatrix(0, 0, NumClusters, centersForPrediction.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the reachability plot data.
    /// </summary>
    /// <returns>Array of (ordering index, reachability distance) tuples.</returns>
    public (int Index, T Reachability)[] GetReachabilityPlot()
    {
        ValidateIsTrained();

        if (_ordering.Length == 0)
        {
            return Array.Empty<(int, T)>();
        }

        // Undefined (infinite) reachabilities are plotted at MaxEpsilon
        var reach = OrderedReachability().ToArray();
        return _ordering.Zip(reach, (pointIdx, r) => (pointIdx, r)).ToArray();
    }

    /// <summary>
    /// Extracts clusters at a different epsilon value.
    /// </summary>
    /// <param name="epsilon">New epsilon for cluster extraction.</param>
    /// <returns>New cluster labels.</returns>
    public Vector<T> ExtractClustersAtEpsilon(double epsilon)
    {
        ValidateIsTrained();

        return ClusteringTensorOps.FromIntArray<T>(DbscanStyleLabels(NumOps.FromDouble(epsilon), out _));
    }
}
