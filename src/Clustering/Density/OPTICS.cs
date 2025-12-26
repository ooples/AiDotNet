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
public class OPTICS<T> : ClusteringBase<T>
{
    private readonly OPTICSOptions<T> _options;
    private double[]? _reachabilityDistances;
    private double[]? _coreDistances;
    private int[]? _ordering;
    private int[]? _predecessor;

    /// <summary>
    /// Noise label for points not in any cluster.
    /// </summary>
    public const int NoiseLabel = -1;

    /// <summary>
    /// Initializes a new OPTICS instance.
    /// </summary>
    /// <param name="options">The OPTICS options.</param>
    public OPTICS(OPTICSOptions<T>? options = null)
        : base(options ?? new OPTICSOptions<T>())
    {
        _options = options ?? new OPTICSOptions<T>();
    }

    /// <summary>
    /// Gets the reachability distances in ordering.
    /// </summary>
    public double[]? ReachabilityDistances => _reachabilityDistances;

    /// <summary>
    /// Gets the core distances for each point.
    /// </summary>
    public double[]? CoreDistances => _coreDistances;

    /// <summary>
    /// Gets the cluster ordering.
    /// </summary>
    public int[]? Ordering => _ordering;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OPTICS<T>(new OPTICSOptions<T>
        {
            MinSamples = _options.MinSamples,
            MaxEpsilon = _options.MaxEpsilon,
            ExtractionMethod = _options.ExtractionMethod,
            Xi = _options.Xi,
            ClusterEpsilon = _options.ClusterEpsilon,
            Algorithm = _options.Algorithm,
            LeafSize = _options.LeafSize,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
        NumFeatures = x.Columns;

        if (n < _options.MinSamples)
        {
            throw new ArgumentException($"Need at least {_options.MinSamples} samples.");
        }

        // Initialize arrays
        _reachabilityDistances = new double[n];
        _coreDistances = new double[n];
        _ordering = new int[n];
        _predecessor = new int[n];
        var processed = new bool[n];

        for (int i = 0; i < n; i++)
        {
            _reachabilityDistances[i] = double.PositiveInfinity;
            _coreDistances[i] = double.PositiveInfinity;
            _predecessor[i] = -1;
        }

        // Build spatial index
        var neighborFinder = CreateNeighborFinder(x);

        // Compute core distances
        T maxEpsT = NumOps.FromDouble(_options.MaxEpsilon);
        for (int i = 0; i < n; i++)
        {
            var query = GetRow(x, i);
            var neighbors = FindNeighbors(neighborFinder, x, query, maxEpsT);

            if (neighbors.Count >= _options.MinSamples)
            {
                // Core distance is the distance to the MinSamples-th nearest neighbor
                var distances = new List<double>();
                var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

                foreach (int neighborIdx in neighbors)
                {
                    var neighbor = GetRow(x, neighborIdx);
                    double dist = NumOps.ToDouble(metric.Compute(query, neighbor));
                    distances.Add(dist);
                }

                distances.Sort();
                _coreDistances[i] = distances[Math.Min(_options.MinSamples - 1, distances.Count - 1)];
            }
        }

        // OPTICS ordering algorithm
        int orderIndex = 0;
        var orderSeeds = new SortedSet<(double Reachability, int Index)>(
            Comparer<(double, int)>.Create((a, b) =>
            {
                int cmp = a.Item1.CompareTo(b.Item1);
                return cmp != 0 ? cmp : a.Item2.CompareTo(b.Item2);
            }));

        for (int i = 0; i < n; i++)
        {
            if (processed[i]) continue;

            processed[i] = true;
            _ordering[orderIndex++] = i;

            if (!double.IsPositiveInfinity(_coreDistances[i]))
            {
                UpdateSeeds(x, i, orderSeeds, processed, neighborFinder, maxEpsT);

                while (orderSeeds.Count > 0)
                {
                    var (reachability, currentIdx) = orderSeeds.Min;
                    orderSeeds.Remove(orderSeeds.Min);

                    if (processed[currentIdx]) continue;

                    processed[currentIdx] = true;
                    _ordering[orderIndex++] = currentIdx;

                    if (!double.IsPositiveInfinity(_coreDistances[currentIdx]))
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

        IsTrained = true;
    }

    private void UpdateSeeds(
        Matrix<T> x,
        int centerIdx,
        SortedSet<(double, int)> seeds,
        bool[] processed,
        object neighborFinder,
        T maxEps)
    {
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centerPoint = GetRow(x, centerIdx);
        var neighbors = FindNeighbors(neighborFinder, x, centerPoint, maxEps);

        double coreDist = _coreDistances![centerIdx];

        foreach (int neighborIdx in neighbors)
        {
            if (processed[neighborIdx]) continue;

            var neighborPoint = GetRow(x, neighborIdx);
            double dist = NumOps.ToDouble(metric.Compute(centerPoint, neighborPoint));
            double reachDist = Math.Max(coreDist, dist);

            if (reachDist < _reachabilityDistances![neighborIdx])
            {
                // Remove old entry if exists
                seeds.Remove((_reachabilityDistances[neighborIdx], neighborIdx));

                _reachabilityDistances[neighborIdx] = reachDist;
                _predecessor![neighborIdx] = centerIdx;

                seeds.Add((reachDist, neighborIdx));
            }
        }
    }

    private void ExtractClusters(int n)
    {
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(NoiseLabel);
        }

        if (_options.ExtractionMethod == OPTICSExtractionMethod.DbscanStyle)
        {
            ExtractDbscanStyle(n);
        }
        else
        {
            ExtractXiMethod(n);
        }
    }

    private void ExtractDbscanStyle(int n)
    {
        double eps = _options.ClusterEpsilon ?? _options.MaxEpsilon;
        int currentCluster = 0;

        for (int i = 0; i < n; i++)
        {
            int pointIdx = _ordering![i];

            if (_reachabilityDistances![pointIdx] > eps)
            {
                // Start new cluster or noise
                if (!double.IsPositiveInfinity(_coreDistances![pointIdx]) && _coreDistances[pointIdx] <= eps)
                {
                    Labels![pointIdx] = NumOps.FromDouble(currentCluster);
                    currentCluster++;
                }
            }
            else
            {
                // Assign to current cluster
                if (currentCluster > 0)
                {
                    Labels![pointIdx] = NumOps.FromDouble(currentCluster - 1);
                }
            }
        }

        NumClusters = currentCluster;
    }

    private void ExtractXiMethod(int n)
    {
        double xi = _options.Xi;
        int minClusterSize = Math.Max(2, (int)(n * _options.MinClusterSizeFraction));

        var steepAreas = new List<(int Start, int End, bool IsUp)>();

        // Find steep up and down areas
        for (int i = 1; i < n; i++)
        {
            int prevIdx = _ordering![i - 1];
            int currIdx = _ordering![i];

            double prevReach = _reachabilityDistances![prevIdx];
            double currReach = _reachabilityDistances![currIdx];

            if (double.IsPositiveInfinity(prevReach)) prevReach = _options.MaxEpsilon;
            if (double.IsPositiveInfinity(currReach)) currReach = _options.MaxEpsilon;

            bool steepUp = currReach >= prevReach * (1 + xi);
            bool steepDown = currReach <= prevReach * (1 - xi);

            if (steepUp || steepDown)
            {
                int start = i - 1;
                int end = i;

                // Extend the steep area
                while (end < n - 1)
                {
                    int nextPrevIdx = _ordering![end];
                    int nextCurrIdx = _ordering![end + 1];

                    double nextPrevReach = _reachabilityDistances![nextPrevIdx];
                    double nextCurrReach = _reachabilityDistances![nextCurrIdx];

                    if (double.IsPositiveInfinity(nextPrevReach)) nextPrevReach = _options.MaxEpsilon;
                    if (double.IsPositiveInfinity(nextCurrReach)) nextCurrReach = _options.MaxEpsilon;

                    bool stillSteep = steepUp
                        ? nextCurrReach >= nextPrevReach * (1 + xi)
                        : nextCurrReach <= nextPrevReach * (1 - xi);

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

        // Assign labels based on clusters
        int currentCluster = 0;
        foreach (var (start, end) in clusters)
        {
            for (int i = start; i <= end && i < n; i++)
            {
                int pointIdx = _ordering![i];
                if (NumOps.ToDouble(Labels![pointIdx]) < 0)
                {
                    Labels![pointIdx] = NumOps.FromDouble(currentCluster);
                }
            }
            currentCluster++;
        }

        NumClusters = currentCluster;
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
        var neighbors = new List<int>();
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        double epsilonDouble = NumOps.ToDouble(epsilon);

        for (int i = 0; i < data.Rows; i++)
        {
            var point = GetRow(data, i);
            T dist = metric.Compute(query, point);
            if (NumOps.ToDouble(dist) <= epsilonDouble)
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

        ClusterCenters = new Matrix<T>(NumClusters, x.Columns);
        var counts = new int[NumClusters];

        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(Labels[i]);
            if (cluster >= 0 && cluster < NumClusters)
            {
                counts[cluster]++;
                for (int j = 0; j < x.Columns; j++)
                {
                    ClusterCenters[cluster, j] = NumOps.Add(ClusterCenters[cluster, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
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

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        var labels = new Vector<T>(x.Rows);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

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
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = k;
                    }
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

    /// <summary>
    /// Gets the reachability plot data.
    /// </summary>
    /// <returns>Array of (ordering index, reachability distance) tuples.</returns>
    public (int Index, double Reachability)[] GetReachabilityPlot()
    {
        ValidateIsTrained();

        int n = _ordering!.Length;
        var plot = new (int, double)[n];

        for (int i = 0; i < n; i++)
        {
            int pointIdx = _ordering[i];
            double reach = _reachabilityDistances![pointIdx];
            if (double.IsPositiveInfinity(reach))
            {
                reach = _options.MaxEpsilon;
            }
            plot[i] = (pointIdx, reach);
        }

        return plot;
    }

    /// <summary>
    /// Extracts clusters at a different epsilon value.
    /// </summary>
    /// <param name="epsilon">New epsilon for cluster extraction.</param>
    /// <returns>New cluster labels.</returns>
    public Vector<T> ExtractClustersAtEpsilon(double epsilon)
    {
        ValidateIsTrained();

        int n = _ordering!.Length;
        var labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            labels[i] = NumOps.FromDouble(NoiseLabel);
        }

        int currentCluster = 0;

        for (int i = 0; i < n; i++)
        {
            int pointIdx = _ordering[i];

            if (_reachabilityDistances![pointIdx] > epsilon)
            {
                if (!double.IsPositiveInfinity(_coreDistances![pointIdx]) && _coreDistances[pointIdx] <= epsilon)
                {
                    labels[pointIdx] = NumOps.FromDouble(currentCluster);
                    currentCluster++;
                }
            }
            else
            {
                if (currentCluster > 0)
                {
                    labels[pointIdx] = NumOps.FromDouble(currentCluster - 1);
                }
            }
        }

        return labels;
    }
}
