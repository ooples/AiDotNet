using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Hierarchical;

/// <summary>
/// CURE (Clustering Using REpresentatives) hierarchical clustering algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CURE is an agglomerative hierarchical clustering algorithm that uses multiple
/// representative points per cluster to better capture non-spherical cluster shapes.
/// Representatives are shrunk toward the cluster center to reduce sensitivity to outliers.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Start with each point as its own cluster
/// 2. Select representative points for each cluster (well-scattered)
/// 3. Shrink representatives toward cluster center
/// 4. Find and merge the two clusters with closest representatives
/// 5. Repeat until desired number of clusters is reached
/// </para>
/// <para><b>For Beginners:</b> CURE finds clusters that aren't round:
///
/// Traditional clustering (like K-Means) assumes round clusters.
/// But real data often has:
/// - Banana-shaped clusters
/// - Spiral clusters
/// - Elongated clusters
///
/// CURE solves this by:
/// 1. Using multiple "marker" points per cluster, not just one center
/// 2. Placing these markers throughout the cluster
/// 3. Measuring cluster similarity by comparing markers
///
/// This way, two banana-shaped clusters can be recognized as separate,
/// even if their centers are close together!
/// </para>
/// </remarks>
public class CURE<T> : ClusteringBase<T>
{
    private readonly CUREOptions<T> _options;
    private readonly IDistanceMetric<T> _distanceMetric;
    private Random _random;
    private List<CureCluster>? _clusters;

    /// <summary>
    /// Initializes a new CURE instance.
    /// </summary>
    /// <param name="options">The CURE configuration options.</param>
    public CURE(CUREOptions<T>? options = null)
        : base(options ?? new CUREOptions<T>())
    {
        // Use the options passed to base constructor to avoid double instantiation
        _options = (CUREOptions<T>)Options;
        _distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        _random = _options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);
        NumClusters = _options.NumClusters;
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CURE<T>(new CUREOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            RandomState = _options.RandomState,
            NumRepresentatives = _options.NumRepresentatives,
            ShrinkFactor = _options.ShrinkFactor,
            SampleFraction = _options.SampleFraction,
            UsePartitioning = _options.UsePartitioning,
            NumPartitions = _options.NumPartitions,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (CURE<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Sample data if needed
        Matrix<T> data;
        int[] sampleIndices;

        if (_options.SampleFraction < 1.0 && n > 100)
        {
            int sampleSize = Math.Max(_options.NumClusters, (int)(n * _options.SampleFraction));
            sampleIndices = Enumerable.Range(0, n).OrderBy(_ => _random.Next()).Take(sampleSize).ToArray();
            data = ExtractSubMatrix(x, sampleIndices);
        }
        else
        {
            sampleIndices = Enumerable.Range(0, n).ToArray();
            data = x;
        }

        int sampleN = data.Rows;

        // Validate NumClusters is within valid range
        if (_options.NumClusters < 1 || _options.NumClusters > sampleN)
        {
            throw new ArgumentException(
                $"NumClusters must be between 1 and {sampleN} (number of data points), got {_options.NumClusters}.");
        }

        // Initialize: each point is its own cluster
        _clusters = new List<CureCluster>();
        for (int i = 0; i < sampleN; i++)
        {
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(data[i, j]);
            }

            var cluster = new CureCluster
            {
                Points = new List<int> { i },
                Center = point,
                Representatives = new List<double[]> { (double[])point.Clone() }
            };
            _clusters.Add(cluster);
        }

        // Agglomerative clustering
        while (_clusters.Count > _options.NumClusters)
        {
            // Find the two closest clusters
            var (cluster1Idx, cluster2Idx) = FindClosestClusters();

            if (cluster1Idx < 0 || cluster2Idx < 0)
            {
                break; // No more clusters to merge
            }

            // Merge the two clusters
            var mergedCluster = MergeClusters(_clusters[cluster1Idx], _clusters[cluster2Idx], data);

            // Remove old clusters and add merged
            if (cluster1Idx > cluster2Idx)
            {
                _clusters.RemoveAt(cluster1Idx);
                _clusters.RemoveAt(cluster2Idx);
            }
            else
            {
                _clusters.RemoveAt(cluster2Idx);
                _clusters.RemoveAt(cluster1Idx);
            }

            _clusters.Add(mergedCluster);
        }

        NumClusters = _clusters.Count;

        // Assign labels to all original points
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(-1);
        }

        // Assign sampled points based on their cluster
        for (int clusterIdx = 0; clusterIdx < _clusters.Count; clusterIdx++)
        {
            foreach (int sampleIdx in _clusters[clusterIdx].Points)
            {
                int originalIdx = sampleIndices[sampleIdx];
                Labels[originalIdx] = NumOps.FromDouble(clusterIdx);
            }
        }

        // Assign non-sampled points to nearest cluster
        if (sampleIndices.Length < n)
        {
            var sampledSet = new HashSet<int>(sampleIndices);
            for (int i = 0; i < n; i++)
            {
                if (!sampledSet.Contains(i))
                {
                    var point = new double[d];
                    for (int j = 0; j < d; j++)
                    {
                        point[j] = NumOps.ToDouble(x[i, j]);
                    }

                    int nearestCluster = FindNearestCluster(point);
                    Labels[i] = NumOps.FromDouble(nearestCluster);
                }
            }
        }

        // Compute cluster centers
        ComputeClusterCenters(x);

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        int n = x.Rows;
        int d = x.Columns;
        var labels = new Vector<T>(n);

        if (_clusters is null || _clusters.Count == 0)
        {
            for (int i = 0; i < n; i++)
            {
                labels[i] = NumOps.FromDouble(-1);
            }
            return labels;
        }

        for (int i = 0; i < n; i++)
        {
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(x[i, j]);
            }

            int nearestCluster = FindNearestCluster(point);
            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    private (int, int) FindClosestClusters()
    {
        int bestI = -1, bestJ = -1;
        double minDistance = double.MaxValue;

        for (int i = 0; i < _clusters!.Count; i++)
        {
            for (int j = i + 1; j < _clusters.Count; j++)
            {
                double dist = ComputeClusterDistance(_clusters[i], _clusters[j]);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        return (bestI, bestJ);
    }

    private double ComputeClusterDistance(CureCluster c1, CureCluster c2)
    {
        // Minimum distance between any pair of representatives
        double minDist = double.MaxValue;

        foreach (var rep1 in c1.Representatives)
        {
            foreach (var rep2 in c2.Representatives)
            {
                double dist = ComputeDistance(rep1, rep2);
                minDist = Math.Min(minDist, dist);
            }
        }

        return minDist;
    }

    private double ComputeDistance(double[] a, double[] b)
    {
        // Convert to Vector<T> and use the configured distance metric
        var vecA = new Vector<T>(a.Length);
        var vecB = new Vector<T>(b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            vecA[i] = NumOps.FromDouble(a[i]);
            vecB[i] = NumOps.FromDouble(b[i]);
        }
        return NumOps.ToDouble(_distanceMetric.Compute(vecA, vecB));
    }

    private CureCluster MergeClusters(CureCluster c1, CureCluster c2, Matrix<T> data)
    {
        int d = data.Columns;

        // Combine points
        var mergedPoints = new List<int>(c1.Points);
        mergedPoints.AddRange(c2.Points);

        // Compute new center
        var center = new double[d];
        foreach (int idx in mergedPoints)
        {
            for (int j = 0; j < d; j++)
            {
                center[j] += NumOps.ToDouble(data[idx, j]);
            }
        }
        for (int j = 0; j < d; j++)
        {
            center[j] /= mergedPoints.Count;
        }

        // Select representatives using farthest-point heuristic
        var representatives = SelectRepresentatives(mergedPoints, data, center);

        // Shrink representatives toward center
        for (int i = 0; i < representatives.Count; i++)
        {
            for (int j = 0; j < d; j++)
            {
                representatives[i][j] = representatives[i][j] +
                    _options.ShrinkFactor * (center[j] - representatives[i][j]);
            }
        }

        return new CureCluster
        {
            Points = mergedPoints,
            Center = center,
            Representatives = representatives
        };
    }

    private List<double[]> SelectRepresentatives(List<int> points, Matrix<T> data, double[] center)
    {
        int d = data.Columns;
        int numReps = Math.Min(_options.NumRepresentatives, points.Count);
        var representatives = new List<double[]>();

        if (numReps == 0)
        {
            return representatives;
        }

        // Start with point farthest from center
        int farthestIdx = -1;
        double maxDist = -1;

        foreach (int idx in points)
        {
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(data[idx, j]);
            }

            double dist = ComputeDistance(point, center);
            if (dist > maxDist)
            {
                maxDist = dist;
                farthestIdx = idx;
            }
        }

        if (farthestIdx >= 0)
        {
            var point = new double[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(data[farthestIdx, j]);
            }
            representatives.Add(point);
        }

        // Add more representatives using farthest-point heuristic
        while (representatives.Count < numReps)
        {
            int nextIdx = -1;
            double maxMinDist = -1;

            foreach (int idx in points)
            {
                // Skip if already a representative
                bool isRep = false;
                var point = new double[d];
                for (int j = 0; j < d; j++)
                {
                    point[j] = NumOps.ToDouble(data[idx, j]);
                }

                foreach (var rep in representatives)
                {
                    if (ComputeDistance(point, rep) < 1e-10)
                    {
                        isRep = true;
                        break;
                    }
                }

                if (isRep) continue;

                // Find minimum distance to any existing representative
                double minDist = double.MaxValue;
                foreach (var rep in representatives)
                {
                    double dist = ComputeDistance(point, rep);
                    minDist = Math.Min(minDist, dist);
                }

                if (minDist > maxMinDist)
                {
                    maxMinDist = minDist;
                    nextIdx = idx;
                }
            }

            if (nextIdx >= 0)
            {
                var point = new double[d];
                for (int j = 0; j < d; j++)
                {
                    point[j] = NumOps.ToDouble(data[nextIdx, j]);
                }
                representatives.Add(point);
            }
            else
            {
                break; // No more points to add
            }
        }

        return representatives;
    }

    private int FindNearestCluster(double[] point)
    {
        int nearest = 0;
        double minDist = double.MaxValue;

        for (int i = 0; i < _clusters!.Count; i++)
        {
            foreach (var rep in _clusters[i].Representatives)
            {
                double dist = ComputeDistance(point, rep);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = i;
                }
            }
        }

        return nearest;
    }

    private void ComputeClusterCenters(Matrix<T> x)
    {
        if (NumClusters == 0)
        {
            ClusterCenters = null;
            return;
        }

        int d = x.Columns;
        int n = x.Rows;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        var counts = new int[NumClusters];

        for (int i = 0; i < n; i++)
        {
            int label = (int)NumOps.ToDouble(Labels![i]);
            if (label >= 0 && label < NumClusters)
            {
                counts[label]++;
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[label, j] = NumOps.Add(ClusterCenters[label, j], x[i, j]);
                }
            }
        }

        for (int k = 0; k < NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    ClusterCenters[k, j] = NumOps.Divide(ClusterCenters[k, j], NumOps.FromDouble(counts[k]));
                }
            }
        }
    }

    private Matrix<T> ExtractSubMatrix(Matrix<T> data, int[] indices)
    {
        var subMatrix = new Matrix<T>(indices.Length, data.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                subMatrix[i, j] = data[indices[i], j];
            }
        }
        return subMatrix;
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0)
        {
            throw new ArgumentException("Input data cannot be empty.");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (x.Columns != NumFeatures)
        {
            throw new ArgumentException($"Expected {NumFeatures} features, got {x.Columns}.");
        }
    }

    private class CureCluster
    {
        public List<int> Points { get; set; } = new List<int>();
        public double[] Center { get; set; } = Array.Empty<double>();
        public List<double[]> Representatives { get; set; } = new List<double[]>();
    }
}
