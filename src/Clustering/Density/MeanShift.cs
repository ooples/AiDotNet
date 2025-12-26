using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.SpatialIndex;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// Mean Shift clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mean Shift is a mode-seeking algorithm that iteratively shifts points toward
/// the weighted mean of points within a bandwidth radius. Points that converge
/// to the same mode form a cluster.
/// </para>
/// <para>
/// Algorithm:
/// 1. Start with each point as a potential mode
/// 2. For each point, compute weighted mean of neighbors within bandwidth
/// 3. Shift point toward this mean
/// 4. Repeat until convergence
/// 5. Merge nearby modes to get final cluster centers
/// </para>
/// <para><b>For Beginners:</b> Mean Shift finds the centers of data "clumps".
///
/// Think of it like:
/// 1. Drop a ball at each data point
/// 2. Each ball rolls uphill (toward denser areas)
/// 3. Balls that end up at the same peak belong to the same cluster
///
/// Benefits:
/// - Doesn't need to know number of clusters
/// - Finds natural cluster shapes
/// - Robust to outliers
///
/// Limitations:
/// - Bandwidth selection is important
/// - Can be slow for large datasets
/// </para>
/// </remarks>
public class MeanShift<T> : ClusteringBase<T>
{
    private readonly MeanShiftOptions<T> _options;
    private double _bandwidth;

    /// <summary>
    /// Initializes a new MeanShift instance.
    /// </summary>
    /// <param name="options">The MeanShift options.</param>
    public MeanShift(MeanShiftOptions<T>? options = null)
        : base(options ?? new MeanShiftOptions<T>())
    {
        _options = options ?? new MeanShiftOptions<T>();
    }

    /// <summary>
    /// Gets the bandwidth used for clustering.
    /// </summary>
    public double Bandwidth => _bandwidth;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new MeanShift<T>(new MeanShiftOptions<T>
        {
            Bandwidth = _options.Bandwidth,
            BandwidthQuantile = _options.BandwidthQuantile,
            ClusterMergeThreshold = _options.ClusterMergeThreshold,
            BinSeeding = _options.BinSeeding,
            ClusterAll = _options.ClusterAll,
            MinBinFrequency = _options.MinBinFrequency,
            Algorithm = _options.Algorithm,
            LeafSize = _options.LeafSize,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (MeanShift<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Estimate bandwidth if not provided
        _bandwidth = _options.Bandwidth ?? EstimateBandwidth(x);

        // Convert data to double
        var data = new double[n, d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                data[i, j] = NumOps.ToDouble(x[i, j]);
            }
        }

        // Get seeds (either binned or all points)
        double[,] seeds;
        if (_options.BinSeeding)
        {
            seeds = GetBinnedSeeds(data, n, d);
        }
        else
        {
            seeds = (double[,])data.Clone();
        }

        int numSeeds = seeds.GetLength(0);

        // Mean shift iterations for each seed
        var convergedCenters = new List<double[]>();

        for (int s = 0; s < numSeeds; s++)
        {
            var center = new double[d];
            for (int j = 0; j < d; j++)
            {
                center[j] = seeds[s, j];
            }

            // Iterate until convergence
            for (int iter = 0; iter < Options.MaxIterations; iter++)
            {
                var newCenter = ComputeMeanShift(data, center, n, d);

                // Check convergence
                double shift = 0;
                for (int j = 0; j < d; j++)
                {
                    double diff = newCenter[j] - center[j];
                    shift += diff * diff;
                }
                shift = Math.Sqrt(shift);

                center = newCenter;

                if (shift < Options.Tolerance)
                {
                    break;
                }
            }

            convergedCenters.Add(center);
        }

        // Merge nearby centers
        double mergeThreshold = _options.ClusterMergeThreshold ?? _bandwidth;
        var finalCenters = MergeCenters(convergedCenters, mergeThreshold, d);

        NumClusters = finalCenters.Count;

        // Set cluster centers
        ClusterCenters = new Matrix<T>(NumClusters, d);
        for (int k = 0; k < NumClusters; k++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[k, j] = NumOps.FromDouble(finalCenters[k][j]);
            }
        }

        // Assign labels
        Labels = new Vector<T>(n);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearestCluster = 0;

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

            Labels[i] = NumOps.FromDouble(nearestCluster);
        }

        IsTrained = true;
    }

    private double EstimateBandwidth(Matrix<T> x)
    {
        int n = x.Rows;
        int d = x.Columns;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Sample a subset of points for efficiency
        int sampleSize = Math.Min(n, 500);
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(sampleSize).ToList();

        // Compute pairwise distances
        var distances = new List<double>();
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = i + 1; j < indices.Count; j++)
            {
                var p1 = GetRow(x, indices[i]);
                var p2 = GetRow(x, indices[j]);
                double dist = NumOps.ToDouble(metric.Compute(p1, p2));
                distances.Add(dist);
            }
        }

        distances.Sort();

        // Return the quantile distance
        int idx = (int)(distances.Count * _options.BandwidthQuantile);
        return distances[Math.Max(0, Math.Min(idx, distances.Count - 1))];
    }

    private double[,] GetBinnedSeeds(double[,] data, int n, int d)
    {
        // Create bins with width = bandwidth
        double binWidth = _bandwidth;

        // Compute bin indices for each point
        var binCounts = new Dictionary<string, (double[] Sum, int Count)>();

        for (int i = 0; i < n; i++)
        {
            var binKey = new int[d];
            for (int j = 0; j < d; j++)
            {
                binKey[j] = (int)Math.Floor(data[i, j] / binWidth);
            }

            string key = string.Join(",", binKey);

            if (!binCounts.ContainsKey(key))
            {
                binCounts[key] = (new double[d], 0);
            }

            var (sum, count) = binCounts[key];
            for (int j = 0; j < d; j++)
            {
                sum[j] += data[i, j];
            }
            binCounts[key] = (sum, count + 1);
        }

        // Filter bins with minimum frequency and compute centroids
        var seeds = new List<double[]>();
        foreach (var kvp in binCounts)
        {
            if (kvp.Value.Count >= _options.MinBinFrequency)
            {
                var centroid = new double[d];
                for (int j = 0; j < d; j++)
                {
                    centroid[j] = kvp.Value.Sum[j] / kvp.Value.Count;
                }
                seeds.Add(centroid);
            }
        }

        // Convert to 2D array
        var result = new double[seeds.Count, d];
        for (int i = 0; i < seeds.Count; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = seeds[i][j];
            }
        }

        return result;
    }

    private double[] ComputeMeanShift(double[,] data, double[] center, int n, int d)
    {
        double bandwidthSq = _bandwidth * _bandwidth;
        var newCenter = new double[d];
        double totalWeight = 0;

        for (int i = 0; i < n; i++)
        {
            // Compute squared distance
            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = data[i, j] - center[j];
                distSq += diff * diff;
            }

            // Gaussian kernel weight
            if (distSq <= bandwidthSq)
            {
                double weight = Math.Exp(-0.5 * distSq / bandwidthSq);
                totalWeight += weight;

                for (int j = 0; j < d; j++)
                {
                    newCenter[j] += weight * data[i, j];
                }
            }
        }

        if (totalWeight > 0)
        {
            for (int j = 0; j < d; j++)
            {
                newCenter[j] /= totalWeight;
            }
        }
        else
        {
            // No neighbors, keep current center
            newCenter = (double[])center.Clone();
        }

        return newCenter;
    }

    private List<double[]> MergeCenters(List<double[]> centers, double threshold, int d)
    {
        var merged = new List<double[]>();
        var used = new bool[centers.Count];

        for (int i = 0; i < centers.Count; i++)
        {
            if (used[i]) continue;

            // Find all centers close to this one
            var cluster = new List<double[]> { centers[i] };
            used[i] = true;

            for (int j = i + 1; j < centers.Count; j++)
            {
                if (used[j]) continue;

                double distSq = 0;
                for (int k = 0; k < d; k++)
                {
                    double diff = centers[i][k] - centers[j][k];
                    distSq += diff * diff;
                }

                if (Math.Sqrt(distSq) < threshold)
                {
                    cluster.Add(centers[j]);
                    used[j] = true;
                }
            }

            // Compute mean of cluster
            var mean = new double[d];
            foreach (var c in cluster)
            {
                for (int k = 0; k < d; k++)
                {
                    mean[k] += c[k];
                }
            }
            for (int k = 0; k < d; k++)
            {
                mean[k] /= cluster.Count;
            }

            merged.Add(mean);
        }

        return merged;
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
            int nearestCluster = 0;

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
}
