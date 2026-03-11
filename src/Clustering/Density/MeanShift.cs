using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Mean Shift: A Robust Approach toward Feature Space Analysis", "https://doi.org/10.1109/34.1000236", Year = 2002, Authors = "Dorin Comaniciu, Peter Meer")]
public class MeanShift<T> : ClusteringBase<T>
{
    private readonly MeanShiftOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T _bandwidth = MathHelper.GetNumericOperations<T>().Zero;

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
    public T Bandwidth => _bandwidth;

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
        _bandwidth = NumOps.FromDouble(_options.Bandwidth ?? EstimateBandwidth(x));
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        // Get seeds (either binned or all points)
        T[,] seeds;
        if (_options.BinSeeding)
        {
            seeds = GetBinnedSeeds(x, n, d);
        }
        else
        {
            var seedsCopy = new T[n, d];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    seedsCopy[i, j] = x[i, j];
            seeds = seedsCopy;
        }

        int numSeeds = seeds.GetLength(0);

        // Mean shift iterations for each seed
        var convergedCenters = new List<T[]>();

        for (int s = 0; s < numSeeds; s++)
        {
            var center = new T[d];
            for (int j = 0; j < d; j++)
            {
                center[j] = seeds[s, j];
            }

            // Iterate until convergence
            for (int iter = 0; iter < Options.MaxIterations; iter++)
            {
                var newCenter = ComputeMeanShift(x, center, n, d);

                // Check convergence
                T shift = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    T diff = NumOps.Subtract(newCenter[j], center[j]);
                    shift = NumOps.Add(shift, NumOps.Multiply(diff, diff));
                }
                shift = NumOps.Sqrt(shift);

                center = newCenter;

                if (NumOps.LessThan(shift, tolerance))
                {
                    break;
                }
            }

            convergedCenters.Add(center);
        }

        // Merge nearby centers
        T mergeThreshold = _options.ClusterMergeThreshold.HasValue
            ? NumOps.FromDouble(_options.ClusterMergeThreshold.Value)
            : _bandwidth;
        var finalCenters = MergeCenters(convergedCenters, mergeThreshold, d);

        NumClusters = finalCenters.Count;

        // Set cluster centers
        ClusterCenters = new Matrix<T>(NumClusters, d);
        for (int k = 0; k < NumClusters; k++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[k, j] = finalCenters[k][j];
            }
        }

        // Assign labels
        Labels = new Vector<T>(n);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(x, i);
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            for (int k = 0; k < NumClusters; k++)
            {
                var center = GetRow(ClusterCenters, k);
                T dist = metric.Compute(point, center);

                if (NumOps.LessThan(dist, minDist))
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
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Sample a subset of points for efficiency
        int sampleSize = Math.Min(n, 500);
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(sampleSize).ToList();

        // Compute pairwise distances (convert to double for sorting/quantile selection)
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

    private T[,] GetBinnedSeeds(Matrix<T> data, int n, int d)
    {
        // Create bins with width = bandwidth
        double binWidth = NumOps.ToDouble(_bandwidth);

        // Compute bin indices for each point
        var binCounts = new Dictionary<string, (T[] Sum, int Count)>();

        for (int i = 0; i < n; i++)
        {
            var binKey = new int[d];
            for (int j = 0; j < d; j++)
            {
                binKey[j] = (int)Math.Floor(NumOps.ToDouble(data[i, j]) / binWidth);
            }

            string key = string.Join(",", binKey);

            if (!binCounts.ContainsKey(key))
            {
                var sumArr = new T[d];
                for (int j = 0; j < d; j++) sumArr[j] = NumOps.Zero;
                binCounts[key] = (sumArr, 0);
            }

            var (sum, count) = binCounts[key];
            for (int j = 0; j < d; j++)
            {
                sum[j] = NumOps.Add(sum[j], data[i, j]);
            }
            binCounts[key] = (sum, count + 1);
        }

        // Filter bins with minimum frequency and compute centroids
        var seeds = new List<T[]>();
        foreach (var kvp in binCounts)
        {
            if (kvp.Value.Count >= _options.MinBinFrequency)
            {
                T countT = NumOps.FromDouble(kvp.Value.Count);
                var centroid = new T[d];
                for (int j = 0; j < d; j++)
                {
                    centroid[j] = NumOps.Divide(kvp.Value.Sum[j], countT);
                }
                seeds.Add(centroid);
            }
        }

        // Convert to 2D array
        var result = new T[seeds.Count, d];
        for (int i = 0; i < seeds.Count; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = seeds[i][j];
            }
        }

        return result;
    }

    private T[] ComputeMeanShift(Matrix<T> data, T[] center, int n, int d)
    {
        T bandwidthSq = NumOps.Multiply(_bandwidth, _bandwidth);
        T halfT = NumOps.FromDouble(0.5);
        var newCenter = new T[d];
        T totalWeight = NumOps.Zero;
        for (int j = 0; j < d; j++) newCenter[j] = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            // Compute squared distance
            T distSq = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(data[i, j], center[j]);
                distSq = NumOps.Add(distSq, NumOps.Multiply(diff, diff));
            }

            // Gaussian kernel weight (only within bandwidth)
            if (!NumOps.GreaterThan(distSq, bandwidthSq))
            {
                T weight = NumOps.Exp(NumOps.Negate(NumOps.Multiply(halfT, NumOps.Divide(distSq, bandwidthSq))));
                totalWeight = NumOps.Add(totalWeight, weight);

                for (int j = 0; j < d; j++)
                {
                    newCenter[j] = NumOps.Add(newCenter[j], NumOps.Multiply(weight, data[i, j]));
                }
            }
        }

        if (NumOps.GreaterThan(totalWeight, NumOps.Zero))
        {
            for (int j = 0; j < d; j++)
            {
                newCenter[j] = NumOps.Divide(newCenter[j], totalWeight);
            }
        }
        else
        {
            // No neighbors, keep current center
            newCenter = (T[])center.Clone();
        }

        return newCenter;
    }

    private List<T[]> MergeCenters(List<T[]> centers, T threshold, int d)
    {
        var merged = new List<T[]>();
        var used = new bool[centers.Count];

        for (int i = 0; i < centers.Count; i++)
        {
            if (used[i]) continue;

            // Find all centers close to this one
            var cluster = new List<T[]> { centers[i] };
            used[i] = true;

            for (int j = i + 1; j < centers.Count; j++)
            {
                if (used[j]) continue;

                T distSq = NumOps.Zero;
                for (int k = 0; k < d; k++)
                {
                    T diff = NumOps.Subtract(centers[i][k], centers[j][k]);
                    distSq = NumOps.Add(distSq, NumOps.Multiply(diff, diff));
                }

                if (NumOps.LessThan(NumOps.Sqrt(distSq), threshold))
                {
                    cluster.Add(centers[j]);
                    used[j] = true;
                }
            }

            // Compute mean of cluster
            var mean = new T[d];
            for (int k = 0; k < d; k++) mean[k] = NumOps.Zero;

            foreach (var c in cluster)
            {
                for (int k = 0; k < d; k++)
                {
                    mean[k] = NumOps.Add(mean[k], c[k]);
                }
            }
            T clusterCount = NumOps.FromDouble(cluster.Count);
            for (int k = 0; k < d; k++)
            {
                mean[k] = NumOps.Divide(mean[k], clusterCount);
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
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    T dist = metric.Compute(point, center);

                    if (NumOps.LessThan(dist, minDist))
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
        return Labels ?? new Vector<T>(0);
    }
}
