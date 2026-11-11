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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MeanShift&lt;double&gt;(new MeanShiftOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Mean Shift: A Robust Approach toward Feature Space Analysis", "https://doi.org/10.1109/34.1000236", Year = 2002, Authors = "Dorin Comaniciu, Peter Meer")]
public partial class MeanShift<T> : ClusteringBase<T>
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
        : base(options ??= new MeanShiftOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the bandwidth used for clustering.
    /// </summary>
    public T Bandwidth => _bandwidth;

    /// <inheritdoc />

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
        _bandwidth = _options.Bandwidth.HasValue ? NumOps.FromDouble(_options.Bandwidth.Value) : EstimateBandwidth(x);
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        // Get seeds (either binned or all points)
        var seeds = _options.BinSeeding ? GetBinnedSeeds(x, n, d) : x.Clone();

        // Mean shift iterations for every seed at once; converged seeds stop moving
        var centers = seeds;
        var active = Engine.Fill(seeds.Rows, NumOps.One).AsTensor();
        for (int iter = 0; iter < Options.MaxIterations && seeds.Rows > 0; iter++)
        {
            var shifted = ComputeMeanShift(x, centers);
            var shift = ClusteringTensorOps.RowEuclidean(Tensor<T>.FromMatrix(shifted), Tensor<T>.FromMatrix(centers));

            var moving = Engine.TensorBroadcastTo(Engine.Reshape(active, new[] { seeds.Rows, 1 }), new[] { seeds.Rows, d });
            centers = Engine.TensorWhere(moving, Tensor<T>.FromMatrix(shifted), Tensor<T>.FromMatrix(centers)).ToMatrix();

            active = Engine.TensorMultiply(active, Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(shift, tolerance)));
            if (!HasActive(active))
            {
                break;
            }
        }

        // Merge nearby centers
        T mergeThreshold = _options.ClusterMergeThreshold.HasValue
            ? NumOps.FromDouble(_options.ClusterMergeThreshold.Value)
            : _bandwidth;
        ClusterCenters = MergeCenters(centers, mergeThreshold, d);
        NumClusters = ClusterCenters.Rows;

        // Assign labels
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        Labels = NumClusters > 0
            ? ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, ClusterCenters))
            : new Vector<T>(n);

        IsTrained = true;
    }

    /// <summary>True while some seed is still moving.</summary>
    private bool HasActive(Tensor<T> active) => NumOps.GreaterThan(Engine.TensorSum(active), NumOps.Zero);

    private T EstimateBandwidth(Matrix<T> x)
    {
        int n = x.Rows;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Sample a subset of points for efficiency
        int sampleSize = Math.Min(n, 500);
        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(sampleSize).ToArray();

        // Distances between every pair of sampled points (upper triangle of the pairwise matrix)
        var pairwise = Tensor<T>.FromMatrix(metric.ComputePairwise(ClusteringTensorOps.SelectRows(x, indices)));
        var ones = Engine.TensorOnesLike(pairwise);
        var upper = Engine.TensorEq(Engine.TensorTriu(ones, 1), ones);
        var distances = Engine.TensorMsort(Engine.TensorMaskedSelect(pairwise, upper)).ToVector();

        // Return the quantile distance
        int idx = (int)(distances.Length * _options.BandwidthQuantile);
        return distances[Math.Max(0, Math.Min(idx, distances.Length - 1))];
    }

    private Matrix<T> GetBinnedSeeds(Matrix<T> data, int n, int d)
    {
        if (n == 0)
        {
            return new Matrix<T>(0, d);
        }

        // Bin index of every coordinate (bin width = bandwidth), encoded by its rank among the distinct bin indices
        var bins = Engine.TensorFloor(Engine.TensorDivideScalar(Tensor<T>.FromMatrix(data), _bandwidth));
        var (_, inverse, _) = Engine.TensorUniqueWithInfo(Engine.Reshape(bins, new[] { n * d }), true, true, false);
        if (inverse is null)
        {
            throw new InvalidOperationException("Unique bin computation did not return the inverse mapping.");
        }

        var binRanks = inverse.ToArray();

        // Group points by bin in order of first appearance
        var binOf = new Dictionary<string, int>();
        var binSizes = new List<int>();
        var pointBins = new int[n];
        for (int i = 0; i < n; i++)
        {
            string key = string.Join(",", new ArraySegment<int>(binRanks, i * d, d));
            if (!binOf.TryGetValue(key, out int bin))
            {
                bin = binSizes.Count;
                binOf[key] = bin;
                binSizes.Add(0);
            }

            binSizes[bin]++;
            pointBins[i] = bin;
        }

        // Keep bins with minimum frequency; their centroids are the seeds
        var seedIndex = new int[binSizes.Count];
        int numSeeds = 0;
        for (int b = 0; b < binSizes.Count; b++)
        {
            seedIndex[b] = binSizes[b] >= _options.MinBinFrequency ? numSeeds++ : -1;
        }

        if (numSeeds == 0)
        {
            return new Matrix<T>(0, d);
        }

        var seedLabels = new int[n];
        for (int i = 0; i < n; i++)
        {
            seedLabels[i] = seedIndex[pointBins[i]];
        }

        return ClusteringTensorOps.ClusterMeans(data, ClusteringTensorOps.FromIntArray<T>(seedLabels), numSeeds, out _);
    }

    private Matrix<T> ComputeMeanShift(Matrix<T> data, Matrix<T> centers)
    {
        // Gaussian kernel weight exp(-0.5 |x - c|^2 / h^2), only within the bandwidth;
        // a center without neighbors stays where it is
        int m = centers.Rows;
        int d = data.Columns;
        T bandwidthSq = NumOps.Multiply(_bandwidth, _bandwidth);
        var dataT = Tensor<T>.FromMatrix(data);

        var shifted = new Matrix<T>(m, d);
        int block = Math.Max(1, (1 << 22) / Math.Max(1, data.Rows));
        for (int start = 0; start < m; start += block)
        {
            int rows = Math.Min(block, m - start);
            var chunk = centers.GetSubMatrix(start, 0, rows, d);
            var distSq = Tensor<T>.FromMatrix(ClusteringTensorOps.SquaredDistances(chunk, data));
            var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(
                Engine.TensorDivideScalar(distSq, bandwidthSq), NumOps.FromDouble(-0.5)));
            var inside = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorGreaterThan(distSq, bandwidthSq));
            var weights = Engine.TensorMultiply(kernel, inside);

            var totals = Engine.ReduceSum(weights, new[] { 1 }, true);
            var hasWeight = Engine.TensorGreaterThan(totals, NumOps.Zero);
            var safeTotals = Engine.TensorWhere(hasWeight, totals, Engine.TensorOnesLike(totals));
            var mean = Engine.TensorDivide(
                Engine.TensorMatMul(weights, dataT), Engine.TensorBroadcastTo(safeTotals, new[] { rows, d }));
            var keep = Engine.TensorBroadcastTo(hasWeight, new[] { rows, d });
            shifted.SetSubMatrix(start, 0, Engine.TensorWhere(keep, mean, Tensor<T>.FromMatrix(chunk)).ToMatrix());
        }

        return shifted;
    }

    private Matrix<T> MergeCenters(Matrix<T> centers, T threshold, int d)
    {
        // Greedy: each unused center absorbs every later unused center closer than the threshold
        int count = centers.Rows;
        var groupOf = new int[count];
        for (int i = 0; i < count; i++)
        {
            groupOf[i] = -1;
        }

        int numGroups = 0;
        for (int i = 0; i < count; i++)
        {
            if (groupOf[i] >= 0) continue;

            int group = numGroups++;
            groupOf[i] = group;

            var close = ClusteringTensorOps.LessThanFlags(ClusteringTensorOps.DistancesToPoint(centers.GetRow(i), centers), threshold);
            for (int j = i + 1; j < count; j++)
            {
                if (groupOf[j] < 0 && close[j])
                {
                    groupOf[j] = group;
                }
            }
        }

        // Compute mean of each group
        return numGroups == 0
            ? new Matrix<T>(0, d)
            : ClusteringTensorOps.ClusterMeans(centers, ClusteringTensorOps.FromIntArray<T>(groupOf), numGroups, out _);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
