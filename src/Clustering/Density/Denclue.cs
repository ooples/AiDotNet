using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Density;

/// <summary>
/// DENCLUE (DENsity-based CLUstEring) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DENCLUE uses kernel density estimation to model the overall point density,
/// then clusters points based on the local maxima (attractors) of the density
/// function. Points are assigned to the attractor they converge to via
/// gradient ascent (hill climbing).
/// </para>
/// <para>
/// Algorithm:
/// 1. For each point, perform hill climbing on the density function
/// 2. Points converging to the same attractor form a cluster
/// 3. Merge attractors that are very close together
/// 4. Mark points with low-density attractors as noise
/// </para>
/// <para><b>For Beginners:</b> DENCLUE is like finding mountain peaks.
///
/// The density function creates a "landscape":
/// - High-density areas are "mountains"
/// - Low-density areas are "valleys"
///
/// Each point "climbs uphill" to find its nearest peak.
/// Points reaching the same peak belong to the same cluster.
///
/// Advantages over DBSCAN:
/// - Can find clusters of varying densities
/// - Produces cluster centers (attractors)
/// - Handles noise naturally
///
/// Best for: Smooth, well-defined clusters where density varies.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new Denclue&lt;double&gt;(new DenclueOptions&lt;double&gt;()))
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
[ResearchPaper("DENCLUE: A New Approach for Discovering Density-Based Clusters in Large Spatial Databases", "https://link.springer.com/chapter/10.1007/978-1-4615-5669-5_7", Year = 1998, Authors = "Alexander Hinneburg, Daniel A. Keim")]
public partial class Denclue<T> : ClusteringBase<T>
{
    private readonly DenclueOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[][]? _attractors;
    private T[]? _attractorDensities;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureMeans;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _featureStds;

    /// <summary>
    /// Initializes a new DENCLUE instance.
    /// </summary>
    /// <param name="options">The DENCLUE options.</param>
    public Denclue(DenclueOptions<T>? options = null)
        : base(options ?? new DenclueOptions<T>())
    {
        NumericGuard.RejectIntegerTypes<T>("DENCLUE");
        _options = (DenclueOptions<T>)Options;
    }

    /// <summary>
    /// Gets the density attractors (cluster centers).
    /// </summary>
    public T[][]? Attractors => _attractors;

    /// <summary>
    /// Gets the density values at each attractor.
    /// </summary>
    public T[]? AttractorDensities => _attractorDensities;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (Denclue<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Normalize features for scale-invariant density estimation
        x = ClusteringTensorOps.StandardizeColumns(x, out var featureMeans, out var featureStds);
        _featureMeans = featureMeans;
        _featureStds = featureStds;

        T minDensity = NumOps.FromDouble(_options.MinDensity);
        T mergeThreshold = NumOps.FromDouble(_options.AttractorMergeThreshold);

        // Find attractors for each point via hill climbing (all points climb together)
        var pointAttractors = HillClimb(x, out var pointDensities);
        var isNoise = ClusteringTensorOps.LessThanFlags(pointDensities, minDensity);
        var densities = pointDensities.ToArray();

        // Merge similar attractors and assign cluster labels (in point order)
        var attractorList = new List<Vector<T>>();
        var densityList = new List<T>();
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            if (isNoise[i])
            {
                // Noise point
                labels[i] = -1;
                continue;
            }

            // Find existing attractor this point matches
            var attractor = pointAttractors.GetRow(i);
            int matchedAttractor = FirstAttractorWithin(attractor, attractorList, mergeThreshold);

            if (matchedAttractor == -1)
            {
                // New attractor
                matchedAttractor = attractorList.Count;
                attractorList.Add(attractor);
                densityList.Add(densities[i]);
            }
            else if (IsHigher(densities[i], densityList[matchedAttractor]))
            {
                // Update attractor density if this one is higher
                attractorList[matchedAttractor] = attractor;
                densityList[matchedAttractor] = densities[i];
            }

            labels[i] = matchedAttractor;
        }

        // Store attractors
        _attractors = attractorList.Select(a => a.ToArray()).ToArray();
        _attractorDensities = densityList.ToArray();

        // Set cluster results
        NumClusters = attractorList.Count;
        ClusterCenters = NumClusters > 0
            ? Matrix<T>.FromRows(attractorList.Cast<IEnumerable<T>>().ToArray())
            : new Matrix<T>(0, d);
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        IsTrained = true;
    }

    /// <summary>Index of the first attractor closer than the threshold (Euclidean), or -1.</summary>
    private static int FirstAttractorWithin(Vector<T> point, List<Vector<T>> attractors, T threshold)
    {
        if (attractors.Count == 0)
        {
            return -1;
        }

        var matrix = Matrix<T>.FromRows(attractors.Cast<IEnumerable<T>>().ToArray());
        var within = ClusteringTensorOps.LessThanFlags(ClusteringTensorOps.DistancesToPoint(point, matrix), threshold);
        return Array.IndexOf(within, true);
    }

    /// <summary>Scalar comparison of two densities.</summary>
    private bool IsHigher(T candidate, T reference) => NumOps.GreaterThan(candidate, reference);

    // Predict override is below (after the attractor computation methods)

    private Matrix<T> HillClimb(Matrix<T> data, out Vector<T> densities)
    {
        // Every point climbs until its mean-shift step is shorter than the threshold; converged points stay put
        T convergenceThreshold = NumOps.FromDouble(_options.ConvergenceThreshold);
        var current = data;
        var next = KernelPass(current, data, out densities);
        var active = Engine.Fill(data.Rows, NumOps.One).AsTensor();

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            var step = Engine.TensorWhere(
                Engine.TensorBroadcastTo(Engine.Reshape(active, new[] { data.Rows, 1 }), new[] { data.Rows, data.Columns }),
                Tensor<T>.FromMatrix(next),
                Tensor<T>.FromMatrix(current)).ToMatrix();

            var following = KernelPass(step, data, out densities);
            var movement = ClusteringTensorOps.RowEuclidean(Tensor<T>.FromMatrix(current), Tensor<T>.FromMatrix(step));

            current = step;
            next = following;
            active = Engine.TensorMultiply(active, Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(movement, convergenceThreshold)));
            if (!HasActive(active))
            {
                break;
            }
        }

        return current;
    }

    /// <summary>True while some point is still climbing.</summary>
    private bool HasActive(Tensor<T> active) => NumOps.GreaterThan(Engine.TensorSum(active), NumOps.Zero);

    /// <summary>
    /// One Gaussian-kernel pass: the density at every point, (2 pi h^2)^(-d/2) / n * sum_j exp(-|p - x_j|^2 / 2h^2),
    /// and its mean-shift update sum_j w_j x_j / sum_j w_j (the point itself when the weights vanish).
    /// </summary>
    private Matrix<T> KernelPass(Matrix<T> points, Matrix<T> data, out Vector<T> densities)
    {
        int m = points.Rows;
        int n = data.Rows;
        int d = data.Columns;
        T twoH2 = NumOps.FromDouble(2.0 * _options.Bandwidth * _options.Bandwidth);
        T scale = NumOps.FromDouble(Math.Pow(2 * Math.PI * _options.Bandwidth * _options.Bandwidth, -d / 2.0) / n);
        var dataT = Tensor<T>.FromMatrix(data);

        var shifted = new Matrix<T>(m, d);
        var densityParts = new List<Vector<T>>();
        int block = Math.Max(1, (1 << 22) / Math.Max(1, n));
        for (int start = 0; start < m; start += block)
        {
            int rows = Math.Min(block, m - start);
            var chunk = points.GetSubMatrix(start, 0, rows, d);
            var weights = Engine.TensorExp(Engine.TensorDivideScalar(
                Engine.TensorNegate(Tensor<T>.FromMatrix(ClusteringTensorOps.SquaredDistances(chunk, data))), twoH2));
            var weightSums = Engine.ReduceSum(weights, new[] { 1 }, true);
            var numerator = Engine.TensorMatMul(weights, dataT);

            var positive = Engine.TensorBroadcastTo(Engine.TensorGreaterThan(weightSums, NumOps.Zero), new[] { rows, d });
            var safeSums = Engine.TensorWhere(
                Engine.TensorGreaterThan(weightSums, NumOps.Zero), weightSums, Engine.TensorOnesLike(weightSums));
            var update = Engine.TensorDivide(numerator, Engine.TensorBroadcastTo(safeSums, new[] { rows, d }));
            shifted.SetSubMatrix(start, 0, Engine.TensorWhere(positive, update, Tensor<T>.FromMatrix(chunk)).ToMatrix());
            densityParts.Add(Engine.TensorMultiplyScalar(Engine.Reshape(weightSums, new[] { rows }), scale).ToVector());
        }

        densities = densityParts.Count == 0 ? new Vector<T>(0) : Vector<T>.Concatenate(densityParts.ToArray());
        return shifted;
    }

    /// <summary>
    /// Gets the density estimate at a given point.
    /// </summary>
    /// <param name="point">The point to evaluate.</param>
    /// <returns>The estimated density.</returns>
    public T GetDensity(Vector<T> point)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || _attractors is null || _attractorDensities is null)
        {
            throw new InvalidOperationException(
                "Model is in an inconsistent state: trained but missing cluster data.");
        }

        int d = NumFeatures;
        if (point.Length != d)
        {
            throw new ArgumentException(
                $"Point has {point.Length} dimensions, but the model was fitted with {d} features.",
                nameof(point));
        }

        if (NumClusters == 0)
        {
            return NumOps.Zero;
        }

        // Use attractors as representative points for density:
        // (2 pi h^2)^(-d/2) sum_c density_c exp(-|p - a_c|^2 / 2h^2)
        T twoH2 = NumOps.FromDouble(2.0 * _options.Bandwidth * _options.Bandwidth);
        T normalization = NumOps.FromDouble(Math.Pow(2 * Math.PI * _options.Bandwidth * _options.Bandwidth, -d / 2.0));
        var attractors = Matrix<T>.FromRows(_attractors.Take(NumClusters).Cast<IEnumerable<T>>().ToArray());
        var kernel = Engine.TensorExp(Engine.TensorDivideScalar(
            Engine.TensorNegate(ClusteringTensorOps.SquaredDistancesToPoint(point, attractors).AsTensor()), twoH2));
        var weights = Vector<T>.FromArray(_attractorDensities.Take(NumClusters).ToArray()).AsTensor();
        return NumOps.Multiply(normalization, Engine.TensorSum(Engine.TensorMultiply(weights, kernel)));
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

        if (ClusterCenters is null || NumClusters <= 0 || x.Rows == 0)
        {
            // No attractors: every point is unassigned
            var unassigned = new Vector<T>(x.Rows);
            return x.Rows == 0
                ? unassigned
                : Engine.TensorFullLike(unassigned.AsTensor(), NumOps.FromDouble(-1)).ToVector();
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
        return Labels ?? throw new InvalidOperationException("Training failed to produce cluster labels.");
    }
}
