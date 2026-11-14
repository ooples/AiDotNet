using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.AutoK;

/// <summary>
/// X-Means clustering with automatic K determination using BIC.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// X-Means iteratively applies K-Means and decides whether to split clusters
/// based on the Bayesian Information Criterion (BIC). It automatically
/// determines the optimal number of clusters.
/// </para>
/// <para>
/// Algorithm:
/// 1. Run K-Means with initial K
/// 2. For each cluster, try splitting into two
/// 3. Compare BIC of parent vs children
/// 4. Accept split if BIC improves
/// 5. Repeat until no splits improve or max K reached
/// </para>
/// <para><b>For Beginners:</b> X-Means is K-Means that chooses K for you.
///
/// Instead of guessing the right number of clusters:
/// - Start small (e.g., K=2)
/// - Try splitting each cluster
/// - Keep splits that make statistical sense
/// - Stop when splitting doesn't help
///
/// BIC tells us when a split is worthwhile:
/// - Lower BIC = better model
/// - Splitting adds complexity (penalized)
/// - Only split if the fit improvement outweighs the penalty
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new XMeans&lt;double&gt;(new XMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("X-means: Extending K-means with Efficient Estimation of the Number of Clusters", "https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf", Year = 2000, Authors = "Dan Pelleg, Andrew Moore")]
public partial class XMeans<T> : ClusteringBase<T>
{
    private readonly XMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private T _bic;


    /// <summary>
    /// Initializes a new XMeans instance.
    /// </summary>
    /// <param name="options">The XMeans options.</param>
    public XMeans(XMeansOptions<T>? options = null)
        : base(options ??= new XMeansOptions<T>())
    {
        _bic = NumOps.Zero;
        _options = options;
    }

    /// <summary>
    /// Gets the final BIC value.
    /// </summary>
    public T BIC => _bic;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (XMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        NumFeatures = d;

        // Start with minimum clusters
        int currentK = _options.MinClusters;

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = currentK,
            MaxIterations = Options.MaxIterations,
            Seed = Options.Seed
        });

        kmeans.Train(x);
        var currentLabels = kmeans.Labels ?? new Vector<T>(n);
        var currentCenters = kmeans.ClusterCenters ?? new Matrix<T>(currentK, d);
        currentK = currentCenters.Rows; // Sync with actual K after potential degenerate merge

        // Iteratively try to split clusters
        bool improved = true;
        while (improved && currentK < _options.MaxClusters)
        {
            improved = false;
            var newCenters = new List<Vector<T>>();
            var clusterMapping = new Dictionary<int, List<int>>();
            var labelIds = ClusteringTensorOps.ToIntArray(currentLabels);

            // For each cluster, decide whether to split
            for (int c = 0; c < currentK; c++)
            {
                // Get points in this cluster
                var clusterPoints = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if (labelIds[i] == c)
                    {
                        clusterPoints.Add(i);
                    }
                }

                if (clusterPoints.Count < 4)
                {
                    // Too few points to split
                    clusterMapping[newCenters.Count] = clusterPoints;
                    newCenters.Add(currentCenters.GetRow(c));
                    continue;
                }

                // Create sub-matrix for this cluster
                var subMatrix = ClusteringTensorOps.SelectRows(x, clusterPoints.ToArray());

                // Compute BIC for parent cluster
                T parentBIC = ComputeClusterBIC(subMatrix, d);

                // Try splitting with K=2
                var subKMeans = new KMeans<T>(new KMeansOptions<T>
                {
                    NumClusters = 2,
                    MaxIterations = Options.MaxIterations,
                    Seed = Options.Seed
                });

                subKMeans.Train(subMatrix);

                var subLabels = subKMeans.Labels ?? new Vector<T>(clusterPoints.Count);
                var subCenters = subKMeans.ClusterCenters ?? new Matrix<T>(2, d);

                // Compute BIC for children
                T childBIC = ComputeSplitBIC(subMatrix, subLabels, subCenters, d);

                if (IsLower(childBIC, parentBIC) && newCenters.Count + 1 < _options.MaxClusters)
                {
                    // Accept split
                    var subIds = ClusteringTensorOps.ToIntArray(subLabels);
                    for (int sc = 0; sc < 2; sc++)
                    {
                        var subClusterPoints = new List<int>();
                        for (int i = 0; i < clusterPoints.Count; i++)
                        {
                            if (subIds[i] == sc)
                            {
                                subClusterPoints.Add(clusterPoints[i]);
                            }
                        }

                        clusterMapping[newCenters.Count] = subClusterPoints;
                        newCenters.Add(subCenters.GetRow(sc));
                    }
                    improved = true;
                }
                else
                {
                    // Keep parent cluster
                    clusterMapping[newCenters.Count] = clusterPoints;
                    newCenters.Add(currentCenters.GetRow(c));
                }
            }

            // Update current state
            currentK = newCenters.Count;
            currentCenters = currentK > 0
                ? Matrix<T>.FromRows(newCenters.Cast<IEnumerable<T>>().ToArray())
                : new Matrix<T>(0, d);

            var newLabels = new int[n];
            for (int c = 0; c < currentK; c++)
            {
                foreach (int i in clusterMapping[c])
                {
                    newLabels[i] = c;
                }
            }

            currentLabels = ClusteringTensorOps.FromIntArray<T>(newLabels);
        }

        // Set final results
        NumClusters = currentK;
        ClusterCenters = currentCenters;
        Labels = currentLabels;
        _bic = ComputeTotalBIC(x, currentLabels, currentCenters, n, d, currentK);

        MergeDegenerateClusters(x);

        // Recompute BIC after merge since cluster count and assignments may have changed
        if (NumClusters < currentK && ClusterCenters is not null && Labels is not null)
        {
            _bic = ComputeTotalBIC(x, Labels, ClusterCenters, n, d, NumClusters);
        }

        IsTrained = true;
    }

    /// <summary>Scalar comparison of two criterion values.</summary>
    private bool IsLower(T candidate, T reference) => NumOps.LessThan(candidate, reference);

    private T ComputeClusterBIC(Matrix<T> clusterData, int d)
    {
        int n = clusterData.Rows;
        if (n == 0) return NumOps.Zero;

        // Compute variance
        T variance = ComputeVariance(clusterData, n, d);

        // Log-likelihood for single Gaussian
        T eps = NumOps.FromDouble(1e-10);
        T logVariance = NumOps.Log(NumOps.Add(variance, eps));
        T log2Pi = NumOps.FromDouble(Math.Log(2 * Math.PI));
        T nT = NumOps.FromDouble(n);
        T dT = NumOps.FromDouble(d);
        T logL = NumOps.Negate(NumOps.Multiply(NumOps.Divide(nT, NumOps.FromDouble(2)),
            NumOps.Add(NumOps.Add(NumOps.Multiply(dT, log2Pi), NumOps.Multiply(dT, logVariance)), dT)));

        // Number of parameters: d (mean) + 1 (variance)
        T numParams = NumOps.FromDouble(d + 1);

        return _options.Criterion == InformationCriterion.BIC
            ? NumOps.Add(NumOps.Multiply(NumOps.FromDouble(-2), logL), NumOps.Multiply(numParams, NumOps.Log(nT)))
            : NumOps.Add(NumOps.Multiply(NumOps.FromDouble(-2), logL), NumOps.Multiply(NumOps.FromDouble(2), numParams));
    }

    private T ComputeSplitBIC(Matrix<T> data, Vector<T> labels, Matrix<T> centers, int d)
    {
        int n = data.Rows;
        if (n == 0) return NumOps.Zero;

        // Each non-empty child contributes d (mean) + 1 (variance) parameters
        T totalLogL = SumClusterLogLikelihood(data, labels, 2, d, out int nonEmpty);
        return Criterion(totalLogL, nonEmpty * (d + 1), n);
    }

    /// <summary>
    /// Sum over clusters 0..k-1 of the spherical Gaussian log-likelihood
    /// -n_c / 2 (d log 2 pi + d log(var_c + eps) + d), where var_c is the per-dimension variance (1e-10 for n_c &lt;= 1).
    /// </summary>
    private T SumClusterLogLikelihood(Matrix<T> data, Vector<T> labels, int k, int d, out int nonEmpty)
    {
        int n = data.Rows;
        var means = ClusteringTensorOps.ClusterMeans(data, labels, k, out var counts);
        var squared = ClusteringTensorOps.AssignedSquaredDistances(data, labels, means);
        var members = ClusteringTensorOps.OneHotMask(labels, k);
        var sumSquares = Engine.TensorMatMul(Engine.TensorTranspose(members), Tensor<T>.FromVector(squared, new[] { n, 1 }));
        var countTensor = counts.AsTensor();

        var variance = Engine.TensorDivideScalar(
            Engine.TensorDivide(Engine.Reshape(sumSquares, new[] { k }), Engine.TensorMax(countTensor, NumOps.One)),
            NumOps.FromDouble(d));
        var singular = Engine.TensorLessThan(countTensor, NumOps.FromDouble(1.5));
        variance = Engine.TensorWhere(singular, Engine.TensorFullLike(variance, NumOps.FromDouble(1e-10)), variance);

        T dT = NumOps.FromDouble(d);
        var logVariance = Engine.TensorLog(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-10)));
        var inner = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(logVariance, dT),
            NumOps.Add(NumOps.Multiply(dT, NumOps.FromDouble(Math.Log(2 * Math.PI))), dT));
        T totalLogL = NumOps.Multiply(NumOps.FromDouble(-0.5), Engine.TensorSum(Engine.TensorMultiply(countTensor, inner)));

        nonEmpty = ClusteringTensorOps.ToIntArray(
            Engine.TensorGreaterThan(countTensor, NumOps.Zero).ToVector()).Sum();
        return totalLogL;
    }

    /// <summary>BIC (-2 logL + p ln n) or AIC (-2 logL + 2 p) for the configured criterion.</summary>
    private T Criterion(T logL, int numParams, int n)
    {
        T paramsT = NumOps.FromDouble(numParams);
        T fit = NumOps.Multiply(NumOps.FromDouble(-2), logL);
        return _options.Criterion == InformationCriterion.BIC
            ? NumOps.Add(fit, NumOps.Multiply(paramsT, NumOps.Log(NumOps.FromDouble(n))))
            : NumOps.Add(fit, NumOps.Multiply(NumOps.FromDouble(2), paramsT));
    }

    private T ComputeTotalBIC(Matrix<T> data, Vector<T> labels, Matrix<T> centers, int n, int d, int k)
    {
        T totalLogL = k > 0 && n > 0 ? SumClusterLogLikelihood(data, labels, k, d, out _) : NumOps.Zero;
        return Criterion(totalLogL, k * (d + 1), n);
    }

    private T ComputeVariance(Matrix<T> data, int n, int d)
    {
        if (n <= 1) return NumOps.FromDouble(1e-10);

        // Mean squared deviation from the column means over all n * d entries
        var centered = ClusteringTensorOps.SubtractRow(data, ClusteringTensorOps.ColumnMeans(data));
        return NumOps.Divide(Engine.TensorSum(Engine.TensorSquare(centered)), NumOps.FromDouble((double)n * d));
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Return stored labels for in-sample prediction
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        int k = ClusterCenters is null ? 0 : Math.Min(NumClusters, ClusterCenters.Rows);
        if (ClusterCenters is null || k <= 0)
        {
            return new Vector<T>(x.Rows);
        }

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var centers = ClusterCenters.Rows == k ? ClusterCenters : ClusterCenters.GetSubMatrix(0, 0, k, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(metric, x, centers));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
