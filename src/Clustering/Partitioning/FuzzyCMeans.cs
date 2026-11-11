using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// Fuzzy C-Means (FCM) soft clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Fuzzy C-Means assigns each point a membership degree to each cluster,
/// rather than a hard assignment. The memberships sum to 1 for each point.
/// </para>
/// <para>
/// Algorithm:
/// 1. Initialize membership matrix randomly
/// 2. Compute cluster centers from weighted memberships
/// 3. Update memberships based on distances to centers
/// 4. Repeat until convergence
/// </para>
/// <para><b>For Beginners:</b> FCM creates "soft" or "fuzzy" clusters.
///
/// Instead of saying "Point X belongs to Cluster 1", FCM says:
/// "Point X is 60% Cluster 1, 30% Cluster 2, 10% Cluster 3"
///
/// This captures uncertainty and allows for overlapping clusters.
/// The fuzziness parameter controls how much overlap is allowed.
///
/// Use FCM when:
/// - Clusters have unclear boundaries
/// - Points naturally fit multiple categories
/// - You need confidence/probability information
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for fuzzy clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new FuzzyCMeans&lt;double&gt;(new FuzzyCMeansOptions&lt;double&gt;()));
///
/// var result = builder.Build(dataMatrix, labels);
/// var predictions = result.Predict(newData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("FCM: The Fuzzy c-Means Clustering Algorithm", "https://doi.org/10.1016/0098-3004(84)90020-7", Year = 1984, Authors = "James C. Bezdek, Robert Ehrlich, William Full")]
public partial class FuzzyCMeans<T> : ClusteringBase<T>
{
    private readonly FuzzyCMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _membershipMatrix = new Matrix<T>(0, 0);

    /// <summary>
    /// Initializes a new FuzzyCMeans instance.
    /// </summary>
    /// <param name="options">The FCM options.</param>
    public FuzzyCMeans(FuzzyCMeansOptions<T>? options = null)
        : base(options ??= new FuzzyCMeansOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the membership matrix (n_samples x n_clusters).
    /// </summary>
    /// <remarks>
    /// Each row sums to 1. membershipMatrix[i, k] is the degree to which
    /// point i belongs to cluster k.
    /// </remarks>
    public Matrix<T> MembershipMatrix => _membershipMatrix.Clone();

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (FuzzyCMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        double m = _options.Fuzziness;
        NumFeatures = d;
        NumClusters = k;

        if (m <= 1)
        {
            throw new ArgumentException("Fuzziness must be greater than 1.");
        }

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Initialize membership matrix randomly
        _membershipMatrix = new Matrix<T>(n, k);
        InitializeMembership(n, k, rand);

        // Cluster centers
        var centers = new Matrix<T>(k, d);

        // Iterative optimization
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Step 1: Update centers
            centers = UpdateCenters(x, m);

            // Step 2: Update memberships
            T maxChange = UpdateMemberships(x, centers, m, metric);

            // Check convergence
            if (IsBelowTolerance(maxChange))
            {
                break;
            }
        }

        // Set cluster centers
        ClusterCenters = centers;

        // Assign hard labels based on maximum membership (first one on ties)
        Labels = ClusteringTensorOps.FromIntArray<T>(
            Engine.TensorArgMax(Tensor<T>.FromMatrix(_membershipMatrix), 1).ToArray());

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <summary>True when the largest membership change is below the tolerance.</summary>
    private bool IsBelowTolerance(T maxChange) => NumOps.LessThan(maxChange, NumOps.FromDouble(Options.Tolerance));

    private void InitializeMembership(int n, int k, Random rand)
    {
        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix[i, c] = NumOps.FromDouble(rand.NextDouble());
                sum = NumOps.Add(sum, _membershipMatrix[i, c]);
            }

            // Normalize to sum to 1
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix[i, c] = NumOps.Divide(_membershipMatrix[i, c], sum);
            }
        }
    }

    private Matrix<T> UpdateCenters(Matrix<T> x, double m)
    {
        // c_k = sum_i u_ik^m x_i / sum_i u_ik^m (zero when the weights vanish)
        var weights = Engine.TensorPower(Tensor<T>.FromMatrix(_membershipMatrix), NumOps.FromDouble(m));
        var numerator = Engine.TensorMatMul(Engine.TensorTranspose(weights), Tensor<T>.FromMatrix(x)).ToMatrix();
        var denominator = Engine.ReduceSum(weights, new[] { 0 }, false).ToVector();
        var positive = Engine.TensorGreaterThan(denominator.AsTensor(), NumOps.Zero);
        var safe = Engine.TensorWhere(positive, denominator.AsTensor(), Engine.TensorOnesLike(denominator.AsTensor()));
        var divisor = Engine.TensorBroadcastTo(
            Engine.Reshape(safe, new[] { numerator.Rows, 1 }), new[] { numerator.Rows, numerator.Columns });
        var centers = Engine.TensorDivide(Tensor<T>.FromMatrix(numerator), divisor).ToMatrix();
        return ClusteringTensorOps.RowsWherePositive(denominator, centers, new Matrix<T>(centers.Rows, centers.Columns));
    }

    private T UpdateMemberships(Matrix<T> x, Matrix<T> centers, double m, IDistanceMetric<T> metric)
    {
        var oldMemberships = Tensor<T>.FromMatrix(_membershipMatrix);
        var newMemberships = Tensor<T>.FromMatrix(MembershipsFromDistances(metric.ComputePairwise(x, centers), m));
        _membershipMatrix = newMemberships.ToMatrix();

        // Largest absolute membership change (zero when there are no memberships)
        if (newMemberships.Length == 0)
        {
            return NumOps.Zero;
        }

        return Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorSubtract(newMemberships, oldMemberships)));
    }

    /// <summary>
    /// u_ic = 1 / sum_j (d_ic / d_ij)^(2 / (m - 1)); a point closer than 1e-10 to a center belongs fully to the
    /// last such center.
    /// </summary>
    private Matrix<T> MembershipsFromDistances(Matrix<T> distances, double m)
    {
        int n = distances.Rows;
        int k = distances.Columns;
        if (n == 0 || k == 0)
        {
            return new Matrix<T>(n, k);
        }

        var dist = Tensor<T>.FromMatrix(distances);
        var nearZero = Engine.TensorLessThan(dist, NumOps.FromDouble(1e-10));

        // Rows with a (near) zero distance: one-hot at the last such center
        var position = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(1, k + 1, 1), new[] { 1, k }), new[] { n, k });
        var lastZero = Engine.TensorArgMax(Engine.TensorMultiply(nearZero, position), 1);
        var oneHot = Engine.TensorOneHot<T>(lastZero, k);
        var hasZero = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.ReduceMax(nearZero, new[] { 1 }, false), new[] { n, 1 }), new[] { n, k });

        // Standard update on the other rows (near-zero entries only occur in rows that take the one-hot)
        var safe = Engine.TensorWhere(nearZero, Engine.TensorOnesLike(dist), dist);
        var numerators = Engine.TensorBroadcastTo(Engine.Reshape(safe, new[] { n, k, 1 }), new[] { n, k, k });
        var denominators = Engine.TensorBroadcastTo(Engine.Reshape(safe, new[] { n, 1, k }), new[] { n, k, k });
        var ratios = Engine.TensorPower(Engine.TensorDivide(numerators, denominators), NumOps.FromDouble(2.0 / (m - 1)));
        var standard = Engine.TensorReciprocal(Engine.ReduceSum(ratios, new[] { 2 }, false));

        return Engine.TensorWhere(hasZero, oneHot, standard).ToMatrix();
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

    /// <summary>
    /// Predicts soft cluster memberships for new data.
    /// </summary>
    /// <param name="x">The data matrix.</param>
    /// <returns>Membership matrix (n_samples x n_clusters).</returns>
    public Matrix<T> PredictMembership(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int k = NumClusters;
        double m = _options.Fuzziness;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        if (ClusterCenters is null)
        {
            // No centers: every ratio is 0 / 0
            var undefined = new Matrix<T>(n, k);
            return n * k == 0
                ? undefined
                : Engine.TensorFullLike(Tensor<T>.FromMatrix(undefined), NumOps.FromDouble(double.NaN)).ToMatrix();
        }

        var centers = ClusterCenters.Rows == k ? ClusterCenters : ClusterCenters.GetSubMatrix(0, 0, k, ClusterCenters.Columns);
        return MembershipsFromDistances(metric.ComputePairwise(x, centers), m);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        if (Labels is null)
        {
            throw new InvalidOperationException("Train did not produce labels. This indicates a bug in the training implementation.");
        }
        return Labels;
    }
}
