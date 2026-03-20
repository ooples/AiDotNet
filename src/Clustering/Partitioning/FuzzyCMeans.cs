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
[ModelPaper("FCM: The Fuzzy c-Means Clustering Algorithm", "https://doi.org/10.1016/0098-3004(84)90020-7", Year = 1984, Authors = "James C. Bezdek, Robert Ehrlich, William Full")]
public class FuzzyCMeans<T> : ClusteringBase<T>
{
    private readonly FuzzyCMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[,]? _membershipMatrix;

    /// <summary>
    /// Initializes a new FuzzyCMeans instance.
    /// </summary>
    /// <param name="options">The FCM options.</param>
    public FuzzyCMeans(FuzzyCMeansOptions<T>? options = null)
        : base(options ?? new FuzzyCMeansOptions<T>())
    {
        _options = options ?? new FuzzyCMeansOptions<T>();
    }

    /// <summary>
    /// Gets the membership matrix (n_samples x n_clusters).
    /// </summary>
    /// <remarks>
    /// Each row sums to 1. membershipMatrix[i, k] is the degree to which
    /// point i belongs to cluster k.
    /// </remarks>
    public T[,]? MembershipMatrix => _membershipMatrix;

    /// <inheritdoc />

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new FuzzyCMeans<T>(new FuzzyCMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            Fuzziness = _options.Fuzziness,
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
        _membershipMatrix = new T[n, k];
        InitializeMembership(n, k, rand);

        // Cluster centers
        var centers = new T[k, d];

        // Iterative optimization
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Step 1: Update centers
            UpdateCenters(x, centers, n, d, k, m);

            // Step 2: Update memberships
            T maxChange = UpdateMemberships(x, centers, n, d, k, m, metric);

            // Check convergence
            if (NumOps.LessThan(maxChange, NumOps.FromDouble(Options.Tolerance)))
            {
                break;
            }
        }

        // Set cluster centers
        ClusterCenters = new Matrix<T>(k, d);
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = centers[c, j];
            }
        }

        // Assign hard labels based on maximum membership
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            T maxMembership = _membershipMatrix[i, 0];

            for (int c = 1; c < k; c++)
            {
                if (NumOps.GreaterThan(_membershipMatrix[i, c], maxMembership))
                {
                    maxMembership = _membershipMatrix[i, c];
                    bestCluster = c;
                }
            }

            Labels[i] = NumOps.FromDouble(bestCluster);
        }

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private void InitializeMembership(int n, int k, Random rand)
    {
        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix![i, c] = NumOps.FromDouble(rand.NextDouble());
                sum = NumOps.Add(sum, _membershipMatrix[i, c]);
            }

            // Normalize to sum to 1
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix![i, c] = NumOps.Divide(_membershipMatrix[i, c], sum);
            }
        }
    }

    private void UpdateCenters(Matrix<T> x, T[,] centers, int n, int d, int k, double m)
    {
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                T numerator = NumOps.Zero;
                T denominator = NumOps.Zero;

                for (int i = 0; i < n; i++)
                {
                    T membership = NumOps.FromDouble(Math.Pow(NumOps.ToDouble(_membershipMatrix![i, c]), m));
                    numerator = NumOps.Add(numerator, NumOps.Multiply(membership, x[i, j]));
                    denominator = NumOps.Add(denominator, membership);
                }

                centers[c, j] = NumOps.GreaterThan(denominator, NumOps.Zero)
                    ? NumOps.Divide(numerator, denominator)
                    : NumOps.Zero;
            }
        }
    }

    private T UpdateMemberships(Matrix<T> x, T[,] centers, int n, int d, int k, double m, IDistanceMetric<T> metric)
    {
        T maxChange = NumOps.Zero;
        double exponent = 2.0 / (m - 1);
        T epsilon = NumOps.FromDouble(1e-10);

        // Cache point arrays for allocation-free distance computation
        var pointArr = new T[d];
        var centerArr = new T[d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
                pointArr[j] = x[i, j];

            // Compute distances to all centers
            var distances = new T[k];
            bool hasZeroDistance = false;
            int zeroCluster = -1;

            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < d; j++)
                    centerArr[j] = centers[c, j];

                distances[c] = metric.ComputeInline(pointArr, centerArr, d);

                if (NumOps.LessThan(distances[c], epsilon))
                {
                    hasZeroDistance = true;
                    zeroCluster = c;
                }
            }

            // Update memberships
            for (int c = 0; c < k; c++)
            {
                T oldMembership = _membershipMatrix![i, c];
                T newMembership;

                if (hasZeroDistance)
                {
                    // Point is exactly at a center
                    newMembership = (c == zeroCluster) ? NumOps.One : NumOps.Zero;
                }
                else
                {
                    // Standard update formula — Math.Pow at double boundary
                    double distC = NumOps.ToDouble(distances[c]);
                    double sum = 0;
                    for (int j = 0; j < k; j++)
                    {
                        sum += Math.Pow(distC / NumOps.ToDouble(distances[j]), exponent);
                    }
                    newMembership = NumOps.FromDouble(1.0 / sum);
                }

                _membershipMatrix[i, c] = newMembership;
                T change = NumOps.Abs(NumOps.Subtract(newMembership, oldMembership));
                if (NumOps.GreaterThan(change, maxChange))
                    maxChange = change;
            }
        }

        return maxChange;
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
                for (int c = 0; c < NumClusters; c++)
                {
                    var center = GetRow(ClusterCenters, c);
                    T dist = metric.Compute(point, center);

                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        nearestCluster = c;
                    }
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    /// <summary>
    /// Predicts soft cluster memberships for new data.
    /// </summary>
    /// <param name="x">The data matrix.</param>
    /// <returns>Membership matrix (n_samples x n_clusters).</returns>
    public T[,] PredictMembership(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int k = NumClusters;
        double m = _options.Fuzziness;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        T epsilon = NumOps.FromDouble(1e-10);

        var memberships = new T[n, k];
        double exponent = 2.0 / (m - 1);

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(x, i);

            var distances = new T[k];
            bool hasZeroDistance = false;
            int zeroCluster = -1;

            if (ClusterCenters is not null)
            {
                for (int c = 0; c < k; c++)
                {
                    var center = GetRow(ClusterCenters, c);
                    distances[c] = metric.Compute(point, center);

                    if (NumOps.LessThan(distances[c], epsilon))
                    {
                        hasZeroDistance = true;
                        zeroCluster = c;
                    }
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (hasZeroDistance)
                {
                    memberships[i, c] = (c == zeroCluster) ? NumOps.One : NumOps.Zero;
                }
                else
                {
                    double distC = NumOps.ToDouble(distances[c]);
                    double sum = 0;
                    for (int j = 0; j < k; j++)
                    {
                        sum += Math.Pow(distC / NumOps.ToDouble(distances[j]), exponent);
                    }
                    memberships[i, c] = NumOps.FromDouble(1.0 / sum);
                }
            }
        }

        return memberships;
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
