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
public class FuzzyCMeans<T> : ClusteringBase<T>
{
    private readonly FuzzyCMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private double[,]? _membershipMatrix;

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
    public double[,]? MembershipMatrix => _membershipMatrix;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

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

        var rand = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSecureRandom();
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Initialize membership matrix randomly
        _membershipMatrix = new double[n, k];
        InitializeMembership(n, k, rand);

        // Cluster centers
        var centers = new double[k, d];

        // Iterative optimization
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Step 1: Update centers
            UpdateCenters(x, centers, n, d, k, m);

            // Step 2: Update memberships
            double maxChange = UpdateMemberships(x, centers, n, d, k, m, metric);

            // Check convergence
            if (maxChange < Options.Tolerance)
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
                ClusterCenters[c, j] = NumOps.FromDouble(centers[c, j]);
            }
        }

        // Assign hard labels based on maximum membership
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            int bestCluster = 0;
            double maxMembership = _membershipMatrix[i, 0];

            for (int c = 1; c < k; c++)
            {
                if (_membershipMatrix[i, c] > maxMembership)
                {
                    maxMembership = _membershipMatrix[i, c];
                    bestCluster = c;
                }
            }

            Labels[i] = NumOps.FromDouble(bestCluster);
        }

        IsTrained = true;
    }

    private void InitializeMembership(int n, int k, Random rand)
    {
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix![i, c] = rand.NextDouble();
                sum += _membershipMatrix[i, c];
            }

            // Normalize to sum to 1
            for (int c = 0; c < k; c++)
            {
                _membershipMatrix![i, c] /= sum;
            }
        }
    }

    private void UpdateCenters(Matrix<T> x, double[,] centers, int n, int d, int k, double m)
    {
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                double numerator = 0;
                double denominator = 0;

                for (int i = 0; i < n; i++)
                {
                    double membership = Math.Pow(_membershipMatrix![i, c], m);
                    numerator += membership * NumOps.ToDouble(x[i, j]);
                    denominator += membership;
                }

                centers[c, j] = denominator > 0 ? numerator / denominator : 0;
            }
        }
    }

    private double UpdateMemberships(Matrix<T> x, double[,] centers, int n, int d, int k, double m, IDistanceMetric<T> metric)
    {
        double maxChange = 0;
        double exponent = 2.0 / (m - 1);

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(x, i);

            // Compute distances to all centers
            var distances = new double[k];
            bool hasZeroDistance = false;
            int zeroCluster = -1;

            for (int c = 0; c < k; c++)
            {
                var center = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    center[j] = NumOps.FromDouble(centers[c, j]);
                }

                distances[c] = NumOps.ToDouble(metric.Compute(point, center));

                if (distances[c] < 1e-10)
                {
                    hasZeroDistance = true;
                    zeroCluster = c;
                }
            }

            // Update memberships
            for (int c = 0; c < k; c++)
            {
                double oldMembership = _membershipMatrix![i, c];
                double newMembership;

                if (hasZeroDistance)
                {
                    // Point is exactly at a center
                    newMembership = (c == zeroCluster) ? 1.0 : 0.0;
                }
                else
                {
                    // Standard update formula
                    double sum = 0;
                    for (int j = 0; j < k; j++)
                    {
                        sum += Math.Pow(distances[c] / distances[j], exponent);
                    }
                    newMembership = 1.0 / sum;
                }

                _membershipMatrix[i, c] = newMembership;
                maxChange = Math.Max(maxChange, Math.Abs(newMembership - oldMembership));
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
            double minDist = double.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int c = 0; c < NumClusters; c++)
                {
                    var center = GetRow(ClusterCenters, c);
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
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
    public double[,] PredictMembership(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int k = NumClusters;
        double m = _options.Fuzziness;
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        var memberships = new double[n, k];
        double exponent = 2.0 / (m - 1);

        for (int i = 0; i < n; i++)
        {
            var point = GetRow(x, i);

            var distances = new double[k];
            bool hasZeroDistance = false;
            int zeroCluster = -1;

            for (int c = 0; c < k; c++)
            {
                var center = GetRow(ClusterCenters!, c);
                distances[c] = NumOps.ToDouble(metric.Compute(point, center));

                if (distances[c] < 1e-10)
                {
                    hasZeroDistance = true;
                    zeroCluster = c;
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (hasZeroDistance)
                {
                    memberships[i, c] = (c == zeroCluster) ? 1.0 : 0.0;
                }
                else
                {
                    double sum = 0;
                    for (int j = 0; j < k; j++)
                    {
                        sum += Math.Pow(distances[c] / distances[j], exponent);
                    }
                    memberships[i, c] = 1.0 / sum;
                }
            }
        }

        return memberships;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }
}
