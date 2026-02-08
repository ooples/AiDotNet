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
public class XMeans<T> : ClusteringBase<T>
{
    private readonly XMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private double _bic;

    /// <summary>
    /// Initializes a new XMeans instance.
    /// </summary>
    /// <param name="options">The XMeans options.</param>
    public XMeans(XMeansOptions<T>? options = null)
        : base(options ?? new XMeansOptions<T>())
    {
        _options = options ?? new XMeansOptions<T>();
    }

    /// <summary>
    /// Gets the final BIC value.
    /// </summary>
    public double BIC => _bic;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new XMeans<T>(new XMeansOptions<T>
        {
            MinClusters = _options.MinClusters,
            MaxClusters = _options.MaxClusters,
            Criterion = _options.Criterion,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

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

        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Start with minimum clusters
        int currentK = _options.MinClusters;

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = currentK,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        });

        kmeans.Train(x);
        var currentLabels = kmeans.Labels!;
        var currentCenters = kmeans.ClusterCenters!;

        // Iteratively try to split clusters
        bool improved = true;
        while (improved && currentK < _options.MaxClusters)
        {
            improved = false;
            var newCenters = new List<double[]>();
            var clusterMapping = new Dictionary<int, List<int>>();

            // For each cluster, decide whether to split
            for (int c = 0; c < currentK; c++)
            {
                // Get points in this cluster
                var clusterPoints = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if ((int)NumOps.ToDouble(currentLabels[i]) == c)
                    {
                        clusterPoints.Add(i);
                    }
                }

                if (clusterPoints.Count < 4)
                {
                    // Too few points to split
                    var center = new double[d];
                    for (int j = 0; j < d; j++)
                    {
                        center[j] = NumOps.ToDouble(currentCenters[c, j]);
                    }
                    clusterMapping[newCenters.Count] = clusterPoints;
                    newCenters.Add(center);
                    continue;
                }

                // Create sub-matrix for this cluster
                var subMatrix = new Matrix<T>(clusterPoints.Count, d);
                for (int i = 0; i < clusterPoints.Count; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        subMatrix[i, j] = x[clusterPoints[i], j];
                    }
                }

                // Compute BIC for parent cluster
                double parentBIC = ComputeClusterBIC(subMatrix, d);

                // Try splitting with K=2
                var subKMeans = new KMeans<T>(new KMeansOptions<T>
                {
                    NumClusters = 2,
                    MaxIterations = Options.MaxIterations,
                    RandomState = Options.RandomState
                });

                subKMeans.Train(subMatrix);

                // Compute BIC for children
                double childBIC = ComputeSplitBIC(subMatrix, subKMeans.Labels!, subKMeans.ClusterCenters!, d);

                if (childBIC < parentBIC && newCenters.Count + 1 < _options.MaxClusters)
                {
                    // Accept split
                    for (int sc = 0; sc < 2; sc++)
                    {
                        var center = new double[d];
                        for (int j = 0; j < d; j++)
                        {
                            center[j] = NumOps.ToDouble(subKMeans.ClusterCenters![sc, j]);
                        }

                        var subClusterPoints = new List<int>();
                        for (int i = 0; i < clusterPoints.Count; i++)
                        {
                            if ((int)NumOps.ToDouble(subKMeans.Labels![i]) == sc)
                            {
                                subClusterPoints.Add(clusterPoints[i]);
                            }
                        }

                        clusterMapping[newCenters.Count] = subClusterPoints;
                        newCenters.Add(center);
                    }
                    improved = true;
                }
                else
                {
                    // Keep parent cluster
                    var center = new double[d];
                    for (int j = 0; j < d; j++)
                    {
                        center[j] = NumOps.ToDouble(currentCenters[c, j]);
                    }
                    clusterMapping[newCenters.Count] = clusterPoints;
                    newCenters.Add(center);
                }
            }

            // Update current state
            currentK = newCenters.Count;
            currentCenters = new Matrix<T>(currentK, d);
            currentLabels = new Vector<T>(n);

            for (int c = 0; c < currentK; c++)
            {
                for (int j = 0; j < d; j++)
                {
                    currentCenters[c, j] = NumOps.FromDouble(newCenters[c][j]);
                }

                foreach (int i in clusterMapping[c])
                {
                    currentLabels[i] = NumOps.FromDouble(c);
                }
            }
        }

        // Set final results
        NumClusters = currentK;
        ClusterCenters = currentCenters;
        Labels = currentLabels;
        _bic = ComputeTotalBIC(x, currentLabels, currentCenters, n, d, currentK);

        IsTrained = true;
    }

    private double ComputeClusterBIC(Matrix<T> clusterData, int d)
    {
        int n = clusterData.Rows;
        if (n == 0) return 0;

        // Compute variance
        double variance = ComputeVariance(clusterData, n, d);

        // Log-likelihood for single Gaussian
        double logL = -n / 2.0 * (d * Math.Log(2 * Math.PI) + d * Math.Log(variance + 1e-10) + d);

        // Number of parameters: d (mean) + 1 (variance)
        int numParams = d + 1;

        return _options.Criterion == InformationCriterion.BIC
            ? -2 * logL + numParams * Math.Log(n)
            : -2 * logL + 2 * numParams;
    }

    private double ComputeSplitBIC(Matrix<T> data, Vector<T> labels, Matrix<T> centers, int d)
    {
        int n = data.Rows;
        if (n == 0) return 0;

        double totalLogL = 0;
        int totalParams = 0;

        for (int c = 0; c < 2; c++)
        {
            var clusterPoints = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if ((int)NumOps.ToDouble(labels[i]) == c)
                {
                    clusterPoints.Add(i);
                }
            }

            if (clusterPoints.Count == 0) continue;

            var clusterData = new Matrix<T>(clusterPoints.Count, d);
            for (int i = 0; i < clusterPoints.Count; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    clusterData[i, j] = data[clusterPoints[i], j];
                }
            }

            double variance = ComputeVariance(clusterData, clusterPoints.Count, d);
            double logL = -clusterPoints.Count / 2.0 * (d * Math.Log(2 * Math.PI) + d * Math.Log(variance + 1e-10) + d);
            totalLogL += logL;
            totalParams += d + 1;
        }

        return _options.Criterion == InformationCriterion.BIC
            ? -2 * totalLogL + totalParams * Math.Log(n)
            : -2 * totalLogL + 2 * totalParams;
    }

    private double ComputeTotalBIC(Matrix<T> data, Vector<T> labels, Matrix<T> centers, int n, int d, int k)
    {
        double totalLogL = 0;
        int totalParams = k * (d + 1);

        for (int c = 0; c < k; c++)
        {
            var clusterPoints = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if ((int)NumOps.ToDouble(labels[i]) == c)
                {
                    clusterPoints.Add(i);
                }
            }

            if (clusterPoints.Count == 0) continue;

            var clusterData = new Matrix<T>(clusterPoints.Count, d);
            for (int i = 0; i < clusterPoints.Count; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    clusterData[i, j] = data[clusterPoints[i], j];
                }
            }

            double variance = ComputeVariance(clusterData, clusterPoints.Count, d);
            double logL = -clusterPoints.Count / 2.0 * (d * Math.Log(2 * Math.PI) + d * Math.Log(variance + 1e-10) + d);
            totalLogL += logL;
        }

        return _options.Criterion == InformationCriterion.BIC
            ? -2 * totalLogL + totalParams * Math.Log(n)
            : -2 * totalLogL + 2 * totalParams;
    }

    private double ComputeVariance(Matrix<T> data, int n, int d)
    {
        if (n <= 1) return 1e-10;

        // Compute mean
        var mean = new double[d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                mean[j] += NumOps.ToDouble(data[i, j]);
            }
        }
        for (int j = 0; j < d; j++)
        {
            mean[j] /= n;
        }

        // Compute variance
        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean[j];
                variance += diff * diff;
            }
        }

        return variance / (n * d);
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

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }
}
