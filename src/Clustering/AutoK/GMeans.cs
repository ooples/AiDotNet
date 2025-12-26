using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.AutoK;

/// <summary>
/// G-Means clustering with automatic K determination using Gaussian tests.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// G-Means tests whether data in each cluster comes from a Gaussian distribution.
/// If not, the cluster is split. It uses the Anderson-Darling test projected
/// onto the principal direction connecting cluster centers.
/// </para>
/// <para>
/// Algorithm:
/// 1. Start with K=1 (all data in one cluster)
/// 2. For each cluster, project data onto the main axis
/// 3. Test if projection is Gaussian (Anderson-Darling)
/// 4. If not Gaussian, split the cluster
/// 5. Repeat until all clusters are Gaussian or max K reached
/// </para>
/// <para><b>For Beginners:</b> G-Means assumes good clusters are Gaussian.
///
/// A Gaussian distribution is the famous "bell curve":
/// - Most points near the center
/// - Fewer points as you go farther out
/// - Symmetric left and right
///
/// If a cluster doesn't look like a bell curve:
/// - It might contain multiple groups
/// - Split it into two and check again
///
/// This works well when true clusters are roughly Gaussian.
/// </para>
/// </remarks>
public class GMeans<T> : ClusteringBase<T>
{
    private readonly GMeansOptions<T> _options;

    /// <summary>
    /// Initializes a new GMeans instance.
    /// </summary>
    /// <param name="options">The GMeans options.</param>
    public GMeans(GMeansOptions<T>? options = null)
        : base(options ?? new GMeansOptions<T>())
    {
        _options = options ?? new GMeansOptions<T>();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new GMeans<T>(new GMeansOptions<T>
        {
            MinClusters = _options.MinClusters,
            MaxClusters = _options.MaxClusters,
            SignificanceLevel = _options.SignificanceLevel,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (GMeans<T>)CreateNewInstance();
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
        int currentK = Math.Max(1, _options.MinClusters);

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = currentK,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        });

        kmeans.Train(x);
        var currentLabels = new int[n];
        for (int i = 0; i < n; i++)
        {
            currentLabels[i] = (int)NumOps.ToDouble(kmeans.Labels![i]);
        }

        var currentCenters = new List<double[]>();
        for (int c = 0; c < currentK; c++)
        {
            var center = new double[d];
            for (int j = 0; j < d; j++)
            {
                center[j] = NumOps.ToDouble(kmeans.ClusterCenters![c, j]);
            }
            currentCenters.Add(center);
        }

        // Iteratively test and split clusters
        bool improved = true;
        while (improved && currentCenters.Count < _options.MaxClusters)
        {
            improved = false;
            var newCenters = new List<double[]>();
            var newLabels = new int[n];
            var clusterMapping = new Dictionary<int, int>();

            for (int c = 0; c < currentCenters.Count; c++)
            {
                // Get points in this cluster
                var clusterPoints = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if (currentLabels[i] == c)
                    {
                        clusterPoints.Add(i);
                    }
                }

                if (clusterPoints.Count < 4)
                {
                    // Too few points to test
                    clusterMapping[c] = newCenters.Count;
                    newCenters.Add(currentCenters[c]);
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

                // Test if Gaussian using Anderson-Darling
                bool isGaussian = TestGaussian(subMatrix, clusterPoints.Count, d);

                if (!isGaussian && newCenters.Count + 1 < _options.MaxClusters)
                {
                    // Split this cluster
                    var subKMeans = new KMeans<T>(new KMeansOptions<T>
                    {
                        NumClusters = 2,
                        MaxIterations = Options.MaxIterations,
                        RandomState = Options.RandomState
                    });

                    subKMeans.Train(subMatrix);

                    // Add both children
                    for (int sc = 0; sc < 2; sc++)
                    {
                        var center = new double[d];
                        for (int j = 0; j < d; j++)
                        {
                            center[j] = NumOps.ToDouble(subKMeans.ClusterCenters![sc, j]);
                        }
                        newCenters.Add(center);
                    }

                    // Update labels for points in this cluster
                    for (int i = 0; i < clusterPoints.Count; i++)
                    {
                        int subLabel = (int)NumOps.ToDouble(subKMeans.Labels![i]);
                        newLabels[clusterPoints[i]] = newCenters.Count - 2 + subLabel;
                    }

                    improved = true;
                }
                else
                {
                    // Keep this cluster
                    clusterMapping[c] = newCenters.Count;
                    newCenters.Add(currentCenters[c]);

                    foreach (int i in clusterPoints)
                    {
                        newLabels[i] = newCenters.Count - 1;
                    }
                }
            }

            currentCenters = newCenters;
            currentLabels = newLabels;
        }

        // Set final results
        NumClusters = currentCenters.Count;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        Labels = new Vector<T>(n);

        for (int c = 0; c < NumClusters; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = NumOps.FromDouble(currentCenters[c][j]);
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(currentLabels[i]);
        }

        IsTrained = true;
    }

    private bool TestGaussian(Matrix<T> data, int n, int d)
    {
        if (n < 4) return true;

        // Project data onto principal axis
        // For simplicity, project onto first principal component
        var projections = ProjectToPrincipalAxis(data, n, d);

        // Standardize projections
        double mean = projections.Average();
        double std = Math.Sqrt(projections.Sum(p => (p - mean) * (p - mean)) / n);

        if (std < 1e-10) return true; // Constant data

        var standardized = projections.Select(p => (p - mean) / std).OrderBy(p => p).ToArray();

        // Anderson-Darling test
        double ad = ComputeAndersonDarling(standardized, n);

        // Get critical value for significance level
        double criticalValue = GetADCriticalValue(_options.SignificanceLevel);

        return ad <= criticalValue;
    }

    private double[] ProjectToPrincipalAxis(Matrix<T> data, int n, int d)
    {
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

        // Center data
        var centered = new double[n, d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                centered[i, j] = NumOps.ToDouble(data[i, j]) - mean[j];
            }
        }

        // Compute first principal component using power iteration
        var pc = new double[d];
        var rand = RandomHelper.CreateSeededRandom(42);
        for (int j = 0; j < d; j++)
        {
            pc[j] = rand.NextDouble() - 0.5;
        }

        // Normalize
        double norm = Math.Sqrt(pc.Sum(v => v * v));
        for (int j = 0; j < d; j++)
        {
            pc[j] /= norm;
        }

        // Power iteration
        for (int iter = 0; iter < 20; iter++)
        {
            var newPc = new double[d];

            // Compute X^T * X * pc
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    double dot = 0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += centered[i, k] * pc[k];
                    }
                    newPc[j] += centered[i, j] * dot;
                }
            }

            // Normalize
            norm = Math.Sqrt(newPc.Sum(v => v * v));
            if (norm < 1e-10) break;

            for (int j = 0; j < d; j++)
            {
                pc[j] = newPc[j] / norm;
            }
        }

        // Project data onto PC
        var projections = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                projections[i] += centered[i, j] * pc[j];
            }
        }

        return projections;
    }

    private double ComputeAndersonDarling(double[] sortedStandardized, int n)
    {
        // Anderson-Darling test statistic
        double sum = 0;

        for (int i = 0; i < n; i++)
        {
            double phi_i = NormalCDF(sortedStandardized[i]);
            double phi_n_minus_i = NormalCDF(sortedStandardized[n - 1 - i]);

            // Clamp to avoid log(0)
            phi_i = Math.Max(1e-10, Math.Min(1 - 1e-10, phi_i));
            phi_n_minus_i = Math.Max(1e-10, Math.Min(1 - 1e-10, phi_n_minus_i));

            sum += (2.0 * (i + 1) - 1) * (Math.Log(phi_i) + Math.Log(1 - phi_n_minus_i));
        }

        double ad = -n - sum / n;

        // Apply correction for small samples
        ad *= (1 + 4.0 / n - 25.0 / (n * n));

        return ad;
    }

    private double NormalCDF(double x)
    {
        // Approximation of standard normal CDF
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    private double Erf(double x)
    {
        // Approximation of error function
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double tau = t * Math.Exp(-x * x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));

        return x >= 0 ? 1 - tau : tau - 1;
    }

    private double GetADCriticalValue(double alpha)
    {
        // Critical values for Anderson-Darling test
        // These are approximate values for normal distribution
        if (alpha >= 0.15) return 0.576;
        if (alpha >= 0.10) return 0.656;
        if (alpha >= 0.05) return 0.787;
        if (alpha >= 0.025) return 0.918;
        if (alpha >= 0.01) return 1.092;
        if (alpha >= 0.005) return 1.227;
        if (alpha >= 0.001) return 1.551;
        return 1.8; // Very strict
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
