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
public class Denclue<T> : ClusteringBase<T>
{
    private readonly DenclueOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private double[][]? _attractors;
    private double[]? _attractorDensities;

    /// <summary>
    /// Initializes a new DENCLUE instance.
    /// </summary>
    /// <param name="options">The DENCLUE options.</param>
    public Denclue(DenclueOptions<T>? options = null)
        : base(options ?? new DenclueOptions<T>())
    {
        _options = options ?? new DenclueOptions<T>();
    }

    /// <summary>
    /// Gets the density attractors (cluster centers).
    /// </summary>
    public double[][]? Attractors => _attractors;

    /// <summary>
    /// Gets the density values at each attractor.
    /// </summary>
    public double[]? AttractorDensities => _attractorDensities;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new Denclue<T>(new DenclueOptions<T>
        {
            Bandwidth = _options.Bandwidth,
            MinDensity = _options.MinDensity,
            ConvergenceThreshold = _options.ConvergenceThreshold,
            AttractorMergeThreshold = _options.AttractorMergeThreshold,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

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

        // Convert data to double array for efficient computation
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                data[i][j] = NumOps.ToDouble(x[i, j]);
            }
        }

        // Find attractors for each point via hill climbing
        var pointAttractors = new double[n][];
        var pointDensities = new double[n];

        for (int i = 0; i < n; i++)
        {
            var result = HillClimb(data[i], data, d);
            pointAttractors[i] = result.attractor;
            pointDensities[i] = result.density;
        }

        // Merge similar attractors and assign cluster labels
        var attractorList = new List<double[]>();
        var densityList = new List<double>();
        var labels = new int[n];
        var attractorMapping = new int[n];

        for (int i = 0; i < n; i++)
        {
            if (pointDensities[i] < _options.MinDensity)
            {
                // Noise point
                labels[i] = -1;
                attractorMapping[i] = -1;
                continue;
            }

            // Find existing attractor this point matches
            int matchedAttractor = -1;
            for (int a = 0; a < attractorList.Count; a++)
            {
                double dist = EuclideanDistance(pointAttractors[i], attractorList[a]);
                if (dist < _options.AttractorMergeThreshold)
                {
                    matchedAttractor = a;
                    break;
                }
            }

            if (matchedAttractor == -1)
            {
                // New attractor
                matchedAttractor = attractorList.Count;
                attractorList.Add(pointAttractors[i]);
                densityList.Add(pointDensities[i]);
            }
            else
            {
                // Update attractor density if this one is higher
                if (pointDensities[i] > densityList[matchedAttractor])
                {
                    attractorList[matchedAttractor] = pointAttractors[i];
                    densityList[matchedAttractor] = pointDensities[i];
                }
            }

            labels[i] = matchedAttractor;
            attractorMapping[i] = matchedAttractor;
        }

        // Store attractors
        _attractors = attractorList.ToArray();
        _attractorDensities = densityList.ToArray();

        // Set cluster results
        NumClusters = attractorList.Count;
        ClusterCenters = new Matrix<T>(NumClusters, d);
        Labels = new Vector<T>(n);

        for (int c = 0; c < NumClusters; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = NumOps.FromDouble(_attractors[c][j]);
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        IsTrained = true;
    }

    private (double[] attractor, double density) HillClimb(double[] point, double[][] data, int d)
    {
        var current = (double[])point.Clone();
        double currentDensity = ComputeDensity(current, data, d);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Compute gradient of density
            var gradient = ComputeDensityGradient(current, data, d);

            // Compute step size using mean shift update
            var next = ComputeMeanShiftUpdate(current, data, d);

            double nextDensity = ComputeDensity(next, data, d);

            // Check convergence
            double movement = EuclideanDistance(current, next);
            if (movement < _options.ConvergenceThreshold)
            {
                return (current, currentDensity);
            }

            current = next;
            currentDensity = nextDensity;
        }

        return (current, currentDensity);
    }

    private double[] ComputeMeanShiftUpdate(double[] point, double[][] data, int d)
    {
        double h = _options.Bandwidth;
        double h2 = h * h;

        var numerator = new double[d];
        double denominator = 0;

        for (int i = 0; i < data.Length; i++)
        {
            double dist2 = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = point[j] - data[i][j];
                dist2 += diff * diff;
            }

            double weight = Math.Exp(-dist2 / (2 * h2));

            for (int j = 0; j < d; j++)
            {
                numerator[j] += weight * data[i][j];
            }
            denominator += weight;
        }

        var result = new double[d];
        if (denominator > 0)
        {
            for (int j = 0; j < d; j++)
            {
                result[j] = numerator[j] / denominator;
            }
        }
        else
        {
            result = (double[])point.Clone();
        }

        return result;
    }

    private double ComputeDensity(double[] point, double[][] data, int d)
    {
        double h = _options.Bandwidth;
        double h2 = h * h;
        double normalization = Math.Pow(2 * Math.PI * h2, -d / 2.0);

        double sum = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double dist2 = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = point[j] - data[i][j];
                dist2 += diff * diff;
            }

            sum += Math.Exp(-dist2 / (2 * h2));
        }

        return normalization * sum / data.Length;
    }

    private double[] ComputeDensityGradient(double[] point, double[][] data, int d)
    {
        double h = _options.Bandwidth;
        double h2 = h * h;
        double normalization = Math.Pow(2 * Math.PI * h2, -d / 2.0) / h2;

        var gradient = new double[d];

        for (int i = 0; i < data.Length; i++)
        {
            double dist2 = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = point[j] - data[i][j];
                dist2 += diff * diff;
            }

            double weight = Math.Exp(-dist2 / (2 * h2));

            for (int j = 0; j < d; j++)
            {
                gradient[j] += weight * (data[i][j] - point[j]);
            }
        }

        for (int j = 0; j < d; j++)
        {
            gradient[j] *= normalization / data.Length;
        }

        return gradient;
    }

    private double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Gets the density estimate at a given point.
    /// </summary>
    /// <param name="point">The point to evaluate.</param>
    /// <returns>The estimated density.</returns>
    public double GetDensity(Vector<T> point)
    {
        ValidateIsTrained();

        if (ClusterCenters is null) return 0;

        int d = NumFeatures;
        var p = new double[d];
        for (int j = 0; j < d; j++)
        {
            p[j] = NumOps.ToDouble(point[j]);
        }

        // Use attractors as representative points for density
        double h = _options.Bandwidth;
        double h2 = h * h;
        double normalization = Math.Pow(2 * Math.PI * h2, -d / 2.0);

        double sum = 0;
        for (int c = 0; c < NumClusters; c++)
        {
            double dist2 = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = p[j] - _attractors![c][j];
                dist2 += diff * diff;
            }

            sum += _attractorDensities![c] * Math.Exp(-dist2 / (2 * h2));
        }

        return sum;
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
            int nearestCluster = -1;

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
