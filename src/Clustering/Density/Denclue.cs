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
/// var options = new DenclueOptions&lt;double&gt;();
/// var denclue = new Denclue&lt;double&gt;(options);
/// denclue.Train(dataMatrix);
/// Vector<double> labels = denclue.Labels;
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("DENCLUE: A New Approach for Discovering Density-Based Clusters in Large Spatial Databases", "https://link.springer.com/chapter/10.1007/978-1-4615-5669-5_7", Year = 1998, Authors = "Alexander Hinneburg, Daniel A. Keim")]
public class Denclue<T> : ClusteringBase<T>
{
    private readonly DenclueOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[][]? _attractors;
    private T[]? _attractorDensities;
    private double[]? _featureMeans;
    private double[]? _featureStds;

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

        // Normalize features for scale-invariant density estimation
        _featureMeans = new double[d];
        _featureStds = new double[d];
        var xNorm = new Matrix<T>(n, d);
        for (int j = 0; j < d; j++)
        {
            double sum = 0, varSum = 0;
            for (int i = 0; i < n; i++)
                sum += NumOps.ToDouble(x[i, j]);
            _featureMeans[j] = sum / n;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(x[i, j]) - _featureMeans[j];
                varSum += diff * diff;
            }
            _featureStds[j] = Math.Sqrt(varSum / n);
            if (_featureStds[j] < 1e-10) _featureStds[j] = 1.0;
            for (int i = 0; i < n; i++)
                xNorm[i, j] = NumOps.FromDouble((NumOps.ToDouble(x[i, j]) - _featureMeans[j]) / _featureStds[j]);
        }
        x = xNorm;

        T minDensity = NumOps.FromDouble(_options.MinDensity);
        T mergeThreshold = NumOps.FromDouble(_options.AttractorMergeThreshold);

        // Find attractors for each point via hill climbing
        var pointAttractors = new T[n][];
        var pointDensities = new T[n];

        for (int i = 0; i < n; i++)
        {
            var point = new T[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = x[i, j];
            }
            var result = HillClimb(point, x, n, d);
            pointAttractors[i] = result.attractor;
            pointDensities[i] = result.density;
        }

        // Merge similar attractors and assign cluster labels
        var attractorList = new List<T[]>();
        var densityList = new List<T>();
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            if (NumOps.LessThan(pointDensities[i], minDensity))
            {
                // Noise point
                labels[i] = -1;
                continue;
            }

            // Find existing attractor this point matches
            int matchedAttractor = -1;
            for (int a = 0; a < attractorList.Count; a++)
            {
                T dist = EuclideanDistanceT(pointAttractors[i], attractorList[a], d);
                if (NumOps.LessThan(dist, mergeThreshold))
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
                if (NumOps.GreaterThan(pointDensities[i], densityList[matchedAttractor]))
                {
                    attractorList[matchedAttractor] = pointAttractors[i];
                    densityList[matchedAttractor] = pointDensities[i];
                }
            }

            labels[i] = matchedAttractor;
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
                ClusterCenters[c, j] = _attractors[c][j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        IsTrained = true;
    }

    // Predict override is below (after the attractor computation methods)

    private (T[] attractor, T density) HillClimb(T[] point, Matrix<T> data, int n, int d)
    {
        var current = (T[])point.Clone();
        T currentDensity = ComputeDensity(current, data, n, d);
        T convergenceThreshold = NumOps.FromDouble(_options.ConvergenceThreshold);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Compute step size using mean shift update
            var next = ComputeMeanShiftUpdate(current, data, n, d);

            T nextDensity = ComputeDensity(next, data, n, d);

            // Check convergence
            T movement = EuclideanDistanceT(current, next, d);
            if (NumOps.LessThan(movement, convergenceThreshold))
            {
                return (next, nextDensity);
            }

            current = next;
            currentDensity = nextDensity;
        }

        return (current, currentDensity);
    }

    private T[] ComputeMeanShiftUpdate(T[] point, Matrix<T> data, int n, int d)
    {
        T h = NumOps.FromDouble(_options.Bandwidth);
        T h2 = NumOps.Multiply(h, h);
        T twoH2 = NumOps.Multiply(NumOps.FromDouble(2.0), h2);

        var numerator = new T[d];
        T denominator = NumOps.Zero;
        for (int j = 0; j < d; j++)
        {
            numerator[j] = NumOps.Zero;
        }

        for (int i = 0; i < n; i++)
        {
            T dist2 = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(point[j], data[i, j]);
                dist2 = NumOps.Add(dist2, NumOps.Multiply(diff, diff));
            }

            T weight = NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, twoH2)));

            for (int j = 0; j < d; j++)
            {
                numerator[j] = NumOps.Add(numerator[j], NumOps.Multiply(weight, data[i, j]));
            }
            denominator = NumOps.Add(denominator, weight);
        }


        var result = new T[d];
        if (NumOps.GreaterThan(denominator, NumOps.Zero))
        {
            for (int j = 0; j < d; j++)
            {
                result[j] = NumOps.Divide(numerator[j], denominator);
            }
        }
        else
        {
            result = (T[])point.Clone();
        }

        return result;
    }

    private T ComputeDensity(T[] point, Matrix<T> data, int n, int d)
    {
        T h = NumOps.FromDouble(_options.Bandwidth);
        T h2 = NumOps.Multiply(h, h);
        T twoH2 = NumOps.Multiply(NumOps.FromDouble(2.0), h2);
        // Normalization: (2*PI*h^2)^(-d/2)
        double normDouble = Math.Pow(2 * Math.PI * _options.Bandwidth * _options.Bandwidth, -d / 2.0);
        T normalization = NumOps.FromDouble(normDouble);

        T sum = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T dist2 = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(point[j], data[i, j]);
                dist2 = NumOps.Add(dist2, NumOps.Multiply(diff, diff));
            }

            sum = NumOps.Add(sum, NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, twoH2))));
        }

        return NumOps.Divide(NumOps.Multiply(normalization, sum), NumOps.FromDouble(n));
    }

    private T EuclideanDistanceT(T[] a, T[] b, int d)
    {
        T sum = NumOps.Zero;
        for (int j = 0; j < d; j++)
        {
            T diff = NumOps.Subtract(a[j], b[j]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
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

        var p = new T[d];
        for (int j = 0; j < d; j++)
        {
            p[j] = point[j];
        }

        // Use attractors as representative points for density
        T h = NumOps.FromDouble(_options.Bandwidth);
        T h2 = NumOps.Multiply(h, h);
        T twoH2 = NumOps.Multiply(NumOps.FromDouble(2.0), h2);
        // Normalization: (2*PI*h^2)^(-d/2)
        double normDouble = Math.Pow(2 * Math.PI * _options.Bandwidth * _options.Bandwidth, -d / 2.0);
        T normalization = NumOps.FromDouble(normDouble);


        T sum = NumOps.Zero;
        for (int c = 0; c < NumClusters; c++)
        {
            T dist2 = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(p[j], _attractors[c][j]);
                dist2 = NumOps.Add(dist2, NumOps.Multiply(diff, diff));
            }

            sum = NumOps.Add(sum, NumOps.Multiply(_attractorDensities[c],
                NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, twoH2)))));
        }

        return NumOps.Multiply(normalization, sum);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        // Normalize input using saved parameters from training
        if (_featureMeans is not null && _featureStds is not null)
        {
            var xNorm = new Matrix<T>(x.Rows, x.Columns);
            for (int i = 0; i < x.Rows; i++)
                for (int j = 0; j < x.Columns; j++)
                    xNorm[i, j] = NumOps.FromDouble(
                        (NumOps.ToDouble(x[i, j]) - _featureMeans[j]) / _featureStds[j]);
            x = xNorm;
        }

        var labels = new Vector<T>(x.Rows);
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            T minDist = NumOps.MaxValue;
            int nearestCluster = -1;

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

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (Denclue<T>)CreateNewInstance();
        clone._attractors = _attractors?.Select(a => (T[])a.Clone()).ToArray();
        clone._attractorDensities = _attractorDensities?.ToArray();
        clone._featureMeans = _featureMeans?.ToArray();
        clone._featureStds = _featureStds?.ToArray();
        clone.NumClusters = NumClusters;
        clone.NumFeatures = NumFeatures;
        clone.IsTrained = IsTrained;

        if (Labels is not null)
        {
            clone.Labels = new Vector<T>(Labels.Length);
            for (int i = 0; i < Labels.Length; i++)
                clone.Labels[i] = Labels[i];
        }

        if (ClusterCenters is not null)
        {
            clone.ClusterCenters = new Matrix<T>(ClusterCenters.Rows, ClusterCenters.Columns);
            for (int i = 0; i < ClusterCenters.Rows; i++)
                for (int j = 0; j < ClusterCenters.Columns; j++)
                    clone.ClusterCenters[i, j] = ClusterCenters[i, j];
        }

        return clone;
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? throw new InvalidOperationException("Training failed to produce cluster labels.");
    }
}
