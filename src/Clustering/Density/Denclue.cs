using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
    private T[][]? _attractors;
    private T[]? _attractorDensities;

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

        // Extract data into T arrays for efficient computation
        var data = new T[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new T[d];
            for (int j = 0; j < d; j++)
            {
                data[i][j] = x[i, j];
            }
        }

        // Find attractors for each point via hill climbing
        var pointAttractors = new T[n][];
        var pointDensities = new T[n];

        for (int i = 0; i < n; i++)
        {
            var result = HillClimb(data[i], data, d);
            pointAttractors[i] = result.attractor;
            pointDensities[i] = result.density;
        }

        // Merge similar attractors and assign cluster labels
        var attractorList = new List<T[]>();
        var densityList = new List<T>();
        var labels = new int[n];
        var attractorMapping = new int[n];
        T minDensityT = NumOps.FromDouble(_options.MinDensity);
        T mergeThresholdT = NumOps.FromDouble(_options.AttractorMergeThreshold);

        for (int i = 0; i < n; i++)
        {
            if (NumOps.LessThan(pointDensities[i], minDensityT))
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
                T dist = EuclideanDistance(pointAttractors[i], attractorList[a]);
                if (NumOps.LessThan(dist, mergeThresholdT))
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
                ClusterCenters[c, j] = _attractors[c][j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        IsTrained = true;
    }

    private (T[] attractor, T density) HillClimb(T[] point, T[][] data, int d)
    {
        var current = (T[])point.Clone();
        T currentDensity = ComputeDensity(current, data, d);
        T convergenceThresholdT = NumOps.FromDouble(_options.ConvergenceThreshold);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Compute step size using mean shift update
            var next = ComputeMeanShiftUpdate(current, data, d);

            T nextDensity = ComputeDensity(next, data, d);

            // Check convergence
            T movement = EuclideanDistance(current, next);
            if (NumOps.LessThan(movement, convergenceThresholdT))
            {
                return (next, nextDensity);
            }

            current = next;
            currentDensity = nextDensity;
        }

        return (current, currentDensity);
    }

    private T[] ComputeMeanShiftUpdate(T[] point, T[][] data, int d)
    {
        T h = NumOps.FromDouble(_options.Bandwidth);
        T h2 = NumOps.Multiply(h, h);
        T two = NumOps.FromDouble(2.0);

        var numerator = new T[d];
        for (int j = 0; j < d; j++) numerator[j] = NumOps.Zero;
        T denominator = NumOps.Zero;

        for (int i = 0; i < data.Length; i++)
        {
            T dist2 = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(point[j], data[i][j]);
                dist2 = NumOps.Add(dist2, NumOps.Multiply(diff, diff));
            }

            T weight = NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, NumOps.Multiply(two, h2))));

            for (int j = 0; j < d; j++)
            {
                numerator[j] = NumOps.Add(numerator[j], NumOps.Multiply(weight, data[i][j]));
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

    private T ComputeDensity(T[] point, T[][] data, int d)
    {
        T h = NumOps.FromDouble(_options.Bandwidth);
        T h2 = NumOps.Multiply(h, h);
        T two = NumOps.FromDouble(2.0);
        T twoPi = NumOps.FromDouble(2.0 * Math.PI);
        T normalization = NumOps.Exp(NumOps.Multiply(
            NumOps.FromDouble(-d / 2.0),
            NumOps.Log(NumOps.Multiply(twoPi, h2))));

        T sum = NumOps.Zero;
        for (int i = 0; i < data.Length; i++)
        {
            T dist2 = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(point[j], data[i][j]);
                dist2 = NumOps.Add(dist2, NumOps.Multiply(diff, diff));
            }

            sum = NumOps.Add(sum, NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, NumOps.Multiply(two, h2)))));
        }

        return NumOps.Divide(NumOps.Multiply(normalization, sum), NumOps.FromDouble(data.Length));
    }

    private T EuclideanDistance(T[] a, T[] b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
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
        T two = NumOps.FromDouble(2.0);
        T twoPi = NumOps.FromDouble(2.0 * Math.PI);
        T normalization = NumOps.Exp(NumOps.Multiply(
            NumOps.FromDouble(-d / 2.0),
            NumOps.Log(NumOps.Multiply(twoPi, h2))));

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
                NumOps.Exp(NumOps.Negate(NumOps.Divide(dist2, NumOps.Multiply(two, h2))))));

        }

        return NumOps.Multiply(normalization, sum);
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
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? throw new InvalidOperationException("Training failed to produce cluster labels.");
    }
}
