using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// Affinity Propagation clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Affinity Propagation identifies exemplars (cluster centers) by passing messages
/// between data points. It doesn't require specifying the number of clusters.
/// </para>
/// <para>
/// The algorithm maintains two types of messages:
/// - Responsibility r(i,k): How well-suited point k is to serve as exemplar for i
/// - Availability a(i,k): How appropriate it would be for i to choose k as its exemplar
/// </para>
/// <para><b>For Beginners:</b> Affinity Propagation works like a democratic election.
///
/// Each point sends two types of messages:
/// 1. "I think you should be my representative" (responsibility)
/// 2. "I'm available to be your representative" (availability)
///
/// The algorithm iterates until consensus:
/// - Some points emerge as exemplars (elected representatives)
/// - Other points choose their nearest exemplar
/// - Each exemplar defines a cluster
///
/// Benefits:
/// - Automatically finds the number of clusters
/// - Exemplars are actual data points
/// - Works well with any similarity measure
/// </para>
/// </remarks>
public class AffinityPropagation<T> : ClusteringBase<T>
{
    private readonly AffinityPropagationOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _exemplarIndices;
    private double[,]? _similarityMatrix;

    /// <summary>
    /// Initializes a new AffinityPropagation instance.
    /// </summary>
    /// <param name="options">The affinity propagation options.</param>
    public AffinityPropagation(AffinityPropagationOptions<T>? options = null)
        : base(options ?? new AffinityPropagationOptions<T>())
    {
        _options = options ?? new AffinityPropagationOptions<T>();
    }

    /// <summary>
    /// Gets the indices of exemplar points.
    /// </summary>
    public int[]? ExemplarIndices => _exemplarIndices;

    /// <summary>
    /// Gets the similarity matrix.
    /// </summary>
    public double[,]? SimilarityMatrix => _similarityMatrix;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new AffinityPropagation<T>(new AffinityPropagationOptions<T>
        {
            Damping = _options.Damping,
            Preference = _options.Preference,
            MaxIterations = _options.MaxIterations,
            ConvergenceIterations = _options.ConvergenceIterations,
            AffinityType = _options.AffinityType,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (AffinityPropagation<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        NumFeatures = x.Columns;

        if (n < 2)
        {
            throw new ArgumentException("Need at least 2 samples.");
        }

        // Compute similarity matrix
        _similarityMatrix = ComputeSimilarityMatrix(x, n);

        // Set preferences (diagonal of similarity matrix)
        double preference = _options.Preference ?? ComputeMedianSimilarity(_similarityMatrix, n);
        for (int i = 0; i < n; i++)
        {
            _similarityMatrix[i, i] = preference;
        }

        // Initialize messages
        var responsibility = new double[n, n];
        var availability = new double[n, n];

        // Message passing
        double damping = _options.Damping;
        int convergenceCount = 0;
        var prevExemplars = new int[n];

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Update responsibilities
            UpdateResponsibilities(n, _similarityMatrix, availability, responsibility, damping);

            // Update availabilities
            UpdateAvailabilities(n, responsibility, availability, damping);

            // Check for convergence
            var currentExemplars = GetExemplars(n, responsibility, availability);

            if (ExemplarsEqual(prevExemplars, currentExemplars))
            {
                convergenceCount++;
                if (convergenceCount >= _options.ConvergenceIterations)
                {
                    break;
                }
            }
            else
            {
                convergenceCount = 0;
            }

            Array.Copy(currentExemplars, prevExemplars, n);
        }

        // Extract final clusters
        var finalExemplars = GetExemplars(n, responsibility, availability);
        var uniqueExemplars = finalExemplars.Distinct().ToList();

        _exemplarIndices = uniqueExemplars.ToArray();
        NumClusters = uniqueExemplars.Count;

        // Assign labels
        Labels = new Vector<T>(n);
        var exemplarToCluster = new Dictionary<int, int>();
        for (int i = 0; i < uniqueExemplars.Count; i++)
        {
            exemplarToCluster[uniqueExemplars[i]] = i;
        }

        for (int i = 0; i < n; i++)
        {
            int exemplar = finalExemplars[i];
            if (exemplarToCluster.TryGetValue(exemplar, out int cluster))
            {
                Labels[i] = NumOps.FromDouble(cluster);
            }
            else
            {
                // Find nearest exemplar
                double maxSim = double.NegativeInfinity;
                int bestExemplar = uniqueExemplars[0];
                foreach (int ex in uniqueExemplars)
                {
                    if (_similarityMatrix[i, ex] > maxSim)
                    {
                        maxSim = _similarityMatrix[i, ex];
                        bestExemplar = ex;
                    }
                }
                Labels[i] = NumOps.FromDouble(exemplarToCluster[bestExemplar]);
            }
        }

        // Set cluster centers as exemplar points
        ClusterCenters = new Matrix<T>(NumClusters, x.Columns);
        for (int k = 0; k < NumClusters; k++)
        {
            int exemplarIdx = _exemplarIndices[k];
            for (int j = 0; j < x.Columns; j++)
            {
                ClusterCenters[k, j] = x[exemplarIdx, j];
            }
        }

        IsTrained = true;
    }

    private double[,] ComputeSimilarityMatrix(Matrix<T> x, int n)
    {
        var similarity = new double[n, n];

        if (_options.AffinityType == AffinityPropagationAffinityType.Precomputed)
        {
            // X is already the similarity matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    similarity[i, j] = NumOps.ToDouble(x[i, j]);
                }
            }
        }
        else
        {
            // Compute negative squared Euclidean distance
            var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

            for (int i = 0; i < n; i++)
            {
                var pointI = GetRow(x, i);
                for (int j = i + 1; j < n; j++)
                {
                    var pointJ = GetRow(x, j);
                    double dist = NumOps.ToDouble(metric.Compute(pointI, pointJ));
                    double negDistSq = -dist * dist;
                    similarity[i, j] = negDistSq;
                    similarity[j, i] = negDistSq;
                }
            }
        }

        return similarity;
    }

    private double ComputeMedianSimilarity(double[,] similarity, int n)
    {
        var values = new List<double>();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    values.Add(similarity[i, j]);
                }
            }
        }

        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 0
            ? (values[mid - 1] + values[mid]) / 2
            : values[mid];
    }

    private void UpdateResponsibilities(int n, double[,] similarity, double[,] availability,
        double[,] responsibility, double damping)
    {
        for (int i = 0; i < n; i++)
        {
            // Compute max of a(i,k') + s(i,k') for k' != k
            var asValues = new double[n];
            for (int k = 0; k < n; k++)
            {
                asValues[k] = availability[i, k] + similarity[i, k];
            }

            for (int k = 0; k < n; k++)
            {
                // Find max excluding k
                double max1 = double.NegativeInfinity;
                double max2 = double.NegativeInfinity;

                for (int kp = 0; kp < n; kp++)
                {
                    if (asValues[kp] > max1)
                    {
                        max2 = max1;
                        max1 = asValues[kp];
                    }
                    else if (asValues[kp] > max2)
                    {
                        max2 = asValues[kp];
                    }
                }

                double maxOther = (Math.Abs(asValues[k] - max1) < 1e-10) ? max2 : max1;
                double newResp = similarity[i, k] - maxOther;

                // Apply damping
                responsibility[i, k] = damping * responsibility[i, k] + (1 - damping) * newResp;
            }
        }
    }

    private void UpdateAvailabilities(int n, double[,] responsibility, double[,] availability, double damping)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                double newAvail;

                if (i == k)
                {
                    // Self-availability: sum of positive responsibilities
                    double sum = 0;
                    for (int ip = 0; ip < n; ip++)
                    {
                        if (ip != k)
                        {
                            sum += Math.Max(0, responsibility[ip, k]);
                        }
                    }
                    newAvail = sum;
                }
                else
                {
                    // Sum of positive responsibilities from others
                    double sum = 0;
                    for (int ip = 0; ip < n; ip++)
                    {
                        if (ip != i && ip != k)
                        {
                            sum += Math.Max(0, responsibility[ip, k]);
                        }
                    }
                    newAvail = Math.Min(0, responsibility[k, k] + sum);
                }

                // Apply damping
                availability[i, k] = damping * availability[i, k] + (1 - damping) * newAvail;
            }
        }
    }

    private int[] GetExemplars(int n, double[,] responsibility, double[,] availability)
    {
        var exemplars = new int[n];

        for (int i = 0; i < n; i++)
        {
            double maxValue = double.NegativeInfinity;
            int bestExemplar = i;

            for (int k = 0; k < n; k++)
            {
                double value = responsibility[i, k] + availability[i, k];
                if (value > maxValue)
                {
                    maxValue = value;
                    bestExemplar = k;
                }
            }

            exemplars[i] = bestExemplar;
        }

        return exemplars;
    }

    private bool ExemplarsEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
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
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    double dist = NumOps.ToDouble(metric.Compute(point, center));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCluster = k;
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

    /// <summary>
    /// Gets the exemplar points as a matrix.
    /// </summary>
    /// <param name="originalData">The original training data.</param>
    /// <returns>Matrix containing exemplar points.</returns>
    public Matrix<T> GetExemplars(Matrix<T> originalData)
    {
        ValidateIsTrained();

        if (_exemplarIndices is null || _exemplarIndices.Length == 0)
        {
            return new Matrix<T>(0, originalData.Columns);
        }

        var exemplars = new Matrix<T>(_exemplarIndices.Length, originalData.Columns);
        for (int i = 0; i < _exemplarIndices.Length; i++)
        {
            int idx = _exemplarIndices[i];
            for (int j = 0; j < originalData.Columns; j++)
            {
                exemplars[i, j] = originalData[idx, j];
            }
        }

        return exemplars;
    }
}
