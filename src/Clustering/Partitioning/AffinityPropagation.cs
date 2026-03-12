using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Clustering by Passing Messages Between Data Points", "https://doi.org/10.1126/science.1136800", Year = 2007, Authors = "Brendan J. Frey, Delbert Dueck")]
public class AffinityPropagation<T> : ClusteringBase<T>
{
    private readonly AffinityPropagationOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _exemplarIndices;
    private T[,]? _similarityMatrix;

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
    public T[,]? SimilarityMatrix => _similarityMatrix;

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
        T preference = _options.Preference.HasValue
            ? NumOps.FromDouble(_options.Preference.Value)
            : ComputeMedianSimilarity(_similarityMatrix, n);
        for (int i = 0; i < n; i++)
        {
            _similarityMatrix[i, i] = preference;
        }

        // Initialize messages
        var responsibility = new T[n, n];
        var availability = new T[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                responsibility[i, j] = NumOps.Zero;
                availability[i, j] = NumOps.Zero;
            }

        // Message passing
        T damping = NumOps.FromDouble(_options.Damping);
        T oneMinusDamping = NumOps.Subtract(NumOps.One, damping);
        int convergenceCount = 0;
        var prevExemplars = new int[n];

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Update responsibilities
            UpdateResponsibilities(n, _similarityMatrix, availability, responsibility, damping, oneMinusDamping);

            // Update availabilities
            UpdateAvailabilities(n, responsibility, availability, damping, oneMinusDamping);

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
                T maxSim = NumOps.MinValue;
                int bestExemplar = uniqueExemplars[0];
                foreach (int ex in uniqueExemplars)
                {
                    if (NumOps.GreaterThan(_similarityMatrix[i, ex], maxSim))
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

    private T[,] ComputeSimilarityMatrix(Matrix<T> x, int n)
    {
        var similarity = new T[n, n];

        if (_options.AffinityType == AffinityPropagationAffinityType.Precomputed)
        {
            // X is already the similarity matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    similarity[i, j] = x[i, j];
                }
            }
        }
        else
        {
            // Compute negative squared Euclidean distance
            var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

            for (int i = 0; i < n; i++)
            {
                similarity[i, i] = NumOps.Zero;
                var pointI = GetRow(x, i);
                for (int j = i + 1; j < n; j++)
                {
                    var pointJ = GetRow(x, j);
                    T dist = metric.Compute(pointI, pointJ);
                    T negDistSq = NumOps.Negate(NumOps.Multiply(dist, dist));
                    similarity[i, j] = negDistSq;
                    similarity[j, i] = negDistSq;
                }
            }
        }

        return similarity;
    }

    private T ComputeMedianSimilarity(T[,] similarity, int n)
    {
        var values = new List<T>();
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

        values.Sort((a, b) => NumOps.Compare(a, b));
        int mid = values.Count / 2;
        return values.Count % 2 == 0
            ? NumOps.Divide(NumOps.Add(values[mid - 1], values[mid]), NumOps.FromDouble(2.0))
            : values[mid];
    }

    private void UpdateResponsibilities(int n, T[,] similarity, T[,] availability,
        T[,] responsibility, T damping, T oneMinusDamping)
    {
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            // Compute max of a(i,k') + s(i,k') for k' != k
            var asValues = new T[n];
            for (int k = 0; k < n; k++)
            {
                asValues[k] = NumOps.Add(availability[i, k], similarity[i, k]);
            }

            for (int k = 0; k < n; k++)
            {
                // Find max excluding k
                T max1 = NumOps.MinValue;
                T max2 = NumOps.MinValue;

                for (int kp = 0; kp < n; kp++)
                {
                    if (NumOps.GreaterThan(asValues[kp], max1))
                    {
                        max2 = max1;
                        max1 = asValues[kp];
                    }
                    else if (NumOps.GreaterThan(asValues[kp], max2))
                    {
                        max2 = asValues[kp];
                    }
                }

                T maxOther = NumOps.LessThan(NumOps.Abs(NumOps.Subtract(asValues[k], max1)), epsilon) ? max2 : max1;
                T newResp = NumOps.Subtract(similarity[i, k], maxOther);

                // Apply damping
                responsibility[i, k] = NumOps.Add(
                    NumOps.Multiply(damping, responsibility[i, k]),
                    NumOps.Multiply(oneMinusDamping, newResp));
            }
        }
    }

    private void UpdateAvailabilities(int n, T[,] responsibility, T[,] availability, T damping, T oneMinusDamping)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                T newAvail;

                if (i == k)
                {
                    // Self-availability: sum of positive responsibilities
                    T sum = NumOps.Zero;
                    for (int ip = 0; ip < n; ip++)
                    {
                        if (ip != k)
                        {
                            T resp = responsibility[ip, k];
                            if (NumOps.GreaterThan(resp, NumOps.Zero))
                                sum = NumOps.Add(sum, resp);
                        }
                    }
                    newAvail = sum;
                }
                else
                {
                    // Sum of positive responsibilities from others
                    T sum = NumOps.Zero;
                    for (int ip = 0; ip < n; ip++)
                    {
                        if (ip != i && ip != k)
                        {
                            T resp = responsibility[ip, k];
                            if (NumOps.GreaterThan(resp, NumOps.Zero))
                                sum = NumOps.Add(sum, resp);
                        }
                    }
                    T rkk = NumOps.Add(responsibility[k, k], sum);
                    newAvail = NumOps.LessThan(rkk, NumOps.Zero) ? rkk : NumOps.Zero;
                }

                // Apply damping
                availability[i, k] = NumOps.Add(
                    NumOps.Multiply(damping, availability[i, k]),
                    NumOps.Multiply(oneMinusDamping, newAvail));
            }
        }
    }

    private int[] GetExemplars(int n, T[,] responsibility, T[,] availability)
    {
        var exemplars = new int[n];

        for (int i = 0; i < n; i++)
        {
            T maxValue = NumOps.MinValue;
            int bestExemplar = i;

            for (int k = 0; k < n; k++)
            {
                T value = NumOps.Add(responsibility[i, k], availability[i, k]);
                if (NumOps.GreaterThan(value, maxValue))
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
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int k = 0; k < NumClusters; k++)
                {
                    var center = GetRow(ClusterCenters, k);
                    T dist = metric.Compute(point, center);

                    if (NumOps.LessThan(dist, minDist))
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
        return Labels ?? new Vector<T>(0);
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
