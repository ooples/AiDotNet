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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new AffinityPropagation&lt;double&gt;(new AffinityPropagationOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Clustering by Passing Messages Between Data Points", "https://doi.org/10.1126/science.1136800", Year = 2007, Authors = "Brendan J. Frey, Delbert Dueck")]
public partial class AffinityPropagation<T> : ClusteringBase<T>
{
    private readonly AffinityPropagationOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _exemplarIndices;
    private Matrix<T>? _similarityMatrix;

    /// <summary>
    /// Initializes a new AffinityPropagation instance.
    /// </summary>
    /// <param name="options">The affinity propagation options.</param>
    public AffinityPropagation(AffinityPropagationOptions<T>? options = null)
        : base(options ??= new AffinityPropagationOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the indices of exemplar points.
    /// </summary>
    public int[]? ExemplarIndices => _exemplarIndices;

    /// <summary>
    /// Gets the similarity matrix.
    /// </summary>
    public Matrix<T>? SimilarityMatrix => _similarityMatrix;

    /// <inheritdoc />

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

        // Skip normalization for precomputed similarity matrices — the input IS the similarity matrix
        Matrix<T> xForSimilarity = _options.AffinityType == AffinityPropagationAffinityType.Precomputed
            ? x
            : StandardizeColumns(x);

        // Compute similarity matrix
        var similarity = ComputeSimilarityMatrix(xForSimilarity, n);

        // Set preferences (diagonal of similarity matrix)
        T preference = _options.Preference.HasValue
            ? NumOps.FromDouble(_options.Preference.Value)
            : ComputeMedianSimilarity(similarity, n);
        var eye = Engine.TensorEye<T>(n);
        similarity = Engine.TensorWhere(eye, Engine.TensorFullLike(similarity, preference), similarity);
        _similarityMatrix = similarity.ToMatrix();

        // Initialize messages
        var responsibility = Tensor<T>.CreateZeros(new[] { n, n });
        var availability = Tensor<T>.CreateZeros(new[] { n, n });

        // Message passing
        T damping = NumOps.FromDouble(_options.Damping);
        T oneMinusDamping = NumOps.Subtract(NumOps.One, damping);
        int convergenceCount = 0;
        var prevExemplars = new int[n];

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Update responsibilities
            responsibility = UpdateResponsibilities(similarity, availability, responsibility, damping, oneMinusDamping);

            // Update availabilities
            availability = UpdateAvailabilities(eye, responsibility, availability, damping, oneMinusDamping);

            // Check for convergence
            var currentExemplars = GetExemplars(responsibility, availability);

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
        var finalExemplars = GetExemplars(responsibility, availability);
        var uniqueExemplars = finalExemplars.Distinct().ToList();

        _exemplarIndices = uniqueExemplars.ToArray();
        NumClusters = uniqueExemplars.Count;

        // Assign labels: every final exemplar is one of the unique exemplars
        var exemplarToCluster = new Dictionary<int, int>();
        for (int i = 0; i < uniqueExemplars.Count; i++)
        {
            exemplarToCluster[uniqueExemplars[i]] = i;
        }

        var clusterIds = new int[n];
        for (int i = 0; i < n; i++)
        {
            clusterIds[i] = exemplarToCluster[finalExemplars[i]];
        }

        Labels = ClusteringTensorOps.FromIntArray<T>(clusterIds);

        // Set cluster centers as exemplar points
        ClusterCenters = ClusteringTensorOps.SelectRows(x, _exemplarIndices);

        // Merge degenerate clusters (handles single natural cluster with many exemplars)
        int premergeK = NumClusters;
        MergeDegenerateClusters(x);

        // Rebuild _exemplarIndices after merge to match post-merge cluster IDs
        if (NumClusters < premergeK && Labels is not null && ClusterCenters is not null)
        {
            _exemplarIndices = NearestMembersToCenters(x, Labels, ClusterCenters, NumClusters);
        }

        IsTrained = true;
    }

    /// <summary>Centers and scales every column to unit (population) standard deviation; near-constant columns keep scale 1.</summary>
    private static Matrix<T> StandardizeColumns(Matrix<T> x) => ClusteringTensorOps.StandardizeColumns(x, out _, out _);

    /// <summary>For each cluster, the first training sample of that cluster closest to its center (0 when the cluster is empty).</summary>
    private int[] NearestMembersToCenters(Matrix<T> x, Vector<T> labels, Matrix<T> centers, int k)
    {
        var activeCenters = centers.Rows == k ? centers : centers.GetSubMatrix(0, 0, k, centers.Columns);
        var distances = Tensor<T>.FromMatrix(ClusteringTensorOps.SquaredDistances(x, activeCenters));
        var members = ClusteringTensorOps.OneHotMask(labels, k);
        var candidates = Engine.TensorWhere(members, distances, Engine.TensorFullLike(distances, NumOps.MaxValue));
        return Engine.TensorArgMin(candidates, 0).ToArray();
    }

    private Tensor<T> ComputeSimilarityMatrix(Matrix<T> x, int n)
    {
        if (_options.AffinityType == AffinityPropagationAffinityType.Precomputed)
        {
            // X is already the similarity matrix
            var precomputed = x.Columns == n ? x : x.GetSubMatrix(0, 0, n, n);
            return Tensor<T>.FromMatrix(precomputed);
        }

        // Negative squared distance with a zero diagonal
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(x));
        return Engine.TensorNegate(Engine.TensorMultiply(distances, distances));
    }

    private T ComputeMedianSimilarity(Tensor<T> similarity, int n)
    {
        // Median of the off-diagonal similarities (mean of the two middle values for an even count)
        var eye = Engine.TensorEye<T>(n);
        var offDiagonal = Engine.TensorEq(eye, Engine.TensorZerosLike(eye));
        var values = Engine.TensorMsort(Engine.TensorMaskedSelect(similarity, offDiagonal)).ToVector();
        int mid = values.Length / 2;
        return values.Length % 2 == 0
            ? NumOps.Divide(NumOps.Add(values[mid - 1], values[mid]), NumOps.FromDouble(2.0))
            : values[mid];
    }

    private Tensor<T> UpdateResponsibilities(Tensor<T> similarity, Tensor<T> availability,
        Tensor<T> responsibility, T damping, T oneMinusDamping)
    {
        int n = similarity.Shape[0];
        var shape = new[] { n, n };

        // Largest and second largest (with multiplicity) of a(i,k') + s(i,k') per row
        var sums = Engine.TensorAdd(availability, similarity);
        var max1 = Engine.TensorBroadcastTo(Engine.ReduceMax(sums, new[] { 1 }, true), shape);
        var first = Engine.TensorOneHot<T>(Engine.TensorArgMax(sums, 1), n);
        var withoutFirst = Engine.TensorWhere(first, Engine.TensorFullLike(sums, NumOps.MinValue), sums);
        var max2 = Engine.TensorBroadcastTo(Engine.ReduceMax(withoutFirst, new[] { 1 }, true), shape);

        // Max over k' != k: the runner-up where this entry is the maximum
        var isMax = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(sums, max1)), NumOps.FromDouble(1e-10));
        var maxOther = Engine.TensorWhere(isMax, max2, max1);
        var newResponsibility = Engine.TensorSubtract(similarity, maxOther);

        // Apply damping
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(responsibility, damping),
            Engine.TensorMultiplyScalar(newResponsibility, oneMinusDamping));
    }

    private Tensor<T> UpdateAvailabilities(Tensor<T> eye, Tensor<T> responsibility, Tensor<T> availability, T damping, T oneMinusDamping)
    {
        int n = responsibility.Shape[0];
        var shape = new[] { n, n };
        var offDiagonal = Engine.ScalarMinusTensor(NumOps.One, eye);

        // Positive responsibilities r(i',k) for i' != k, and their column sums
        var positive = Engine.TensorMultiply(Engine.TensorReLU(responsibility), offDiagonal);
        var columnSums = Engine.TensorBroadcastTo(Engine.ReduceSum(positive, new[] { 0 }, true), shape);

        // a(k,k) = sum_{i' != k} max(0, r(i',k))
        // a(i,k) = min(0, r(k,k) + sum_{i' != i,k} max(0, r(i',k)))
        var selfResponsibility = Engine.TensorBroadcastTo(
            Engine.ReduceSum(Engine.TensorMultiply(responsibility, eye), new[] { 0 }, true), shape);
        var others = Engine.TensorMin(
            Engine.TensorSubtract(Engine.TensorAdd(selfResponsibility, columnSums), positive), NumOps.Zero);
        var newAvailability = Engine.TensorWhere(eye, columnSums, others);

        // Apply damping
        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(availability, damping),
            Engine.TensorMultiplyScalar(newAvailability, oneMinusDamping));
    }

    private int[] GetExemplars(Tensor<T> responsibility, Tensor<T> availability)
    {
        // First k maximizing r(i,k) + a(i,k)
        return Engine.TensorArgMax(Engine.TensorAdd(responsibility, availability), 1).ToArray();
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

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

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

        return ClusteringTensorOps.SelectRows(originalData, _exemplarIndices);
    }
}
