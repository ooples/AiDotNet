using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// K-Medoids (PAM - Partitioning Around Medoids) clustering implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// K-Medoids clusters data into k groups using medoids (actual data points)
/// as cluster centers. It minimizes the sum of dissimilarities between
/// points and their assigned medoids.
/// </para>
/// <para>
/// Algorithm (PAM):
/// 1. Initialize k medoids (randomly or using BUILD)
/// 2. Assign each point to nearest medoid
/// 3. For each (medoid, non-medoid) pair:
///    - Compute cost of swapping them
///    - If cost decreases, perform swap
/// 4. Repeat until no improvement
/// </para>
/// <para><b>For Beginners:</b> K-Medoids is K-Means with real data points as centers.
///
/// Key benefits:
/// - More robust to outliers than K-Means
/// - Works with any distance metric
/// - Cluster centers are interpretable (they're real data points)
///
/// Example use case:
/// - Clustering customers: The medoid is a "typical" customer
/// - You can examine this customer directly
/// - With K-Means, the center might not represent any real customer
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new KMedoids&lt;double&gt;(new KMedoidsOptions&lt;double&gt;()))
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
[ResearchPaper("Finding Groups in Data: An Introduction to Cluster Analysis", "https://doi.org/10.1002/9780470316801", Year = 1990, Authors = "Leonard Kaufman, Peter J. Rousseeuw")]
public partial class KMedoids<T> : ClusteringBase<T>
{
    private readonly KMedoidsOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _medoidIndices;

    /// <summary>
    /// Initializes a new KMedoids instance.
    /// </summary>
    /// <param name="options">The KMedoids options.</param>
    public KMedoids(KMedoidsOptions<T>? options = null)
        : base(options ??= new KMedoidsOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the indices of medoid points in the original data.
    /// </summary>
    public int[]? MedoidIndices => _medoidIndices;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (KMedoids<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        NumFeatures = d;
        NumClusters = k;

        if (k > n)
        {
            throw new ArgumentException($"Number of clusters ({k}) cannot exceed number of samples ({n}).");
        }

        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Precompute distance matrix
        var distMatrix = ComputeDistanceMatrix(x, n, metric);

        // Initialize medoids
        int[] medoids;
        switch (_options.Init)
        {
            case KMedoidsInit.Build:
                medoids = InitializeBuild(distMatrix, n, k);
                break;
            case KMedoidsInit.KMedoidsPlusPlus:
                medoids = InitializeKMedoidsPlusPlus(distMatrix, n, k, rand);
                break;
            default:
                medoids = InitializeRandom(n, k, rand);
                break;
        }

        // Run selected algorithm
        switch (_options.Algorithm)
        {
            case KMedoidsAlgorithm.FastPAM:
                medoids = RunFastPAM(distMatrix, medoids, n, k);
                break;
            case KMedoidsAlgorithm.Alternate:
                medoids = RunAlternate(distMatrix, medoids, n, k);
                break;
            default:
                medoids = RunPAM(distMatrix, medoids, n, k);
                break;
        }

        _medoidIndices = medoids;

        // Assign labels and compute inertia
        var toMedoids = ClusteringTensorOps.SelectColumns(distMatrix, medoids).ToMatrix();
        Labels = ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.ArgMinRows(toMedoids));
        Inertia = Engine.TensorSum(ClusteringTensorOps.MinRows(toMedoids).AsTensor());

        // Set cluster centers as medoid points
        ClusterCenters = ClusteringTensorOps.SelectRows(x, medoids);
        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private Tensor<T> ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        // Symmetric pairwise distances with a zero diagonal
        return Tensor<T>.FromMatrix(metric.ComputePairwise(x));
    }

    private int[] InitializeRandom(int n, int k, Random rand)
    {
        var indices = new HashSet<int>();
        while (indices.Count < k)
        {
            indices.Add(rand.Next(n));
        }
        return indices.ToArray();
    }

    private int[] InitializeKMedoidsPlusPlus(Tensor<T> distMatrix, int n, int k, Random rand)
    {
        var medoids = new List<int>();

        // First medoid: random
        medoids.Add(rand.Next(n));

        // Remaining medoids: probabilistic based on the squared distance to the nearest medoid
        for (int m = 1; m < k; m++)
        {
            var minDistances = ClusteringTensorOps.MinRows(ClusteringTensorOps.SelectColumns(distMatrix, medoids.ToArray()).ToMatrix());
            var weights = Engine.Multiply(minDistances, minDistances);

            // Select the next medoid among the non-medoids, proportional to the squared distance
            var candidates = Enumerable.Range(0, n).Where(i => !medoids.Contains(i)).ToArray();
            int position = ClusteringTensorOps.SampleProportionalOrNone(ClusteringTensorOps.SelectElements(weights, candidates), rand);
            medoids.Add(position >= 0 ? candidates[position] : 0);
        }

        return medoids.ToArray();
    }

    private int[] InitializeBuild(Tensor<T> distMatrix, int n, int k)
    {
        var medoids = new List<int>();

        // First medoid: point with minimum total distance to all others
        medoids.Add(ClusteringTensorOps.ArgMin(Engine.ReduceSum(distMatrix, new[] { 1 }, false).ToVector()));

        // Remaining medoids: greedily add point that reduces cost most
        while (medoids.Count < k)
        {
            var candidates = Enumerable.Range(0, n).Where(i => !medoids.Contains(i)).ToArray();
            if (candidates.Length > 0)
            {
                // gain(i) = sum_j max(nearest medoid distance of j - d(j, i), 0)
                var current = ClusteringTensorOps.MinRows(ClusteringTensorOps.SelectColumns(distMatrix, medoids.ToArray()).ToMatrix());
                var reductions = Engine.TensorSubtract(
                    Engine.TensorBroadcastTo(Tensor<T>.FromVector(current, new[] { n, 1 }), new[] { n, n }), distMatrix);
                var gains = Engine.ReduceSum(Engine.TensorReLU(reductions), new[] { 0 }, false).ToVector();
                medoids.Add(candidates[ClusteringTensorOps.ArgMax(ClusteringTensorOps.SelectElements(gains, candidates))]);
            }
            else
            {
                // Fallback to random
                var rand = Random ?? RandomHelper.CreateSecureRandom();
                int candidate;
                do
                {
                    candidate = rand.Next(n);
                } while (medoids.Contains(candidate));
                medoids.Add(candidate);
            }
        }

        return medoids.ToArray();
    }

    private int[] RunPAM(Tensor<T> distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        T currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            bool improved = false;

            // Try swaps medoid by medoid; the first candidate (in index order) that lowers the cost is taken
            for (int m = 0; m < k && !improved; m++)
            {
                var costs = SwapCosts(distMatrix, currentMedoids, m, n);
                var accept = ClusteringTensorOps.LessThanFlags(costs, ImprovementThreshold(currentCost));
                for (int i = 0; i < n; i++)
                {
                    if (accept[i] && Array.IndexOf(currentMedoids, i) < 0)
                    {
                        currentMedoids[m] = i;
                        currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);
                        improved = true;
                        break;
                    }
                }
            }

            if (!improved) break;
        }

        return currentMedoids;
    }

    /// <summary>The cost a swap must beat: current cost minus the tolerance.</summary>
    private T ImprovementThreshold(T currentCost) => NumOps.Subtract(currentCost, NumOps.FromDouble(Options.Tolerance));

    /// <summary>
    /// Total cost of replacing medoid m by each point i: sum over j of min(nearest other medoid of j, d(j, i)).
    /// </summary>
    private Vector<T> SwapCosts(Tensor<T> distMatrix, int[] medoids, int m, int n)
    {
        var others = medoids.Where((_, index) => index != m).ToArray();
        Tensor<T> candidateCosts = distMatrix;
        if (others.Length > 0)
        {
            var nearestOther = ClusteringTensorOps.MinRows(ClusteringTensorOps.SelectColumns(distMatrix, others).ToMatrix());
            candidateCosts = Engine.TensorMin(
                Engine.TensorBroadcastTo(Tensor<T>.FromVector(nearestOther, new[] { n, 1 }), new[] { n, n }), distMatrix);
        }

        return Engine.ReduceSum(candidateCosts, new[] { 0 }, false).ToVector();
    }

    private int[] RunFastPAM(Tensor<T> distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Nearest and second-nearest medoid of every point
            UpdateNearestMedoids(distMatrix, currentMedoids, n, k, out var nearest, out var nearestDist, out var secondDist);

            // delta[m, i] = sum over points of the change in their nearest distance when medoid m becomes point i:
            //   points of m move to min(d(j, i), second_j); the others move to d(j, i) only when it is closer.
            var nearestMask = ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(nearest), k);
            var nearestColumn = Engine.TensorBroadcastTo(Tensor<T>.FromVector(nearestDist, new[] { n, 1 }), new[] { n, n });
            var secondColumn = Engine.TensorBroadcastTo(Tensor<T>.FromVector(secondDist, new[] { n, 1 }), new[] { n, n });
            var ownChange = Engine.TensorSubtract(Engine.TensorMin(distMatrix, secondColumn), nearestColumn);
            var otherChange = Engine.TensorMin(Engine.TensorSubtract(distMatrix, nearestColumn), NumOps.Zero);
            var otherMask = Engine.ScalarMinusTensor(NumOps.One, nearestMask);
            var delta = Engine.TensorAdd(
                Engine.TensorMatMul(Engine.TensorTranspose(nearestMask), ownChange),
                Engine.TensorMatMul(Engine.TensorTranspose(otherMask), otherChange));

            // Only non-medoid candidates count; the best swap is the first (m, i) with the most negative delta
            var medoidFlags = new int[n];
            foreach (int medoid in currentMedoids)
            {
                medoidFlags[medoid] = 1;
            }

            var isMedoid = ClusteringTensorOps.FromIntArray<T>(medoidFlags);
            var excluded = Engine.TensorBroadcastTo(Tensor<T>.FromVector(isMedoid, new[] { 1, n }), new[] { k, n });
            var candidateDelta = Engine.TensorWhere(excluded, Engine.TensorZerosLike(delta), delta);
            var flat = candidateDelta.ToVector();
            int best = ClusteringTensorOps.ArgMin(flat);

            if (!IsImprovingSwap(Engine.TensorMinValue(candidateDelta)))
            {
                break;
            }

            // Perform best swap
            currentMedoids[best / n] = best % n;
        }

        return currentMedoids;
    }

    /// <summary>True when the swap delta is below minus the tolerance.</summary>
    private bool IsImprovingSwap(T bestDelta)
        => NumOps.LessThan(bestDelta, NumOps.Zero) && NumOps.LessThan(bestDelta, NumOps.Negate(NumOps.FromDouble(Options.Tolerance)));

    private int[] RunAlternate(Tensor<T> distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        var labels = new int[n];

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Assign step
            var nearest = ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SelectColumns(distMatrix, currentMedoids).ToMatrix());
            bool changed = !nearest.SequenceEqual(labels);
            labels = nearest;

            // Update step: in each cluster, the member with the smallest total distance to the other members
            var mask = ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(labels), k);
            var withinCost = Engine.ReduceSum(
                Engine.TensorMultiply(Engine.TensorMatMul(distMatrix, mask), mask), new[] { 1 }, false).ToVector();
            for (int m = 0; m < k; m++)
            {
                var clusterPoints = Enumerable.Range(0, n).Where(i => labels[i] == m).ToArray();
                if (clusterPoints.Length == 0) continue;

                int bestMedoid = clusterPoints[ClusteringTensorOps.ArgMin(ClusteringTensorOps.SelectElements(withinCost, clusterPoints))];
                if (currentMedoids[m] != bestMedoid)
                {
                    currentMedoids[m] = bestMedoid;
                    changed = true;
                }
            }

            if (!changed) break;
        }

        return currentMedoids;
    }

    private void UpdateNearestMedoids(Tensor<T> distMatrix, int[] medoids, int n, int k,
        out int[] nearest, out Vector<T> nearestDist, out Vector<T> secondDist)
    {
        var toMedoids = ClusteringTensorOps.SelectColumns(distMatrix, medoids);
        var asMatrix = toMedoids.ToMatrix();
        nearest = ClusteringTensorOps.ArgMinRows(asMatrix);
        nearestDist = ClusteringTensorOps.MinRows(asMatrix);

        // Second-nearest: the smallest distance once the nearest medoid's entry is set aside
        var nearestMask = ClusteringTensorOps.OneHotMask(ClusteringTensorOps.FromIntArray<T>(nearest), k);
        var rest = Engine.TensorWhere(nearestMask, Engine.TensorFullLike(toMedoids, NumOps.MaxValue), toMedoids);
        secondDist = ClusteringTensorOps.MinRows(rest.ToMatrix());
    }

    private T ComputeTotalCost(Tensor<T> distMatrix, int[] medoids, int n)
    {
        return Engine.TensorSum(ClusteringTensorOps.MinRows(ClusteringTensorOps.SelectColumns(distMatrix, medoids).ToMatrix()).AsTensor());
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

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
        return Labels ?? throw new InvalidOperationException("Training failed to produce cluster labels.");
    }

    /// <summary>
    /// Gets the medoid points as a matrix.
    /// </summary>
    /// <param name="originalData">The original training data.</param>
    /// <returns>Matrix containing medoid points.</returns>
    public Matrix<T> GetMedoids(Matrix<T> originalData)
    {
        ValidateIsTrained();

        if (_medoidIndices is null || _medoidIndices.Length == 0)
        {
            return new Matrix<T>(0, originalData.Columns);
        }

        return ClusteringTensorOps.SelectRows(originalData, _medoidIndices);
    }
}
