using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// CLARANS (Clustering Large Applications based on Randomized Search) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLARANS is a partitioning algorithm that uses medoids (actual data points)
/// as cluster centers. It improves on PAM by using randomized search to
/// explore the solution space more efficiently.
/// </para>
/// <para>
/// The algorithm:
/// 1. Start with random medoids
/// 2. For each iteration, consider swapping a medoid with a non-medoid
/// 3. Accept swaps that reduce total cost
/// 4. Repeat from different starting points, keep best result
/// </para>
/// <para><b>For Beginners:</b> CLARANS is like K-Means but uses actual data points.
///
/// Key differences from K-Means:
/// - Cluster centers (medoids) are real data points
/// - More robust: outliers don't pull centers as much
/// - Any distance metric works (not just Euclidean)
///
/// The "randomized" part:
/// - Instead of checking ALL possible swaps (slow!)
/// - Randomly sample potential swaps
/// - Still finds good solutions, just faster
///
/// Think of finding the best meeting spot for friends:
/// - Must be at someone's house (medoid = actual point)
/// - Try swapping whose house, keep improvements
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CLARANS&lt;double&gt;(new CLARANSOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("CLARANS: A Method for Clustering Objects for Spatial Data Mining", "https://doi.org/10.1109/69.971187", Year = 2002, Authors = "Raymond T. Ng, Jiawei Han")]
public partial class CLARANS<T> : ClusteringBase<T>
{
    private readonly CLARANSOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _medoidIndices;
    private T _bestCost = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Initializes a new CLARANS instance.
    /// </summary>
    /// <param name="options">The CLARANS options.</param>
    public CLARANS(CLARANSOptions<T>? options = null)
        : base(options ?? new CLARANSOptions<T>())
    {
        // Use the options passed to base constructor to avoid double instantiation
        _options = (CLARANSOptions<T>)Options;
    }

    /// <summary>
    /// Gets the indices of medoid points.
    /// </summary>
    public int[]? MedoidIndices => _medoidIndices;

    /// <summary>
    /// Gets the best cost found.
    /// </summary>
    public T BestCost => _bestCost;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (CLARANS<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <summary>
    /// Trains the CLARANS clustering model on the given data.
    /// </summary>
    /// <param name="x">The input data matrix.</param>
    public override void Train(Matrix<T> x)
    {
        // CLARANS is unsupervised, so we can ignore labels
        Train(x, new Vector<T>(x.Rows));
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        NumFeatures = d;
        NumClusters = k;

        if (k >= n)
        {
            throw new ArgumentException($"Number of clusters ({k}) must be less than number of samples ({n}).");
        }

        var rand = Random ?? RandomHelper.CreateSecureRandom();
        var metric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Precompute distance matrix for efficiency
        var distMatrix = ComputeDistanceMatrix(x, n, metric);

        // Compute max neighbors to explore
        int maxNeighbor = _options.MaxNeighbor ?? Math.Max(250, (int)(0.0125 * n * k));

        _bestCost = NumOps.MaxValue;
        int[]? bestMedoids = null;

        // Run numLocal iterations
        for (int iter = 0; iter < _options.NumLocal; iter++)
        {
            // Initialize random medoids
            var currentMedoids = SelectRandomMedoids(n, k, rand);
            T currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);

            // Local search: random single-medoid swaps, accepted when they lower the cost
            int neighborsExplored = 0;
            while (neighborsExplored < maxNeighbor)
            {
                // Select random medoid to swap
                int medoidIdx = rand.Next(k);

                // Select random non-medoid
                int newMedoid;
                do
                {
                    newMedoid = rand.Next(n);
                } while (currentMedoids.Contains(newMedoid));

                // Compute cost of swap
                var newMedoids = (int[])currentMedoids.Clone();
                newMedoids[medoidIdx] = newMedoid;
                T newCost = ComputeTotalCost(distMatrix, newMedoids, n);

                if (IsLower(newCost, currentCost))
                {
                    // Accept swap, reset counter
                    currentMedoids = newMedoids;
                    currentCost = newCost;
                    neighborsExplored = 0;
                }
                else
                {
                    neighborsExplored++;
                }
            }

            // Update best if improved
            if (IsLower(currentCost, _bestCost))
            {
                _bestCost = currentCost;
                bestMedoids = (int[])currentMedoids.Clone();
            }
        }

        _medoidIndices = bestMedoids ?? SelectRandomMedoids(n, k, rand);

        // Set cluster centers as medoid points
        ClusterCenters = ClusteringTensorOps.SelectRows(x, _medoidIndices);

        // Assign labels
        Labels = ClusteringTensorOps.FromIntArray<T>(
            ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SelectColumns(distMatrix, _medoidIndices).ToMatrix()));

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <summary>Scalar comparison of two costs.</summary>
    private bool IsLower(T candidate, T reference) => NumOps.LessThan(candidate, reference);

    private Tensor<T> ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        // Symmetric pairwise distances with a zero diagonal
        return Tensor<T>.FromMatrix(metric.ComputePairwise(x));
    }

    private int[] SelectRandomMedoids(int n, int k, Random rand)
    {
        var indices = new HashSet<int>();
        while (indices.Count < k)
        {
            indices.Add(rand.Next(n));
        }
        return indices.ToArray();
    }

    private T ComputeTotalCost(Tensor<T> distMatrix, int[] medoids, int n)
    {
        // Sum over points of the distance to their nearest medoid
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
        return Labels ?? new Vector<T>(0);
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
