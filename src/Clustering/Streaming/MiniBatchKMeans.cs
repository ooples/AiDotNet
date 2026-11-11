using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Streaming;

/// <summary>
/// Mini-Batch K-Means clustering for large datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mini-Batch K-Means is a variant of K-Means that uses small random batches
/// of data for center updates. This reduces computation time significantly
/// while producing results close to standard K-Means.
/// </para>
/// <para>
/// Algorithm:
/// 1. Initialize K cluster centers (K-Means++ or random)
/// 2. Repeat:
///    a. Sample a mini-batch of b points
///    b. Assign each point to nearest center
///    c. Update centers with weighted average
/// 3. Stop when converged or max iterations reached
/// </para>
/// <para><b>For Beginners:</b> Mini-Batch K-Means is K-Means on a diet.
///
/// Instead of using ALL data for each update:
/// - Pick a small random sample (mini-batch)
/// - Learn from just that sample
/// - Repeat many times
///
/// Why this works:
/// - Random samples represent the whole dataset
/// - Many small updates ≈ one big update
/// - Much faster for large datasets
///
/// Speed comparison (1 million points):
/// - K-Means: Minutes to hours
/// - Mini-Batch: Seconds to minutes
///
/// Quality is usually 90-99% of regular K-Means!
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MiniBatchKMeans&lt;double&gt;(new MiniBatchKMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Web-Scale K-Means Clustering", "https://doi.org/10.1145/1772690.1772862", Year = 2010, Authors = "David Sculley")]
public partial class MiniBatchKMeans<T> : ClusteringBase<T>
{
    private readonly MiniBatchKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private int[]? _centerCounts;

    /// <summary>
    /// Initializes a new Mini-Batch K-Means instance.
    /// </summary>
    /// <param name="options">The Mini-Batch K-Means options.</param>
    public MiniBatchKMeans(MiniBatchKMeansOptions<T>? options = null)
        : base(options ??= new MiniBatchKMeansOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the final inertia (sum of squared distances to nearest center).
    /// </summary>
    public T FinalInertia { get; private set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets the number of iterations performed.
    /// </summary>
    public int IterationsPerformed { get; private set; }

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (MiniBatchKMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        int batchSize = Math.Min(_options.BatchSize, n);
        NumFeatures = d;
        NumClusters = k;

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize centers with K-Means++
        var centers = InitializeCentersKMeansPlusPlus(x, k, n, d, rand);
        var counts = new int[k];
        _centerCounts = counts;
        var noImprovementCount = 0;
        T prevInertia = NumOps.MaxValue;

        // Mini-batch training
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            IterationsPerformed = iter + 1;

            // Sample mini-batch (simple random sampling)
            var batchIndices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(batchSize).ToArray();
            var batch = ClusteringTensorOps.SelectRows(x, batchIndices);

            // Squared distances of the batch to the current centers and the nearest center of each point
            var batchDistances = ClusteringTensorOps.SquaredDistances(batch, centers);
            var batchAssignments = ClusteringTensorOps.ArgMinRows(batchDistances);

            // Update centers: the per-point step c <- (1 - 1/count) c + (1/count) x over the batch is the running
            // mean c <- (N c + sum of new points) / (N + m) for a center with N earlier and m new points.
            centers = RunningMeanUpdate(centers, batch, batchAssignments, counts);

            // Compute inertia on batch for convergence check
            T batchInertia = Engine.TensorSum(ClusteringTensorOps.MinRows(batchDistances).AsTensor());

            if (!ImprovedEnough(batchInertia, prevInertia))
            {
                noImprovementCount++;
                if (noImprovementCount >= _options.MaxNoImprovement)
                {
                    break;
                }
            }
            else
            {
                noImprovementCount = 0;
            }

            prevInertia = batchInertia;
        }

        // Final assignment and inertia computation
        var distances = ClusteringTensorOps.SquaredDistances(x, centers);
        var labels = ClusteringTensorOps.ArgMinRows(distances);
        FinalInertia = Engine.TensorSum(ClusteringTensorOps.MinRows(distances).AsTensor());

        // Set results
        ClusterCenters = centers;
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        Inertia = FinalInertia;
        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <summary>True when the batch inertia fell below 0.9999 of the previous one.</summary>
    private bool ImprovedEnough(T batchInertia, T prevInertia)
        => NumOps.LessThan(batchInertia, NumOps.Multiply(prevInertia, NumOps.FromDouble(0.9999)));

    /// <summary>
    /// Folds a batch into the centers: c = (N c + sum of the m assigned points) / (N + m); centers without new
    /// points are unchanged. Updates the running counts.
    /// </summary>
    private Matrix<T> RunningMeanUpdate(Matrix<T> centers, Matrix<T> batch, int[] assignments, int[] counts)
    {
        int k = centers.Rows;
        var sums = ClusteringTensorOps.ClusterSums(batch, ClusteringTensorOps.FromIntArray<T>(assignments), k, out var batchCounts);
        var prior = ClusteringTensorOps.FromIntArray<T>(counts);
        var scaledCenters = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(centers),
            Engine.TensorBroadcastTo(Tensor<T>.FromVector(prior, new[] { k, 1 }), new[] { k, centers.Columns }));
        var weighted = Engine.TensorAdd(scaledCenters, Tensor<T>.FromMatrix(sums)).ToMatrix();
        var updated = ClusteringTensorOps.RowsWherePositive(
            batchCounts, ClusteringTensorOps.DivideRows(weighted, Engine.Add(prior, batchCounts)), centers);

        var added = ClusteringTensorOps.ToIntArray(batchCounts);
        for (int c = 0; c < k; c++)
        {
            counts[c] += added[c];
        }

        return updated;
    }

    /// <summary>One online step: moves center c towards the point with rate 1 / count.</summary>
    private void MoveTowards(Matrix<T> centers, int c, Vector<T> point, int[] counts)
    {
        counts[c]++;
        T eta = NumOps.Divide(NumOps.One, NumOps.FromDouble(counts[c]));
        T oneMinusEta = NumOps.Subtract(NumOps.One, eta);
        var moved = Engine.Add(Engine.Multiply(centers.GetRow(c), oneMinusEta), Engine.Multiply(point, eta));
        Engine.SetRow(centers, c, moved);
    }

    private Matrix<T> InitializeCentersKMeansPlusPlus(Matrix<T> x, int k, int n, int d, Random rand)
    {
        var chosen = new int[k];

        // First center: random
        chosen[0] = rand.Next(n);

        // Remaining centers: probabilistic based on the squared distance to the nearest chosen center
        Vector<T>? distances = null;
        for (int c = 1; c < k; c++)
        {
            var toLast = ClusteringTensorOps.SquaredDistancesToPoint(x.GetRow(chosen[c - 1]), x);
            distances = distances is null ? toLast : Engine.Min(distances, toLast);
            chosen[c] = ClusteringTensorOps.SampleProportional(distances, rand);
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

    /// <summary>
    /// Performs incremental training with additional data.
    /// </summary>
    /// <param name="x">New data to incorporate.</param>
    public void PartialFit(Matrix<T> x)
    {
        if (!IsTrained)
        {
            Train(x);
            return;
        }

        if (ClusterCenters is null || _centerCounts is null) return;

        int n = x.Rows;
        int k = NumClusters;
        var centers = ClusterCenters.Rows == k ? ClusterCenters : ClusterCenters.GetSubMatrix(0, 0, k, ClusterCenters.Columns);

        // Online updates: every point moves its nearest center, so each assignment sees the earlier updates
        for (int i = 0; i < n; i++)
        {
            var point = x.GetRow(i);
            int bestCluster = ClusteringTensorOps.ArgMin(ClusteringTensorOps.SquaredDistancesToPoint(point, centers));
            MoveTowards(centers, bestCluster, point, _centerCounts);
        }

        // Update stored centers
        if (!ReferenceEquals(centers, ClusterCenters))
        {
            ClusterCenters.SetSubMatrix(0, 0, centers);
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        if (ClusterCenters is null || NumClusters == 0)
        {
            return new Vector<T>(x.Rows);
        }

        var centers = ClusterCenters.Rows == NumClusters
            ? ClusterCenters
            : ClusterCenters.GetSubMatrix(0, 0, NumClusters, ClusterCenters.Columns);
        return ClusteringTensorOps.FromIntArray<T>(
            ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(x, centers)));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }
}
