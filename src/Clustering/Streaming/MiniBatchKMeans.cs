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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Web-Scale K-Means Clustering", "https://doi.org/10.1145/1772690.1772862", Year = 2010, Authors = "David Sculley")]
public class MiniBatchKMeans<T> : ClusteringBase<T>
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
        : base(options ?? new MiniBatchKMeansOptions<T>())
    {
        _options = options ?? new MiniBatchKMeansOptions<T>();
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
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new MiniBatchKMeans<T>(new MiniBatchKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            BatchSize = _options.BatchSize,
            MaxNoImprovement = _options.MaxNoImprovement,
            ReassignEmptyClusters = _options.ReassignEmptyClusters,
            MaxIterations = _options.MaxIterations
        });
    }

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
        _centerCounts = new int[k];
        var noImprovementCount = 0;
        T prevInertia = NumOps.MaxValue;
        T convergenceFactor = NumOps.FromDouble(0.9999);

        // Mini-batch training
        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            IterationsPerformed = iter + 1;

            // Sample mini-batch (simple random sampling)
            var batchIndices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(batchSize).ToArray();

            // Cache distances for batch
            var batchDistances = new T[batchSize, k];
            var batchAssignments = new int[batchSize];

            for (int b = 0; b < batchSize; b++)
            {
                int pointIdx = batchIndices[b];
                T minDist = NumOps.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < k; c++)
                {
                    T dist = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        T diff = NumOps.Subtract(x[pointIdx, j], centers[c][j]);
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }

                    batchDistances[b, c] = dist;
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                batchAssignments[b] = bestCluster;
            }

            // Update centers
            for (int b = 0; b < batchSize; b++)
            {
                int pointIdx = batchIndices[b];
                int c = batchAssignments[b];
                _centerCounts[c]++;

                T eta = NumOps.Divide(NumOps.One, NumOps.FromDouble(_centerCounts[c]));
                T oneMinusEta = NumOps.Subtract(NumOps.One, eta);

                for (int j = 0; j < d; j++)
                {
                    centers[c][j] = NumOps.Add(
                        NumOps.Multiply(oneMinusEta, centers[c][j]),
                        NumOps.Multiply(eta, x[pointIdx, j]));
                }
            }

            // Compute inertia on batch for convergence check
            T batchInertia = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                batchInertia = NumOps.Add(batchInertia, batchDistances[b, batchAssignments[b]]);
            }

            if (!NumOps.LessThan(batchInertia, NumOps.Multiply(prevInertia, convergenceFactor)))
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
        var labels = new int[n];
        FinalInertia = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T minDist = NumOps.MaxValue;
            int bestCluster = 0;

            for (int c = 0; c < k; c++)
            {
                T dist = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    T diff = NumOps.Subtract(x[i, j], centers[c][j]);
                    dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                }

                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                    bestCluster = c;
                }
            }

            labels[i] = bestCluster;
            FinalInertia = NumOps.Add(FinalInertia, minDist);
        }

        // Set results
        ClusterCenters = new Matrix<T>(k, d);
        Labels = new Vector<T>(n);

        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = centers[c][j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        Inertia = FinalInertia;
        IsTrained = true;
    }

    private T[][] InitializeCentersKMeansPlusPlus(Matrix<T> x, int k, int n, int d, Random rand)
    {
        var centers = new T[k][];

        // First center: random
        int firstIdx = rand.Next(n);
        centers[0] = new T[d];
        for (int j = 0; j < d; j++)
        {
            centers[0][j] = x[firstIdx, j];
        }

        // Remaining centers: probabilistic based on distance
        // Keep distances as double at random selection boundary
        var distances = new double[n];
        for (int i = 0; i < n; i++)
        {
            distances[i] = double.MaxValue;
        }

        for (int c = 1; c < k; c++)
        {
            double totalWeight = 0;

            for (int i = 0; i < n; i++)
            {
                T distT = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    T diff = NumOps.Subtract(x[i, j], centers[c - 1][j]);
                    distT = NumOps.Add(distT, NumOps.Multiply(diff, diff));
                }

                double dist = NumOps.ToDouble(distT);
                distances[i] = Math.Min(distances[i], dist);
                totalWeight += distances[i];
            }

            double threshold = rand.NextDouble() * totalWeight;
            double cumWeight = 0;
            int selected = 0;

            for (int i = 0; i < n; i++)
            {
                cumWeight += distances[i];
                if (cumWeight >= threshold)
                {
                    selected = i;
                    break;
                }
            }

            centers[c] = new T[d];
            for (int j = 0; j < d; j++)
            {
                centers[c][j] = x[selected, j];
            }
        }

        return centers;
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

        int n = x.Rows;
        int d = NumFeatures;
        int k = NumClusters;
        int batchSize = Math.Min(_options.BatchSize, n);

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        if (ClusterCenters is null || _centerCounts is null) return;

        // Get current centers
        var centers = new T[k][];
        for (int c = 0; c < k; c++)
        {
            centers[c] = new T[d];
            for (int j = 0; j < d; j++)
            {
                centers[c][j] = ClusterCenters[c, j];
            }
        }

        // Process in batches
        for (int start = 0; start < n; start += batchSize)
        {
            int end = Math.Min(start + batchSize, n);

            for (int i = start; i < end; i++)
            {
                // Find nearest center
                T minDist = NumOps.MaxValue;
                int bestCluster = 0;

                for (int c = 0; c < k; c++)
                {
                    T dist = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        T diff = NumOps.Subtract(x[i, j], centers[c][j]);
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }

                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                // Update center
                _centerCounts[bestCluster]++;
                T eta = NumOps.Divide(NumOps.One, NumOps.FromDouble(_centerCounts[bestCluster]));
                T oneMinusEta = NumOps.Subtract(NumOps.One, eta);

                for (int j = 0; j < d; j++)
                {
                    centers[bestCluster][j] = NumOps.Add(
                        NumOps.Multiply(oneMinusEta, centers[bestCluster][j]),
                        NumOps.Multiply(eta, x[i, j]));
                }
            }
        }

        // Update stored centers
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = centers[c][j];
            }
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        int d = NumFeatures;
        var labels = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            if (ClusterCenters is not null)
            {
                for (int c = 0; c < NumClusters; c++)
                {
                    T dist = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        T diff = NumOps.Subtract(x[i, j], ClusterCenters[c, j]);
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }

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
        return Labels ?? new Vector<T>(0);
    }
}
