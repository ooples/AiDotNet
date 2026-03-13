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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Finding Groups in Data: An Introduction to Cluster Analysis", "https://doi.org/10.1002/9780470316801", Year = 1990, Authors = "Leonard Kaufman, Peter J. Rousseeuw")]
public class KMedoids<T> : ClusteringBase<T>
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
        : base(options ?? new KMedoidsOptions<T>())
    {
        _options = options ?? new KMedoidsOptions<T>();
    }

    /// <summary>
    /// Gets the indices of medoid points in the original data.
    /// </summary>
    public int[]? MedoidIndices => _medoidIndices;

    /// <inheritdoc />

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new KMedoids<T>(new KMedoidsOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxIterations = _options.MaxIterations,
            Init = _options.Init,
            Algorithm = _options.Algorithm,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
        Labels = new Vector<T>(n);
        T totalInertia = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            int nearestMedoid = 0;
            T minDist = NumOps.MaxValue;

            for (int m = 0; m < k; m++)
            {
                T dist = distMatrix[i, medoids[m]];
                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                    nearestMedoid = m;
                }
            }

            Labels[i] = NumOps.FromDouble(nearestMedoid);
            totalInertia = NumOps.Add(totalInertia, minDist);
        }

        Inertia = totalInertia;

        // Set cluster centers as medoid points
        ClusterCenters = new Matrix<T>(k, d);
        for (int m = 0; m < k; m++)
        {
            int medoidIdx = medoids[m];
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[m, j] = x[medoidIdx, j];
            }
        }

        IsTrained = true;
    }

    private T[,] ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        var distMatrix = new T[n, n];

        for (int i = 0; i < n; i++)
        {
            distMatrix[i, i] = NumOps.Zero;
            var pointI = GetRow(x, i);
            for (int j = i + 1; j < n; j++)
            {
                var pointJ = GetRow(x, j);
                T dist = metric.Compute(pointI, pointJ);
                distMatrix[i, j] = dist;
                distMatrix[j, i] = dist;
            }
        }

        return distMatrix;
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

    private int[] InitializeKMedoidsPlusPlus(T[,] distMatrix, int n, int k, Random rand)
    {
        var medoids = new List<int>();

        // First medoid: random
        medoids.Add(rand.Next(n));

        // Remaining medoids: probabilistic based on distance
        for (int m = 1; m < k; m++)
        {
            var minDistances = new T[n];
            T totalDist = NumOps.Zero;

            for (int i = 0; i < n; i++)
            {
                T minDist = NumOps.MaxValue;
                foreach (int med in medoids)
                {
                    if (NumOps.LessThan(distMatrix[i, med], minDist))
                        minDist = distMatrix[i, med];
                }
                minDistances[i] = NumOps.Multiply(minDist, minDist); // Square for probability
                totalDist = NumOps.Add(totalDist, minDistances[i]);
            }

            // Select next medoid proportional to squared distance
            double target = rand.NextDouble() * NumOps.ToDouble(totalDist);
            double cumulative = 0;
            int selected = 0;

            for (int i = 0; i < n; i++)
            {
                if (medoids.Contains(i)) continue;
                cumulative += NumOps.ToDouble(minDistances[i]);
                if (cumulative >= target)
                {
                    selected = i;
                    break;
                }
            }

            medoids.Add(selected);
        }

        return medoids.ToArray();
    }

    private int[] InitializeBuild(T[,] distMatrix, int n, int k)
    {
        var medoids = new List<int>();

        // First medoid: point with minimum total distance to all others
        int firstMedoid = 0;
        T minTotalDist = NumOps.MaxValue;

        for (int i = 0; i < n; i++)
        {
            T totalDist = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                totalDist = NumOps.Add(totalDist, distMatrix[i, j]);
            }

            if (NumOps.LessThan(totalDist, minTotalDist))
            {
                minTotalDist = totalDist;
                firstMedoid = i;
            }
        }

        medoids.Add(firstMedoid);

        // Remaining medoids: greedily add point that reduces cost most
        while (medoids.Count < k)
        {
            int bestCandidate = -1;
            T bestGain = NumOps.MinValue;

            for (int i = 0; i < n; i++)
            {
                if (medoids.Contains(i)) continue;

                T gain = NumOps.Zero;
                for (int j = 0; j < n; j++)
                {
                    // Current nearest medoid distance
                    T currentDist = NumOps.MaxValue;
                    foreach (int med in medoids)
                    {
                        if (NumOps.LessThan(distMatrix[j, med], currentDist))
                            currentDist = distMatrix[j, med];
                    }

                    // Distance to candidate
                    T candidateDist = distMatrix[j, i];

                    // Gain is reduction in distance
                    T reduction = NumOps.Subtract(currentDist, candidateDist);
                    if (NumOps.GreaterThan(reduction, NumOps.Zero))
                        gain = NumOps.Add(gain, reduction);
                }

                if (NumOps.GreaterThan(gain, bestGain))
                {
                    bestGain = gain;
                    bestCandidate = i;
                }
            }

            if (bestCandidate >= 0)
            {
                medoids.Add(bestCandidate);
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

    private int[] RunPAM(T[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        T currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            bool improved = false;

            // Try all possible swaps
            for (int m = 0; m < k && !improved; m++)
            {
                int oldMedoid = currentMedoids[m];

                for (int i = 0; i < n && !improved; i++)
                {
                    if (Array.IndexOf(currentMedoids, i) >= 0) continue;

                    // Try swapping medoid m with point i
                    currentMedoids[m] = i;
                    T newCost = ComputeTotalCost(distMatrix, currentMedoids, n);

                    if (NumOps.LessThan(newCost, NumOps.Subtract(currentCost, tolerance)))
                    {
                        currentCost = newCost;
                        improved = true;
                    }
                    else
                    {
                        currentMedoids[m] = oldMedoid;
                    }
                }
            }

            if (!improved) break;
        }

        return currentMedoids;
    }

    private int[] RunFastPAM(T[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();

        // Precompute nearest and second-nearest medoid for each point
        var nearest = new int[n];
        var nearestDist = new T[n];
        var secondDist = new T[n];

        UpdateNearestMedoids(distMatrix, currentMedoids, n, k, nearest, nearestDist, secondDist);

        T tolerance = NumOps.FromDouble(Options.Tolerance);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            int bestM = -1;
            int bestI = -1;
            T bestDelta = NumOps.Zero;

            // For each medoid
            for (int m = 0; m < k; m++)
            {
                // For each non-medoid
                for (int i = 0; i < n; i++)
                {
                    if (Array.IndexOf(currentMedoids, i) >= 0) continue;

                    // Compute swap cost delta
                    T delta = NumOps.Zero;

                    for (int j = 0; j < n; j++)
                    {
                        T distToNew = distMatrix[j, i];

                        if (nearest[j] == m)
                        {
                            // Currently assigned to medoid being swapped
                            T newDist = NumOps.LessThan(distToNew, secondDist[j]) ? distToNew : secondDist[j];
                            delta = NumOps.Add(delta, NumOps.Subtract(newDist, nearestDist[j]));
                        }
                        else if (NumOps.LessThan(distToNew, nearestDist[j]))
                        {
                            // Would be reassigned to new medoid
                            delta = NumOps.Add(delta, NumOps.Subtract(distToNew, nearestDist[j]));
                        }
                    }

                    if (NumOps.LessThan(delta, bestDelta))
                    {
                        bestDelta = delta;
                        bestM = m;
                        bestI = i;
                    }
                }
            }

            if (bestM < 0 || !NumOps.LessThan(bestDelta, NumOps.Negate(tolerance)))
            {
                break;
            }

            // Perform best swap
            currentMedoids[bestM] = bestI;
            UpdateNearestMedoids(distMatrix, currentMedoids, n, k, nearest, nearestDist, secondDist);
        }

        return currentMedoids;
    }

    private int[] RunAlternate(T[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        var labels = new int[n];

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Assign step
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearestIdx = 0;
                T minDist = NumOps.MaxValue;

                for (int m = 0; m < k; m++)
                {
                    T dist = distMatrix[i, currentMedoids[m]];
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        nearestIdx = m;
                    }
                }

                if (labels[i] != nearestIdx)
                {
                    labels[i] = nearestIdx;
                    changed = true;
                }
            }

            // Update step: find best medoid for each cluster
            for (int m = 0; m < k; m++)
            {
                var clusterPoints = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if (labels[i] == m)
                    {
                        clusterPoints.Add(i);
                    }
                }

                if (clusterPoints.Count == 0) continue;

                // Find point that minimizes sum of distances to others in cluster
                int bestMedoid = currentMedoids[m];
                T bestCost = NumOps.MaxValue;

                foreach (int candidate in clusterPoints)
                {
                    T cost = NumOps.Zero;
                    foreach (int other in clusterPoints)
                    {
                        cost = NumOps.Add(cost, distMatrix[candidate, other]);
                    }

                    if (NumOps.LessThan(cost, bestCost))
                    {
                        bestCost = cost;
                        bestMedoid = candidate;
                    }
                }

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

    private void UpdateNearestMedoids(T[,] distMatrix, int[] medoids, int n, int k,
        int[] nearest, T[] nearestDist, T[] secondDist)
    {
        for (int i = 0; i < n; i++)
        {
            nearest[i] = 0;
            nearestDist[i] = NumOps.MaxValue;
            secondDist[i] = NumOps.MaxValue;

            for (int m = 0; m < k; m++)
            {
                T dist = distMatrix[i, medoids[m]];
                if (NumOps.LessThan(dist, nearestDist[i]))
                {
                    secondDist[i] = nearestDist[i];
                    nearestDist[i] = dist;
                    nearest[i] = m;
                }
                else if (NumOps.LessThan(dist, secondDist[i]))
                {
                    secondDist[i] = dist;
                }
            }
        }
    }

    private T ComputeTotalCost(T[,] distMatrix, int[] medoids, int n)
    {
        T totalCost = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T minDist = NumOps.MaxValue;
            foreach (int m in medoids)
            {
                if (NumOps.LessThan(distMatrix[i, m], minDist))
                    minDist = distMatrix[i, m];
            }
            totalCost = NumOps.Add(totalCost, minDist);
        }

        return totalCost;
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

        var medoids = new Matrix<T>(_medoidIndices.Length, originalData.Columns);
        for (int i = 0; i < _medoidIndices.Length; i++)
        {
            int idx = _medoidIndices[i];
            for (int j = 0; j < originalData.Columns; j++)
            {
                medoids[i, j] = originalData[idx, j];
            }
        }

        return medoids;
    }
}
