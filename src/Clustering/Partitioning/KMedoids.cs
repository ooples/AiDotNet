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
    protected override ModelType GetModelType() => ModelType.Clustering;

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
        double totalInertia = 0;

        for (int i = 0; i < n; i++)
        {
            int nearestMedoid = 0;
            double minDist = double.MaxValue;

            for (int m = 0; m < k; m++)
            {
                double dist = distMatrix[i, medoids[m]];
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestMedoid = m;
                }
            }

            Labels[i] = NumOps.FromDouble(nearestMedoid);
            totalInertia += minDist;
        }

        Inertia = NumOps.FromDouble(totalInertia);

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

    private double[,] ComputeDistanceMatrix(Matrix<T> x, int n, IDistanceMetric<T> metric)
    {
        var distMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            var pointI = GetRow(x, i);
            for (int j = i + 1; j < n; j++)
            {
                var pointJ = GetRow(x, j);
                double dist = NumOps.ToDouble(metric.Compute(pointI, pointJ));
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

    private int[] InitializeKMedoidsPlusPlus(double[,] distMatrix, int n, int k, Random rand)
    {
        var medoids = new List<int>();

        // First medoid: random
        medoids.Add(rand.Next(n));

        // Remaining medoids: probabilistic based on distance
        for (int m = 1; m < k; m++)
        {
            var minDistances = new double[n];
            double totalDist = 0;

            for (int i = 0; i < n; i++)
            {
                double minDist = double.MaxValue;
                foreach (int med in medoids)
                {
                    minDist = Math.Min(minDist, distMatrix[i, med]);
                }
                minDistances[i] = minDist * minDist; // Square for probability
                totalDist += minDistances[i];
            }

            // Select next medoid proportional to squared distance
            double target = rand.NextDouble() * totalDist;
            double cumulative = 0;
            int selected = 0;

            for (int i = 0; i < n; i++)
            {
                if (medoids.Contains(i)) continue;
                cumulative += minDistances[i];
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

    private int[] InitializeBuild(double[,] distMatrix, int n, int k)
    {
        var medoids = new List<int>();

        // First medoid: point with minimum total distance to all others
        int firstMedoid = 0;
        double minTotalDist = double.MaxValue;

        for (int i = 0; i < n; i++)
        {
            double totalDist = 0;
            for (int j = 0; j < n; j++)
            {
                totalDist += distMatrix[i, j];
            }

            if (totalDist < minTotalDist)
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
            double bestGain = double.NegativeInfinity;

            for (int i = 0; i < n; i++)
            {
                if (medoids.Contains(i)) continue;

                double gain = 0;
                for (int j = 0; j < n; j++)
                {
                    // Current nearest medoid distance
                    double currentDist = double.MaxValue;
                    foreach (int med in medoids)
                    {
                        currentDist = Math.Min(currentDist, distMatrix[j, med]);
                    }

                    // Distance to candidate
                    double candidateDist = distMatrix[j, i];

                    // Gain is reduction in distance
                    gain += Math.Max(0, currentDist - candidateDist);
                }

                if (gain > bestGain)
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

    private int[] RunPAM(double[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        double currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);

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
                    double newCost = ComputeTotalCost(distMatrix, currentMedoids, n);

                    if (newCost < currentCost - Options.Tolerance)
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

    private int[] RunFastPAM(double[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();

        // Precompute nearest and second-nearest medoid for each point
        var nearest = new int[n];
        var nearestDist = new double[n];
        var secondDist = new double[n];

        UpdateNearestMedoids(distMatrix, currentMedoids, n, k, nearest, nearestDist, secondDist);

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            int bestM = -1;
            int bestI = -1;
            double bestDelta = 0;

            // For each medoid
            for (int m = 0; m < k; m++)
            {
                int oldMedoid = currentMedoids[m];

                // For each non-medoid
                for (int i = 0; i < n; i++)
                {
                    if (Array.IndexOf(currentMedoids, i) >= 0) continue;

                    // Compute swap cost delta
                    double delta = 0;

                    for (int j = 0; j < n; j++)
                    {
                        double distToNew = distMatrix[j, i];

                        if (nearest[j] == m)
                        {
                            // Currently assigned to medoid being swapped
                            double newDist = Math.Min(distToNew, secondDist[j]);
                            delta += newDist - nearestDist[j];
                        }
                        else if (distToNew < nearestDist[j])
                        {
                            // Would be reassigned to new medoid
                            delta += distToNew - nearestDist[j];
                        }
                    }

                    if (delta < bestDelta)
                    {
                        bestDelta = delta;
                        bestM = m;
                        bestI = i;
                    }
                }
            }

            if (bestM < 0 || bestDelta >= -Options.Tolerance)
            {
                break;
            }

            // Perform best swap
            currentMedoids[bestM] = bestI;
            UpdateNearestMedoids(distMatrix, currentMedoids, n, k, nearest, nearestDist, secondDist);
        }

        return currentMedoids;
    }

    private int[] RunAlternate(double[,] distMatrix, int[] medoids, int n, int k)
    {
        var currentMedoids = (int[])medoids.Clone();
        var labels = new int[n];

        for (int iter = 0; iter < Options.MaxIterations; iter++)
        {
            // Assign step
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearest = 0;
                double minDist = double.MaxValue;

                for (int m = 0; m < k; m++)
                {
                    double dist = distMatrix[i, currentMedoids[m]];
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearest = m;
                    }
                }

                if (labels[i] != nearest)
                {
                    labels[i] = nearest;
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
                double bestCost = double.MaxValue;

                foreach (int candidate in clusterPoints)
                {
                    double cost = 0;
                    foreach (int other in clusterPoints)
                    {
                        cost += distMatrix[candidate, other];
                    }

                    if (cost < bestCost)
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

    private void UpdateNearestMedoids(double[,] distMatrix, int[] medoids, int n, int k,
        int[] nearest, double[] nearestDist, double[] secondDist)
    {
        for (int i = 0; i < n; i++)
        {
            nearest[i] = 0;
            nearestDist[i] = double.MaxValue;
            secondDist[i] = double.MaxValue;

            for (int m = 0; m < k; m++)
            {
                double dist = distMatrix[i, medoids[m]];
                if (dist < nearestDist[i])
                {
                    secondDist[i] = nearestDist[i];
                    nearestDist[i] = dist;
                    nearest[i] = m;
                }
                else if (dist < secondDist[i])
                {
                    secondDist[i] = dist;
                }
            }
        }
    }

    private double ComputeTotalCost(double[,] distMatrix, int[] medoids, int n)
    {
        double totalCost = 0;

        for (int i = 0; i < n; i++)
        {
            double minDist = double.MaxValue;
            foreach (int m in medoids)
            {
                minDist = Math.Min(minDist, distMatrix[i, m]);
            }
            totalCost += minDist;
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
