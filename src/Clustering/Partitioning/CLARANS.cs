using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

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
public class CLARANS<T> : ClusteringBase<T>
{
    private readonly CLARANSOptions<T> _options;
    private int[]? _medoidIndices;
    private double _bestCost;

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
    public double BestCost => _bestCost;

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CLARANS<T>(new CLARANSOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxNeighbor = _options.MaxNeighbor,
            NumLocal = _options.NumLocal,
            DistanceMetric = _options.DistanceMetric
        });
    }

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

        _bestCost = double.MaxValue;
        int[]? bestMedoids = null;

        // Run numLocal iterations
        for (int iter = 0; iter < _options.NumLocal; iter++)
        {
            // Initialize random medoids
            var currentMedoids = SelectRandomMedoids(n, k, rand);
            double currentCost = ComputeTotalCost(distMatrix, currentMedoids, n);

            // Local search
            int neighborsExplored = 0;

            while (neighborsExplored < maxNeighbor)
            {
                // Select random medoid to swap
                int medoidIdx = rand.Next(k);
                int oldMedoid = currentMedoids[medoidIdx];

                // Select random non-medoid
                int newMedoid;
                do
                {
                    newMedoid = rand.Next(n);
                } while (currentMedoids.Contains(newMedoid));

                // Compute cost of swap
                var newMedoids = (int[])currentMedoids.Clone();
                newMedoids[medoidIdx] = newMedoid;
                double newCost = ComputeTotalCost(distMatrix, newMedoids, n);

                if (newCost < currentCost)
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
            if (currentCost < _bestCost)
            {
                _bestCost = currentCost;
                bestMedoids = (int[])currentMedoids.Clone();
            }
        }

        _medoidIndices = bestMedoids ?? SelectRandomMedoids(n, k, rand);

        // Set cluster centers as medoid points
        ClusterCenters = new Matrix<T>(k, d);
        for (int i = 0; i < k; i++)
        {
            int medoidIdx = _medoidIndices[i];
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[i, j] = x[medoidIdx, j];
            }
        }

        // Assign labels
        Labels = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            int nearestMedoid = 0;
            double minDist = double.MaxValue;

            for (int m = 0; m < k; m++)
            {
                double dist = distMatrix[i, _medoidIndices[m]];
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestMedoid = m;
                }
            }

            Labels[i] = NumOps.FromDouble(nearestMedoid);
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

    private int[] SelectRandomMedoids(int n, int k, Random rand)
    {
        var indices = new HashSet<int>();
        while (indices.Count < k)
        {
            indices.Add(rand.Next(n));
        }
        return indices.ToArray();
    }

    private double ComputeTotalCost(double[,] distMatrix, int[] medoids, int n)
    {
        double totalCost = 0;

        for (int i = 0; i < n; i++)
        {
            double minDist = double.MaxValue;
            foreach (int m in medoids)
            {
                if (distMatrix[i, m] < minDist)
                {
                    minDist = distMatrix[i, m];
                }
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
