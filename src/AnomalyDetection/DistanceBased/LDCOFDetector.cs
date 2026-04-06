using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LDCOF combines clustering with density-based outlier detection.
/// It first clusters the data, then computes outlier scores based on how a point's density
/// compares to its cluster's density. Points in sparse regions of dense clusters are flagged.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Cluster the data using k-means
/// 2. Compute local density for each point
/// 3. Compare point density to cluster average density
/// 4. Large deviations indicate outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When data has natural cluster structure
/// - For detecting local outliers within clusters
/// - When global methods miss cluster-specific anomalies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of clusters: 8
/// - Number of neighbors (k): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Amer, M., Goldstein, M. (2012). "Nearest-Neighbor and Clustering based
/// Anomaly Detection Algorithms for RapidMiner." Workshop on Open Source Data Mining Software.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Enhancing the Effectiveness of Clustering-Based Outlier Detection", "https://goldstein.center/publications.html", Year = 2012, Authors = "Mennatallah Amer, Markus Goldstein")]
public class LDCOFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _numClusters;
    private readonly int _numNeighbors;
    private Matrix<T>? _trainingData;
    private Matrix<T>? _clusterCenters;
    private int[]? _clusterAssignments;
    private Vector<T>? _clusterDensities;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int NumClusters => _numClusters;

    /// <summary>
    /// Gets the number of neighbors for density estimation.
    /// </summary>
    public int NumNeighbors => _numNeighbors;

    /// <summary>
    /// Creates a new LDCOF anomaly detector.
    /// </summary>
    /// <param name="numClusters">Number of clusters. Default is 8.</param>
    /// <param name="numNeighbors">Number of neighbors for density. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LDCOFDetector(int numClusters = 8, int numNeighbors = 10,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numClusters < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClusters),
                "NumClusters must be at least 1. Recommended is 8.");
        }

        if (numNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "NumNeighbors must be at least 1. Recommended is 10.");
        }

        _numClusters = numClusters;
        _numNeighbors = numNeighbors;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        _trainingData = X;

        // Run k-means clustering
        int effectiveClusters = Math.Min(_numClusters, n);
        (_clusterCenters, _clusterAssignments) = KMeansClustering(X, effectiveClusters);

        // Compute cluster densities
        _clusterDensities = ComputeClusterDensities();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> centers, int[] assignments) KMeansClustering(Matrix<T> data, int k)
    {
        int n = data.Rows;
        int d = data.Columns;
        var random = RandomHelper.CreateSeededRandom(_randomSeed);

        // Initialize centers randomly
        var centers = new Matrix<T>(k, d);
        var indices = Enumerable.Range(0, n).OrderBy(_ => random.NextDouble()).Take(k).ToArray();
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < d; j++)
            {
                centers[i, j] = data[indices[i], j];
            }
        }

        var assignments = new int[n];
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assign points to nearest center
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                T minDist = NumOps.MaxValue;
                int bestCluster = 0;

                var point = new Vector<T>(data.GetRowReadOnlySpan(i).ToArray());

                for (int c = 0; c < k; c++)
                {
                    var centroid = new Vector<T>(centers.GetRowReadOnlySpan(c).ToArray());
                    var diff = Engine.Subtract(point, centroid);
                    T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
                    if (NumOps.LessThan(dist, minDist))
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                if (assignments[i] != bestCluster)
                {
                    assignments[i] = bestCluster;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update centers
            var counts = new int[k];
            var newCenters = new Matrix<T>(k, d);

            for (int i = 0; i < n; i++)
            {
                int c = assignments[i];
                counts[c]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[c, j] = NumOps.Add(newCenters[c, j], data[i, j]);
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        newCenters[c, j] = NumOps.Divide(newCenters[c, j], NumOps.FromDouble(counts[c]));
                    }
                }
                else
                {
                    // Re-initialize empty cluster
                    int randomIdx = random.Next(n);
                    for (int j = 0; j < d; j++)
                    {
                        newCenters[c, j] = data[randomIdx, j];
                    }
                }
            }

            centers = newCenters;
        }

        return (centers, assignments);
    }

    private Vector<T> ComputeClusterDensities()
    {
        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterAssignments = _clusterAssignments;
        if (clusterCenters == null || trainingData == null || clusterAssignments == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int k = clusterCenters.Rows;
        var densities = new Vector<T>(k);

        for (int c = 0; c < k; c++)
        {
            // Get indices of points in this cluster
            var clusterIndices = Enumerable.Range(0, trainingData.Rows)
                .Where(i => clusterAssignments[i] == c)
                .ToArray();

            if (clusterIndices.Length == 0)
            {
                densities[c] = NumOps.One;
                continue;
            }

            // Compute average k-distance within cluster
            T totalKDist = NumOps.Zero;
            int effectiveK = Math.Min(_numNeighbors, clusterIndices.Length - 1);

            foreach (int pIdx in clusterIndices)
            {
                var point = new Vector<T>(trainingData.GetRowReadOnlySpan(pIdx).ToArray());

                var distances = clusterIndices
                    .Where(qIdx => qIdx != pIdx)
                    .Select(qIdx =>
                    {
                        var other = new Vector<T>(trainingData.GetRowReadOnlySpan(qIdx).ToArray());
                        var diff = Engine.Subtract(point, other);
                        return NumOps.Sqrt(Engine.DotProduct(diff, diff));
                    })
                    .OrderBy(d => NumOps.ToDouble(d))
                    .ToArray();

                if (distances.Length > 0 && effectiveK > 0)
                {
                    totalKDist = NumOps.Add(totalKDist, distances[Math.Min(effectiveK - 1, distances.Length - 1)]);
                }
            }

            // Average k-distance (inverse is density)
            T avgKDist = clusterIndices.Length > 0
                ? NumOps.Divide(totalKDist, NumOps.FromDouble(clusterIndices.Length))
                : NumOps.One;

            T eps = NumOps.FromDouble(1e-10);
            densities[c] = NumOps.Divide(NumOps.One, NumOps.Add(avgKDist, eps));
        }

        return densities;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterDensities = _clusterDensities;
        if (clusterCenters == null || trainingData == null || clusterDensities == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int effectiveK = Math.Min(_numNeighbors, trainingData.Rows - 1);
        T eps = NumOps.FromDouble(1e-10);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Find nearest cluster
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;
            for (int c = 0; c < clusterCenters.Rows; c++)
            {
                var centroid = new Vector<T>(clusterCenters.GetRowReadOnlySpan(c).ToArray());
                var diff = Engine.Subtract(point, centroid);
                T dist = NumOps.Sqrt(Engine.DotProduct(diff, diff));
                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
                    nearestCluster = c;
                }
            }

            // Compute local density (k-distance to training points)
            var distances = Enumerable.Range(0, trainingData.Rows)
                .Select(ti =>
                {
                    var trainPoint = new Vector<T>(trainingData.GetRowReadOnlySpan(ti).ToArray());
                    var diff = Engine.Subtract(point, trainPoint);
                    return NumOps.Sqrt(Engine.DotProduct(diff, diff));
                })
                .OrderBy(d => NumOps.ToDouble(d))
                .ToArray();

            T kDist = effectiveK > 0 && effectiveK <= distances.Length
                ? distances[effectiveK - 1]
                : (distances.Length > 0 ? distances[distances.Length - 1] : NumOps.Zero);
            T localDensity = NumOps.Divide(NumOps.One, NumOps.Add(kDist, eps));

            // LDCOF score: ratio of cluster density to local density
            T score = NumOps.Divide(clusterDensities[nearestCluster], NumOps.Add(localDensity, eps));

            scores[i] = score;
        }

        return scores;
    }
}
