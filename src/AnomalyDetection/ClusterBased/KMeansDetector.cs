using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using K-Means clustering distance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> K-Means clustering groups data into k clusters based on distance
/// to cluster centers (centroids). Points far from their nearest centroid are considered
/// anomalies because they don't fit well into any cluster.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Partition data into k clusters using K-Means
/// 2. For each point, find distance to nearest centroid
/// 3. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When data naturally forms spherical clusters
/// - When you have a rough idea of the number of clusters
/// - Fast and scalable to large datasets
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (clusters): 8 (or estimated from data)
/// - Max iterations: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class KMeansDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private readonly int _maxIterations;
    private Matrix<T>? _centroids;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// </summary>
    public int MaxIterations => _maxIterations;

    /// <summary>
    /// Creates a new K-Means anomaly detector.
    /// </summary>
    /// <param name="k">Number of clusters. Default is 8.</param>
    /// <param name="maxIterations">Maximum iterations for K-Means. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public KMeansDetector(int k = 8, int maxIterations = 100, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended value is 8 or use the elbow method.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _k = k;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < _k)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be at least k ({_k}).",
                nameof(X));
        }

        // Run K-Means clustering
        _centroids = RunKMeans(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Find distance to nearest centroid
            double minDist = double.MaxValue;

            var point = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = X[i, j];
            }

            var centroids = _centroids;
            if (centroids == null)
            {
                throw new InvalidOperationException("Model not properly fitted.");
            }

            for (int c = 0; c < centroids.Rows; c++)
            {
                var centroid = new Vector<T>(centroids.Columns);
                for (int j = 0; j < centroids.Columns; j++)
                {
                    centroid[j] = centroids[c, j];
                }

                double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                if (dist < minDist)
                {
                    minDist = dist;
                }
            }

            scores[i] = NumOps.FromDouble(minDist);
        }

        return scores;
    }

    private Matrix<T> RunKMeans(Matrix<T> X)
    {
        var random = new Random(_randomSeed);
        int n = X.Rows;
        int d = X.Columns;

        // Initialize centroids using K-Means++
        var centroids = InitializeCentroids(X, random);

        // Assignment array
        var assignments = new int[n];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assignment step: assign each point to nearest centroid
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearestCentroid = 0;
                double minDist = double.MaxValue;

                var point = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    point[j] = X[i, j];
                }

                for (int c = 0; c < _k; c++)
                {
                    var centroid = new Vector<T>(d);
                    for (int j = 0; j < d; j++)
                    {
                        centroid[j] = centroids[c, j];
                    }

                    double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCentroid = c;
                    }
                }

                if (assignments[i] != nearestCentroid)
                {
                    assignments[i] = nearestCentroid;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update step: recalculate centroids
            var newCentroids = new Matrix<T>(_k, d);
            var counts = new int[_k];

            for (int i = 0; i < n; i++)
            {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++)
                {
                    newCentroids[cluster, j] = NumOps.Add(newCentroids[cluster, j], X[i, j]);
                }
            }

            for (int c = 0; c < _k; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        newCentroids[c, j] = NumOps.Divide(newCentroids[c, j], NumOps.FromDouble(counts[c]));
                    }
                }
                else
                {
                    // Re-initialize empty cluster
                    int randomIdx = random.Next(n);
                    for (int j = 0; j < d; j++)
                    {
                        newCentroids[c, j] = X[randomIdx, j];
                    }
                }
            }

            centroids = newCentroids;
        }

        return centroids;
    }

    private Matrix<T> InitializeCentroids(Matrix<T> X, Random random)
    {
        // K-Means++ initialization
        int n = X.Rows;
        int d = X.Columns;
        var centroids = new Matrix<T>(_k, d);

        // Choose first centroid randomly
        int firstIdx = random.Next(n);
        for (int j = 0; j < d; j++)
        {
            centroids[0, j] = X[firstIdx, j];
        }

        // Choose remaining centroids
        for (int c = 1; c < _k; c++)
        {
            // Calculate distance to nearest centroid for each point
            var distances = new double[n];
            double totalDist = 0;

            for (int i = 0; i < n; i++)
            {
                double minDist = double.MaxValue;
                var point = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    point[j] = X[i, j];
                }

                for (int prevC = 0; prevC < c; prevC++)
                {
                    var centroid = new Vector<T>(d);
                    for (int j = 0; j < d; j++)
                    {
                        centroid[j] = centroids[prevC, j];
                    }

                    double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, centroid));
                    if (dist < minDist)
                    {
                        minDist = dist;
                    }
                }

                distances[i] = minDist * minDist;
                totalDist += distances[i];
            }

            // Choose next centroid with probability proportional to distance squared
            double threshold = random.NextDouble() * totalDist;
            double cumulative = 0;
            int selectedIdx = 0;

            for (int i = 0; i < n; i++)
            {
                cumulative += distances[i];
                if (cumulative >= threshold)
                {
                    selectedIdx = i;
                    break;
                }
            }

            for (int j = 0; j < d; j++)
            {
                centroids[c, j] = X[selectedIdx, j];
            }
        }

        return centroids;
    }
}
