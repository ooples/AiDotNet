using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.AngleBased;

/// <summary>
/// Detects anomalies using Fast Angle-Based Outlier Detection (FastABOD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FastABOD is an optimized version of ABOD that uses only
/// k-nearest neighbors instead of all points. This makes it much faster while
/// maintaining good detection quality.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point p, find k nearest neighbors
/// 2. Compute angles only between neighbor pairs
/// 3. Calculate angle variance as outlier score
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Large datasets where full ABOD is too slow
/// - High-dimensional data
/// - When you need faster runtime with slight accuracy trade-off
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kriegel, H., et al. (2008). "Angle-Based Outlier Detection in High-dimensional Data." KDD.
/// </para>
/// </remarks>
public class FastABODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Creates a new FastABOD anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public FastABODDetector(int k = 10, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 2 for angle computation. Recommended is 10.");
        }

        _k = k;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= _k)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than k ({_k}).",
                nameof(X));
        }

        _trainingData = X;

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

        for (int p = 0; p < X.Rows; p++)
        {
            // Get query point
            var pointP = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                pointP[j] = NumOps.ToDouble(X[p, j]);
            }

            // Find k-nearest neighbors
            var neighbors = GetKNearestNeighbors(pointP);

            // Compute ABOF using only neighbors
            double abof = ComputeABOF(pointP, neighbors);

            // ABOF is inversely related to outlierness
            double score = abof > 0 ? 1.0 / abof : double.MaxValue;
            scores[p] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private List<(int Index, double[] Point, double Distance)> GetKNearestNeighbors(double[] queryPoint)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int d = trainingData.Columns;
        var distances = new List<(int Index, double[] Point, double Distance)>();

        for (int i = 0; i < trainingData.Rows; i++)
        {
            var point = new double[d];
            double dist = 0;

            for (int j = 0; j < d; j++)
            {
                point[j] = NumOps.ToDouble(trainingData[i, j]);
                double diff = point[j] - queryPoint[j];
                dist += diff * diff;
            }

            dist = Math.Sqrt(dist);

            if (dist > 1e-10) // Exclude the point itself
            {
                distances.Add((i, point, dist));
            }
        }

        return distances
            .OrderBy(x => x.Distance)
            .Take(_k)
            .ToList();
    }

    private double ComputeABOF(double[] point, List<(int Index, double[] Point, double Distance)> neighbors)
    {
        if (neighbors.Count < 2) return double.MaxValue;

        int d = point.Length;
        var weightedAngles = new List<double>();
        var weights = new List<double>();

        // Compute angles between all pairs of neighbors
        for (int i = 0; i < neighbors.Count; i++)
        {
            var pointA = neighbors[i].Point;
            double distA = neighbors[i].Distance;

            // Vector PA = A - P
            var pa = new double[d];
            for (int j = 0; j < d; j++)
            {
                pa[j] = pointA[j] - point[j];
            }

            for (int k = i + 1; k < neighbors.Count; k++)
            {
                var pointB = neighbors[k].Point;
                double distB = neighbors[k].Distance;

                // Vector PB = B - P
                var pb = new double[d];
                for (int j = 0; j < d; j++)
                {
                    pb[j] = pointB[j] - point[j];
                }

                // Compute angle using dot product
                double dot = 0;
                for (int j = 0; j < d; j++)
                {
                    dot += pa[j] * pb[j];
                }

                double cosAngle = dot / (distA * distB);
                cosAngle = Math.Max(-1, Math.Min(1, cosAngle));

                // Weight by inverse of product of distances squared
                double weight = 1.0 / (distA * distA * distB * distB);

                weightedAngles.Add(cosAngle * weight);
                weights.Add(weight);
            }
        }

        if (weights.Count == 0) return double.MaxValue;

        // Compute weighted variance
        // Formula: mean = Σ(w_i * v_i) / Σ(w_i)
        //          variance = Σ(w_i * (v_i - mean)²) / Σ(w_i)
        double sumWeights = weights.Sum();
        if (sumWeights < 1e-10) return double.MaxValue;

        // weightedAngles[i] = cosAngle_i * weight_i, so weighted mean is correct
        double mean = weightedAngles.Sum() / sumWeights;

        double variance = 0;
        for (int i = 0; i < weightedAngles.Count; i++)
        {
            // Extract the original angle from the weighted angle
            double angle = weightedAngles[i] / weights[i];
            double diff = angle - mean;
            variance += weights[i] * diff * diff;
        }
        variance /= sumWeights;

        return variance;
    }
}
