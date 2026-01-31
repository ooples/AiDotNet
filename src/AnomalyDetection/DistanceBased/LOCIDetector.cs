using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Local Correlation Integral (LOCI).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LOCI is a density-based outlier detection method that automatically
/// determines the appropriate neighborhood size. It computes a Multi-Granularity Deviation
/// Factor (MDEF) that indicates how much a point's local density deviates from its
/// neighborhood's density.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For multiple radius values, compute local density
/// 2. Compare point's density to average density of neighbors (MDEF)
/// 3. Flag points with MDEF exceeding a threshold based on standard deviation
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you don't want to manually tune neighborhood size
/// - Data has varying local densities
/// - You want automatic threshold determination
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha: 0.5 (neighborhood multiplier)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Papadimitriou, S., et al. (2003). "LOCI: Fast Outlier Detection Using
/// the Local Correlation Integral." ICDE.
/// </para>
/// </remarks>
public class LOCIDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private readonly int _kMax;
    private Matrix<T>? _trainingData;
    private double _maxRadius;

    /// <summary>
    /// Gets the alpha parameter (sampling neighborhood ratio).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of neighbors to consider.
    /// </summary>
    public int KMax => _kMax;

    /// <summary>
    /// Creates a new LOCI anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// Neighborhood multiplier. The sampling neighborhood radius is alpha times the counting radius.
    /// Default is 0.5. Must be between 0 and 1.
    /// </param>
    /// <param name="kMax">Maximum number of neighbors to consider. Default is 20.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LOCIDetector(double alpha = 0.5, int kMax = 20, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (alpha <= 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 (exclusive) and 1 (inclusive). Recommended value is 0.5.");
        }

        if (kMax < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kMax),
                "KMax must be at least 1. Recommended value is 20.");
        }

        _alpha = alpha;
        _kMax = kMax;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < 3)
        {
            throw new ArgumentException(
                "LOCI requires at least 3 samples.",
                nameof(X));
        }

        _trainingData = X;

        // Estimate max radius from data
        _maxRadius = EstimateMaxRadius(X);

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
            double maxMdef = 0;

            // Test multiple radii
            int numRadii = 10;
            for (int ri = 1; ri <= numRadii; ri++)
            {
                double r = (_maxRadius * ri) / numRadii;
                double alphaR = _alpha * r;

                // Count points in r-neighborhood (counting neighborhood)
                int nR = CountPointsInRadius(X, i, _trainingData!, r);

                if (nR < 2) continue;

                // Get points in alpha*r neighborhood (sampling neighborhood)
                var samplingNeighbors = GetPointsInRadius(X, i, _trainingData!, alphaR);

                if (samplingNeighbors.Count == 0) continue;

                // Compute average n(r) of neighbors in sampling neighborhood
                double avgNR = 0;
                double varNR = 0;

                foreach (int neighborIdx in samplingNeighbors)
                {
                    int neighborNR = CountPointsInRadiusFromTraining(neighborIdx, r);
                    avgNR += neighborNR;
                }
                avgNR /= samplingNeighbors.Count;

                // Compute variance
                foreach (int neighborIdx in samplingNeighbors)
                {
                    int neighborNR = CountPointsInRadiusFromTraining(neighborIdx, r);
                    varNR += (neighborNR - avgNR) * (neighborNR - avgNR);
                }
                varNR /= samplingNeighbors.Count;

                // MDEF = 1 - n(p,r) / avg_n(r)
                double mdef = avgNR > 0 ? 1.0 - (nR / avgNR) : 0;

                // Sigma-MDEF (normalized deviation)
                double sigmaMdef = avgNR > 0 ? Math.Sqrt(varNR) / avgNR : 0;

                // Final score: MDEF / sigma-MDEF (or just MDEF if sigma is 0)
                double lociScore = sigmaMdef > 0 ? Math.Abs(mdef) / sigmaMdef : Math.Abs(mdef);

                if (lociScore > maxMdef)
                {
                    maxMdef = lociScore;
                }
            }

            scores[i] = NumOps.FromDouble(maxMdef);
        }

        return scores;
    }

    private double EstimateMaxRadius(Matrix<T> X)
    {
        // Estimate maximum radius as the max distance between any two points
        // For efficiency, sample a subset
        double maxDist = 0;
        int sampleSize = Math.Min(100, X.Rows);
        var random = new Random(_randomSeed);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx1 = random.Next(X.Rows);
            int idx2 = random.Next(X.Rows);

            var p1 = new Vector<T>(X.Columns);
            var p2 = new Vector<T>(X.Columns);

            for (int j = 0; j < X.Columns; j++)
            {
                p1[j] = X[idx1, j];
                p2[j] = X[idx2, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(p1, p2));
            if (dist > maxDist) maxDist = dist;
        }

        return maxDist;
    }

    private int CountPointsInRadius(Matrix<T> X, int pointIdx, Matrix<T> reference, double radius)
    {
        int count = 0;
        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        for (int i = 0; i < reference.Rows; i++)
        {
            var refPoint = new Vector<T>(reference.Columns);
            for (int j = 0; j < reference.Columns; j++)
            {
                refPoint[j] = reference[i, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint));
            if (dist <= radius) count++;
        }

        return count;
    }

    private int CountPointsInRadiusFromTraining(int trainingIdx, double radius)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int count = 0;
        var point = new Vector<T>(trainingData.Columns);
        for (int j = 0; j < trainingData.Columns; j++)
        {
            point[j] = trainingData[trainingIdx, j];
        }

        for (int i = 0; i < trainingData.Rows; i++)
        {
            var refPoint = new Vector<T>(trainingData.Columns);
            for (int j = 0; j < trainingData.Columns; j++)
            {
                refPoint[j] = trainingData[i, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint));
            if (dist <= radius) count++;
        }

        return count;
    }

    private List<int> GetPointsInRadius(Matrix<T> X, int pointIdx, Matrix<T> reference, double radius)
    {
        var neighbors = new List<int>();
        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        for (int i = 0; i < reference.Rows; i++)
        {
            var refPoint = new Vector<T>(reference.Columns);
            for (int j = 0; j < reference.Columns; j++)
            {
                refPoint[j] = reference[i, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint));
            if (dist <= radius && dist > 0)
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }
}
