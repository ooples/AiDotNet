using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.AngleBased;

/// <summary>
/// Detects anomalies using Angle-Based Outlier Detection (ABOD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ABOD detects outliers by analyzing the angles formed between
/// a point and all pairs of other points. Outliers tend to have angles that point in
/// similar directions (low variance), while inliers have diverse angle patterns.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point p, compute angles between p and all pairs of other points
/// 2. Calculate the variance of these angles (weighted by distance)
/// 3. Low variance indicates the point is an outlier
/// </para>
/// <para>
/// <b>When to use:</b>
/// - High-dimensional data (works better than distance-based methods)
/// - When the "curse of dimensionality" affects other methods
/// - Data has complex structure
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Contamination: 0.1 (10%)
/// - For large datasets, use FastABOD variant
/// </para>
/// <para>
/// Reference: Kriegel, H., et al. (2008). "Angle-Based Outlier Detection in High-dimensional Data." KDD.
/// </para>
/// </remarks>
public class ABODDetector<T> : AnomalyDetectorBase<T>
{
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Creates a new ABOD anomaly detector.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ABODDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < 3)
        {
            throw new ArgumentException(
                "ABOD requires at least 3 samples.",
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

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int n = trainingData.Rows;

        for (int p = 0; p < X.Rows; p++)
        {
            // Get query point
            var pointP = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                pointP[j] = NumOps.ToDouble(X[p, j]);
            }

            // Compute ABOF (Angle-Based Outlier Factor)
            double abof = ComputeABOF(pointP);

            // ABOF is inversely related to outlierness
            // Low ABOF = outlier, High ABOF = inlier
            // We invert it for consistent scoring (high = outlier)
            double score = abof > 0 ? 1.0 / abof : double.MaxValue;
            scores[p] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private double ComputeABOF(double[] point)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int n = trainingData.Rows;
        int d = trainingData.Columns;

        // For each pair of points (a, b), compute angle at point p
        var weightedAngles = new List<double>();
        var weights = new List<double>();

        for (int i = 0; i < n; i++)
        {
            var pointA = new double[d];
            for (int j = 0; j < d; j++)
            {
                pointA[j] = NumOps.ToDouble(trainingData[i, j]);
            }

            // Vector PA = A - P
            var pa = new double[d];
            double paNorm = 0;
            for (int j = 0; j < d; j++)
            {
                pa[j] = pointA[j] - point[j];
                paNorm += pa[j] * pa[j];
            }
            paNorm = Math.Sqrt(paNorm);
            if (paNorm < 1e-10) continue; // Skip if same point

            for (int k = i + 1; k < n; k++)
            {
                var pointB = new double[d];
                for (int j = 0; j < d; j++)
                {
                    pointB[j] = NumOps.ToDouble(trainingData[k, j]);
                }

                // Vector PB = B - P
                var pb = new double[d];
                double pbNorm = 0;
                for (int j = 0; j < d; j++)
                {
                    pb[j] = pointB[j] - point[j];
                    pbNorm += pb[j] * pb[j];
                }
                pbNorm = Math.Sqrt(pbNorm);
                if (pbNorm < 1e-10) continue; // Skip if same point

                // Compute angle using dot product: cos(angle) = (PA . PB) / (|PA| * |PB|)
                double dot = 0;
                for (int j = 0; j < d; j++)
                {
                    dot += pa[j] * pb[j];
                }

                double cosAngle = dot / (paNorm * pbNorm);
                cosAngle = Math.Max(-1, Math.Min(1, cosAngle)); // Clamp for numerical stability

                // Weight by inverse of product of distances squared
                double weight = 1.0 / (paNorm * paNorm * pbNorm * pbNorm);

                weightedAngles.Add(cosAngle * weight);
                weights.Add(weight);
            }
        }

        if (weights.Count == 0) return double.MaxValue;

        // Compute weighted variance of angles
        double sumWeights = weights.Sum();
        if (sumWeights < 1e-10) return double.MaxValue;

        double mean = weightedAngles.Sum() / sumWeights;

        double variance = 0;
        for (int i = 0; i < weightedAngles.Count; i++)
        {
            double diff = (weightedAngles[i] / weights[i]) - (mean / weights.Average());
            variance += weights[i] * diff * diff;
        }
        variance /= sumWeights;

        return variance;
    }
}
