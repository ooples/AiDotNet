using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Angle-Based Outlier Detection in High-dimensional Data", "https://doi.org/10.1145/1401890.1401946", Year = 2008, Authors = "Hans-Peter Kriegel, Matthias Schubert, Arthur Zimek")]
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
        int d = X.Columns;

        for (int p = 0; p < X.Rows; p++)
        {
            // Extract query point row as Vector<T> for vectorized ops
            var pointP = new Vector<T>(X.GetRowReadOnlySpan(p).ToArray());

            // Compute ABOF (Angle-Based Outlier Factor)
            T abof = ComputeABOF(pointP, trainingData);

            // ABOF is inversely related to outlierness — invert for consistent scoring
            scores[p] = NumOps.GreaterThan(abof, NumOps.Zero)
                ? NumOps.Divide(NumOps.One, abof)
                : NumOps.MaxValue;
        }

        return scores;
    }

    private T ComputeABOF(Vector<T> point, Matrix<T> trainingData)
    {
        int n = trainingData.Rows;
        int d = trainingData.Columns;
        T epsilon = NumOps.FromDouble(1e-10);

        var weightedAngles = new List<T>();
        var weights = new List<T>();

        for (int i = 0; i < n; i++)
        {
            // Extract row as Vector<T> and compute PA = A - P via IEngine
            var pointA = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());
            var pa = Engine.Subtract(pointA, point);

            // Squared norm via dot product (SIMD-accelerated)
            T paNormSq = Engine.DotProduct(pa, pa);
            T paNorm = NumOps.Sqrt(paNormSq);
            if (NumOps.LessThan(paNorm, epsilon)) continue;

            for (int k = i + 1; k < n; k++)
            {
                var pointB = new Vector<T>(trainingData.GetRowReadOnlySpan(k).ToArray());
                var pb = Engine.Subtract(pointB, point);

                T pbNormSq = Engine.DotProduct(pb, pb);
                T pbNorm = NumOps.Sqrt(pbNormSq);
                if (NumOps.LessThan(pbNorm, epsilon)) continue;

                // cos(angle) = (PA · PB) / (|PA| * |PB|) via vectorized dot product
                T dot = Engine.DotProduct(pa, pb);
                T cosAngle = NumOps.Divide(dot, NumOps.Multiply(paNorm, pbNorm));

                // Clamp for numerical stability
                T negOne = NumOps.FromDouble(-1.0);
                if (NumOps.LessThan(cosAngle, negOne)) cosAngle = negOne;
                if (NumOps.GreaterThan(cosAngle, NumOps.One)) cosAngle = NumOps.One;

                // Weight = 1 / (|PA|² * |PB|²)
                T weight = NumOps.Divide(NumOps.One, NumOps.Multiply(paNormSq, pbNormSq));

                weightedAngles.Add(NumOps.Multiply(cosAngle, weight));
                weights.Add(weight);
            }
        }

        if (weights.Count == 0) return NumOps.MaxValue;

        // Weighted variance of angles
        T sumWeights = NumOps.Zero;
        T sumWeightedAngles = NumOps.Zero;
        foreach (var w in weights) sumWeights = NumOps.Add(sumWeights, w);
        foreach (var wa in weightedAngles) sumWeightedAngles = NumOps.Add(sumWeightedAngles, wa);

        if (NumOps.LessThan(sumWeights, epsilon)) return NumOps.MaxValue;

        T mean = NumOps.Divide(sumWeightedAngles, sumWeights);

        T variance = NumOps.Zero;
        for (int i = 0; i < weightedAngles.Count; i++)
        {
            T angle = NumOps.Divide(weightedAngles[i], weights[i]);
            T diff = NumOps.Subtract(angle, mean);
            variance = NumOps.Add(variance,
                NumOps.Multiply(weights[i], NumOps.Multiply(diff, diff)));
        }

        return NumOps.Divide(variance, sumWeights);
    }
}
