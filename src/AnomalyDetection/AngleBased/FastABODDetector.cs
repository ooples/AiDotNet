using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Angle-Based Outlier Detection in High-dimensional Data", "https://doi.org/10.1145/1401890.1401946", Year = 2008, Authors = "Hans-Peter Kriegel, Matthias Schubert, Arthur Zimek")]
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

        bool isSameAsTraining = ReferenceEquals(X, _trainingData);
        var scores = new Vector<T>(X.Rows);

        for (int p = 0; p < X.Rows; p++)
        {
            var pointP = new Vector<T>(X.GetRowReadOnlySpan(p).ToArray());

            int queryIndex = isSameAsTraining ? p : -1;
            var neighbors = GetKNearestNeighbors(pointP, queryIndex);

            T abof = ComputeABOF(pointP, neighbors);

            scores[p] = NumOps.GreaterThan(abof, NumOps.Zero)
                ? NumOps.Divide(NumOps.One, abof)
                : NumOps.MaxValue;
        }

        return scores;
    }

    private List<(int Index, Vector<T> Point, T Distance)> GetKNearestNeighbors(Vector<T> queryPoint, int queryIndex = -1)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var distances = new List<(int Index, Vector<T> Point, T Distance)>();

        for (int i = 0; i < trainingData.Rows; i++)
        {
            if (i == queryIndex) continue;

            var point = new Vector<T>(trainingData.GetRowReadOnlySpan(i).ToArray());

            // Euclidean distance via IEngine: sqrt(dot(diff, diff))
            var diff = Engine.Subtract(point, queryPoint);
            T distSq = Engine.DotProduct(diff, diff);
            T dist = NumOps.Sqrt(distSq);

            // Skip zero-distance duplicates
            if (NumOps.GreaterThan(dist, NumOps.Zero))
            {
                distances.Add((i, point, dist));
            }
        }

        return distances
            .OrderBy(x => NumOps.ToDouble(x.Distance))
            .Take(_k)
            .ToList();
    }

    private T ComputeABOF(Vector<T> point, List<(int Index, Vector<T> Point, T Distance)> neighbors)
    {
        if (neighbors.Count < 2) return NumOps.MaxValue;

        T epsilon = NumOps.FromDouble(1e-10);
        var weightedAngles = new List<T>();
        var weights = new List<T>();

        for (int i = 0; i < neighbors.Count; i++)
        {
            var pointA = neighbors[i].Point;
            T distA = neighbors[i].Distance;

            // PA = A - P via IEngine
            var pa = Engine.Subtract(pointA, point);

            for (int k = i + 1; k < neighbors.Count; k++)
            {
                var pointB = neighbors[k].Point;
                T distB = neighbors[k].Distance;

                // PB = B - P via IEngine
                var pb = Engine.Subtract(pointB, point);

                // cos(angle) = (PA · PB) / (|PA| * |PB|) via vectorized dot product
                T dot = Engine.DotProduct(pa, pb);
                T cosAngle = NumOps.Divide(dot, NumOps.Multiply(distA, distB));

                T negOne = NumOps.FromDouble(-1.0);
                if (NumOps.LessThan(cosAngle, negOne)) cosAngle = negOne;
                if (NumOps.GreaterThan(cosAngle, NumOps.One)) cosAngle = NumOps.One;

                // Weight = 1 / (distA² * distB²)
                T distASq = NumOps.Multiply(distA, distA);
                T distBSq = NumOps.Multiply(distB, distB);
                T weight = NumOps.Divide(NumOps.One, NumOps.Multiply(distASq, distBSq));

                weightedAngles.Add(NumOps.Multiply(cosAngle, weight));
                weights.Add(weight);
            }
        }

        if (weights.Count == 0) return NumOps.MaxValue;

        // Weighted variance
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
