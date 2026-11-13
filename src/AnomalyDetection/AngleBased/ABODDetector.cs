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
public partial class ABODDetector<T> : AnomalyDetectorBase<T>
{
    [Buffer]
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

        var engine = Engine;
        int n = trainingData.Rows;
        var training = AnomalyTensorOps.ToTensor(trainingData);
        var queries = AnomalyTensorOps.ToTensor(X);

        // Pairs (A, B) of training points with A before B
        var ones = Tensor<T>.CreateOnes(new[] { n, n });
        var pairs = engine.TensorTranspose(engine.TensorTril(ones, -1));

        // Constant tensors shared by every query
        var constants = new AbofConstants
        {
            Pairs = pairs,
            PairOnes = ones,
            PairZeros = Tensor<T>.CreateZeros(new[] { n, n }),
            PointOnes = Tensor<T>.CreateOnes(new[] { n }),
            MaxValue = engine.Fill(1, NumOps.MaxValue).AsTensor(),
            Epsilon = NumOps.FromDouble(1e-10),
            MinusOne = NumOps.FromDouble(-1.0)
        };

        // ABOF (Angle-Based Outlier Factor) of every query point
        var abofs = new Tensor<T>[X.Rows];
        for (int p = 0; p < X.Rows; p++)
        {
            abofs[p] = ComputeABOF(
                AnomalyTensorOps.SubtractRow(training, engine.TensorSliceAxis(queries, 0, p)), constants);
        }

        // ABOF is inversely related to outlierness — invert for consistent scoring
        var abof = engine.TensorConcatenate(abofs, 0);
        var positive = engine.TensorGreaterThan(abof, NumOps.Zero);
        return engine.TensorWhere(positive,
            engine.TensorReciprocal(engine.TensorWhere(positive, abof, Tensor<T>.CreateOnes(new[] { X.Rows }))),
            engine.Fill(X.Rows, NumOps.MaxValue).AsTensor()).ToVector();
    }

    // ABOF of a point from the difference vectors PA = A - P to every training point A ([n, d]), over the
    // pairs (A, B) marked in pairs ([n, n] 0/1): the variance of cos(PA, PB) weighted by 1 / (|PA|^2 |PB|^2),
    // skipping training points closer than 1e-10. Returns a [1] tensor; MaxValue without (weighted) pairs.
    private Tensor<T> ComputeABOF(Tensor<T> differences, AbofConstants constants)
    {
        var engine = Engine;
        int n = differences.Shape[0];
        var pairs = constants.Pairs;
        T epsilon = constants.Epsilon;
        var zeros = constants.PairZeros;
        var maxValue = constants.MaxValue;

        // |PA|^2 and |PA| via dot products; points at distance < epsilon do not take part
        var normSq = engine.ReduceSum(engine.TensorSquare(differences), new[] { 1 }, false);
        var norm = engine.TensorSqrt(normSq);
        var valid = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(norm, epsilon));
        var safeNorm = engine.TensorWhere(valid, norm, constants.PointOnes);
        var safeNormSq = engine.TensorWhere(valid, normSq, constants.PointOnes);
        var mask = engine.TensorMultiply(pairs, engine.TensorMultiply(
            AnomalyTensorOps.BroadcastColumn(valid, n), AnomalyTensorOps.BroadcastRow(valid, n)));

        // cos(angle) = (PA . PB) / (|PA| |PB|), clamped to [-1, 1]; weight = 1 / (|PA|^2 |PB|^2)
        var dots = engine.TensorMatMul(differences, engine.TensorTranspose(differences));
        var cosAngle = engine.TensorClamp(
            engine.TensorDivide(dots, AnomalyTensorOps.MultiplyRow(AnomalyTensorOps.BroadcastColumn(safeNorm, n), safeNorm)),
            constants.MinusOne, NumOps.One);
        var weight = engine.TensorWhere(mask, engine.TensorReciprocal(
            AnomalyTensorOps.MultiplyRow(AnomalyTensorOps.BroadcastColumn(safeNormSq, n), safeNormSq)), zeros);
        var weightedAngles = engine.TensorWhere(mask, engine.TensorMultiply(cosAngle, weight), zeros);

        // Weighted mean and variance of the angles (angle = weighted angle / weight)
        var pairCount = SumAll(mask);
        var sumWeights = SumAll(weight);
        var mean = engine.TensorDivide(SumAll(weightedAngles), sumWeights);
        var angle = engine.TensorDivide(weightedAngles, engine.TensorWhere(mask, weight, constants.PairOnes));
        var diff = engine.TensorSubtract(angle, engine.TensorBroadcastTo(engine.Reshape(mean, new[] { 1, 1 }), new[] { n, n }));
        var variance = SumAll(engine.TensorWhere(mask, engine.TensorMultiply(weight, engine.TensorSquare(diff)), zeros));

        var abof = engine.TensorWhere(engine.TensorLessThan(sumWeights, epsilon), maxValue,
            engine.TensorDivide(variance, sumWeights));
        return engine.TensorWhere(engine.TensorGreaterThan(pairCount, NumOps.Zero), abof, maxValue);
    }

    // Tensors and scalars that do not change between the queries of one scoring call.
    private sealed class AbofConstants
    {
        public required Tensor<T> Pairs { get; init; }
        public required Tensor<T> PairOnes { get; init; }
        public required Tensor<T> PairZeros { get; init; }
        public required Tensor<T> PointOnes { get; init; }
        public required Tensor<T> MaxValue { get; init; }
        public required T Epsilon { get; init; }
        public required T MinusOne { get; init; }
    }

    // Sum of all entries of a [n, n] tensor as a [1] tensor.
    private Tensor<T> SumAll(Tensor<T> values)
    {
        var engine = Engine;
        return engine.ReduceSum(engine.ReduceSum(values, new[] { 1 }, false), new[] { 0 }, true);
    }
}
