using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Fisher-Relief Hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Fisher Score (parametric, class separability) with ReliefF
/// (instance-based, neighbor relationships) to leverage both perspectives.
/// </para>
/// <para><b>For Beginners:</b> Fisher Score looks at class averages while
/// ReliefF looks at individual sample relationships. Combining them gives
/// you features that work well from both viewpoints - good class separation
/// AND good at distinguishing similar vs different samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FisherReliefHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fisherWeight;
    private readonly int _nNeighbors;

    private Vector<T>? _fisherScores;
    private Vector<T>? _reliefScores;
    private Vector<T>? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FisherWeight => _fisherWeight;
    public Vector<T>? FisherScores => _fisherScores;
    public Vector<T>? ReliefScores => _reliefScores;
    public Vector<T>? HybridScores => _hybridScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FisherReliefHybrid(
        int nFeaturesToSelect = 10,
        double fisherWeight = 0.5,
        int nNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fisherWeight < 0 || fisherWeight > 1)
            throw new ArgumentException("Fisher weight must be between 0 and 1.", nameof(fisherWeight));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fisherWeight = fisherWeight;
        _nNeighbors = nNeighbors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherReliefHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (p == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Compute Fisher Scores
        var fisher = ComputeFisherScores(X, y, n, p);

        // Compute ReliefF Scores
        var relief = ComputeReliefScores(X, y, n, p);

        // Normalize both score arrays and combine them
        var hybrid = engine.TensorAdd(
            engine.TensorMultiplyScalar(Normalize(fisher), NumOps.FromDouble(_fisherWeight)),
            engine.TensorMultiplyScalar(Normalize(relief), NumOps.FromDouble(1 - _fisherWeight)));

        _fisherScores = fisher.ToVector();
        _reliefScores = relief.ToVector();
        _hybridScores = hybrid.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(hybrid, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Two-class Fisher criterion per feature, (mean0 - mean1)^2 / (var0 + var1) with population variances; the
    /// classes are target &lt; 0.5 and target &gt;= 0.5, and every score is 0 when a class has fewer than 2 samples.
    /// </summary>
    private Tensor<T> ComputeFisherScores(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        var engine = Engine;
        var class0 = engine.TensorLessThan(y, NumOps.FromDouble(0.5));
        var class1 = PreprocessingTensorOps.OneMinus(class0);
        int count0 = PreprocessingTensorOps.CountNonZero(class0);
        int count1 = n - count0;
        if (count0 < 2 || count1 < 2)
            return new Tensor<T>(new[] { p });

        var mean0 = ClassMean(X, class0, count0, n);
        var mean1 = ClassMean(X, class1, count1, n);
        var var0 = ClassMean(engine.TensorSquare(engine.TensorSubtract(X, mean0)), class0, count0, n);
        var var1 = ClassMean(engine.TensorSquare(engine.TensorSubtract(X, mean1)), class1, count1, n);
        var denom = engine.Reshape(engine.TensorAdd(var0, var1), new[] { p });
        var numerator = engine.Reshape(engine.TensorSquare(engine.TensorSubtract(mean0, mean1)), new[] { p });
        var defined = engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-10));
        var safeDenom = engine.TensorWhere(defined, denom, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        return engine.TensorWhere(defined, engine.TensorDivide(numerator, safeDenom), new Tensor<T>(new[] { p }));
    }

    /// <summary>Column means ([1, p]) over the rows of one class (a 0/1 mask of length n).</summary>
    private Tensor<T> ClassMean(Tensor<T> X, Tensor<T> mask, int count, int n)
    {
        var engine = Engine;
        var sums = engine.TensorMatMul(engine.Reshape(mask, new[] { 1, n }), X);
        return engine.TensorDivideScalar(sums, FeatureSelectionTensorOps.Count<T>(count));
    }

    /// <summary>
    /// ReliefF (Kononenko 1994) over m = min(50, n) evenly spaced instances: per-feature differences are scaled by
    /// the feature range, neighbours are the k nearest by the sum of those differences (Manhattan), and the misses
    /// of every other class C are weighted by P(C) / (1 - P(class of the instance)).
    /// </summary>
    private Tensor<T> ComputeReliefScores(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        var engine = Engine;
        var scores = new Tensor<T>(new[] { 1, p });
        int sampleSize = Math.Min(50, n);
        if (sampleSize == 0)
            return engine.Reshape(scores, new[] { p });

        // diff(A, I1, I2) = |I1[A] - I2[A]| / (max(A) - min(A)); a constant feature never differs
        var range = engine.Reshape(engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(X), PreprocessingTensorOps.ColumnMin(X)), new[] { 1, p });
        var safeRange = engine.TensorWhere(engine.TensorGreaterThan(range, NumOps.Zero), range, PreprocessingTensorOps.Filled(new[] { 1, p }, NumOps.One));
        var scaled = engine.TensorDivide(X, safeRange);

        var classIds = FeatureSelectionInformationOps.ClassIndices(y, out int classCount);
        var classOf = classIds.AsSpan().ToArray();
        var membership = engine.TensorOneHot<T>(classIds, classCount); // [n, C]
        var priors = engine.ReduceMean(membership, new[] { 0 }, false); // [C]

        // weight[c_R, C] = P(C) / (1 - P(c_R)) for C != c_R
        var notOwn = PreprocessingTensorOps.OneMinus(engine.TensorEye<T>(classCount));
        var rest = engine.TensorBroadcastTo(engine.Reshape(PreprocessingTensorOps.OneMinus(priors), new[] { classCount, 1 }), new[] { classCount, classCount });
        var safeRest = engine.TensorWhere(engine.TensorGreaterThan(rest, NumOps.Zero), rest, PreprocessingTensorOps.Filled(new[] { classCount, classCount }, NumOps.One));
        var missWeights = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorBroadcastTo(engine.Reshape(priors, new[] { 1, classCount }), new[] { classCount, classCount }), safeRest), notOwn);

        T perUpdate = NumOps.Multiply(FeatureSelectionTensorOps.Count<T>(sampleSize), FeatureSelectionTensorOps.Count<T>(_nNeighbors));
        var everyRow = PreprocessingTensorOps.Filled(new[] { n }, NumOps.One);

        for (int s = 0; s < sampleSize; s++)
        {
            int idx = s * n / sampleSize;
            var instance = engine.TensorSlice(scaled, new[] { idx, 0 }, new[] { 1, p });
            var differences = engine.TensorAbs(engine.TensorSubtract(scaled, engine.TensorBroadcastTo(instance, new[] { n, p }))); // [n, p]
            var closeness = engine.TensorNegate(engine.ReduceSum(differences, new[] { 1 }, false)); // [n], larger = nearer
            var notSelf = engine.TensorSubtract(everyRow, FeatureSelectionTensorOps.OneHot<T>(idx, n));

            int own = classOf[idx];
            for (int c = 0; c < classCount; c++)
            {
                var candidates = engine.TensorMultiply(notSelf, engine.Reshape(engine.TensorSlice(membership, new[] { 0, c }, new[] { n, 1 }), new[] { n }));
                var nearest = FeatureSelectionTensorOps.TopKWhere(closeness, candidates, _nNeighbors);
                if (nearest.Length == 0)
                    continue;

                var neighbourDiffs = engine.ReduceSum(
                    engine.TensorIndexSelect(differences, new Tensor<int>(nearest, new[] { nearest.Length }), 0), new[] { 0 }, true); // [1, p]
                if (c == own)
                {
                    scores = engine.TensorSubtract(scores, engine.TensorDivideScalar(neighbourDiffs, perUpdate));
                }
                else
                {
                    var weight = engine.TensorBroadcastTo(engine.TensorSlice(missWeights, new[] { own, c }, new[] { 1, 1 }), new[] { 1, p });
                    scores = engine.TensorAdd(scores, engine.TensorDivideScalar(engine.TensorMultiply(neighbourDiffs, weight), perUpdate));
                }
            }
        }

        return engine.Reshape(scores, new[] { p });
    }

    /// <summary>Min-max scaling to [0, 1]; every score becomes 0.5 when the range is below 1e-10.</summary>
    private Tensor<T> Normalize(Tensor<T> scores)
    {
        var engine = Engine;
        T min = engine.TensorMinValue(scores);
        T range = NumOps.Subtract(engine.TensorMaxValue(scores), min);
        if (NumOps.LessThan(range, NumOps.FromDouble(1e-10)))
            return PreprocessingTensorOps.Filled(scores.Shape.ToArray(), NumOps.FromDouble(0.5));

        return engine.TensorDivideScalar(engine.TensorSubtractScalar(scores, min), range);
    }

    /// <inheritdoc />

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherReliefHybrid has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FisherReliefHybrid does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherReliefHybrid has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
