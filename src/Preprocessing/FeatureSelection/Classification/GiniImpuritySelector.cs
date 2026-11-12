using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Gini Impurity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their ability to reduce Gini impurity when used
/// for splitting data, similar to decision tree feature importance.
/// </para>
/// <para><b>For Beginners:</b> Gini impurity measures how mixed up the classes are.
/// A pure group (all same class) has impurity 0. This selector keeps features that
/// best reduce impurity when used to split the data - the same criterion decision
/// trees use.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GiniImpuritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSplits;

    private Vector<T>? _giniReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GiniReductions => _giniReductions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GiniImpuritySelector(
        int nFeaturesToSelect = 10,
        int nSplits = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSplits = nSplits;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GiniImpuritySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var membership = engine.TensorTranspose(PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length)); // [k, n]
        // Class counts are kept in double: summed in a narrow T (Half stops counting at 2048) they saturate
        var membershipCounts = membership.Cast<double>();
        var classTotals = engine.ReduceSum(membershipCounts, new[] { 1 }, true);                                 // [k, 1]
        double count = n;
        double parentGini = 1.0 - engine.TensorSumOfSquares(engine.TensorDivideScalar(classTotals, count));

        // Candidate thresholds min + s * (max - min) / (nSplits + 1) for s = 1..nSplits, one row per split
        var minVal = PreprocessingTensorOps.ColumnMin(X);
        var step = engine.TensorDivideScalar(
            engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(X), minVal), NumOps.FromDouble(_nSplits + 1));
        var splitNumbers = new int[_nSplits];
        for (int s = 0; s < _nSplits; s++)
            splitNumbers[s] = s + 1;
        var thresholds = engine.TensorAdd(
            engine.Reshape(minVal, new[] { 1, p }),
            engine.TensorOuter(new Tensor<int>(splitNumbers, new[] { _nSplits }).Cast<T>(), step)); // [nSplits, p]

        var parent = PreprocessingTensorOps.Filled(new[] { p }, parentGini);
        var bestReduction = new Tensor<double>(new[] { p });
        for (int s = 0; s < _nSplits; s++)
        {
            // Split data: left is x <= threshold
            var threshold = engine.TensorBroadcastTo(PreprocessingTensorOps.Row(thresholds, s), new[] { n, p });
            var left = PreprocessingTensorOps.Not(engine.TensorGreaterThan(X, threshold)).Cast<double>();
            var leftCounts = engine.TensorMatMul(membershipCounts, left);                                // [k, p]
            var rightCounts = engine.TensorSubtract(classTotals, leftCounts);
            var leftN = engine.ReduceSum(leftCounts, new[] { 0 }, false);                           // [p]
            var rightN = engine.ReduceSum(rightCounts, new[] { 0 }, false);

            // Child Gini values: 1 - sum_c (count_c / size)^2
            var leftGini = PreprocessingTensorOps.OneMinus(
                engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(leftCounts, leftN)), new[] { 0 }, false));
            var rightGini = PreprocessingTensorOps.OneMinus(
                engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(rightCounts, rightN)), new[] { 0 }, false));

            var weightedChildGini = engine.TensorDivideScalar(
                engine.TensorAdd(engine.TensorMultiply(leftN, leftGini), engine.TensorMultiply(rightN, rightGini)), count);
            var reduction = engine.TensorSubtract(parent, weightedChildGini);

            // Splits with an empty side are skipped
            var valid = engine.TensorMultiply(engine.TensorGreaterThan(leftN, 0.0), engine.TensorGreaterThan(rightN, 0.0));
            bestReduction = engine.TensorMax(bestReduction, engine.TensorWhere(valid, reduction, bestReduction));
        }

        _giniReductions = bestReduction.Cast<T>().ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(bestReduction, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("GiniImpuritySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GiniImpuritySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniImpuritySelector has not been fitted.");

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
