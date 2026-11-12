using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Text;

/// <summary>
/// Chi-Square test for text classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Applies chi-square test to measure the association between terms and
/// document classes. Terms with high chi-square values are more discriminative.
/// </para>
/// <para><b>For Beginners:</b> This test finds words that appear significantly
/// more often in one class than expected by chance. These discriminative words
/// are the best features for text classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ChiSquareText<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _binarize;

    private Vector<T>? _chiSquareScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ChiSquareScores => _chiSquareScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ChiSquareText(
        int nFeaturesToSelect = 100,
        bool binarize = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _binarize = binarize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquareText requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique classes and each document's class
        var labels = Tensor<T>.FromVector(target);
        var classList = PreprocessingTensorOps.SortedUnique(labels);
        int nClasses = classList.Length;
        var classIndex = PreprocessingTensorOps.LookupSorted(classList, labels, out _);
        var membership = Engine.TensorOneHot<T>(classIndex, nClasses);

        // Term presence and the one-vs-rest 2x2 contingency counts of every (class, term):
        // a = present & class, b = present & other, c = absent & class, d = absent & other
        var present = Engine.TensorGreaterThan(Tensor<T>.FromMatrix(data), NumOps.FromDouble(_binarize ? 0 : 0.5));
        var shape = new[] { nClasses, p };
        T total = NumOps.FromDouble(n);
        var a = Engine.TensorMatMul(Engine.TensorTranspose(membership), present);
        var termCount = Engine.TensorBroadcastTo(Engine.ReduceSum(present, new[] { 0 }, true), shape);
        var classCount = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.ReduceSum(membership, new[] { 0 }, false), new[] { nClasses, 1 }), shape);
        var absentCount = Engine.TensorAddScalar(Engine.TensorNegate(termCount), total);
        var otherCount = Engine.TensorAddScalar(Engine.TensorNegate(classCount), total);
        var b = Engine.TensorSubtract(termCount, a);
        var c = Engine.TensorSubtract(classCount, a);
        var d = Engine.TensorSubtract(otherCount, b);

        // chi2(t, c) = sum over the four cells of (observed - expected)^2 / expected (cells with expected > 0)
        var chiSq = Engine.TensorAdd(
            Engine.TensorAdd(CellTerm(a, termCount, classCount, total), CellTerm(b, termCount, otherCount, total)),
            Engine.TensorAdd(CellTerm(c, absentCount, classCount, total), CellTerm(d, absentCount, otherCount, total)));

        // chi2_avg(t) = sum_c P(c) chi2(t, c) (Yang and Pedersen, 1997)
        var prior = Engine.TensorDivideScalar(classCount, total);
        var scores = Engine.ReduceSum(Engine.TensorMultiply(prior, chiSq), new[] { 0 }, false);
        _chiSquareScores = scores.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// <c>(observed - expected)^2 / expected</c> with <c>expected = rowTotal * columnTotal / n</c>, 0 where the
    /// expected count is not positive.
    /// </summary>
    private Tensor<T> CellTerm(Tensor<T> observed, Tensor<T> rowTotal, Tensor<T> columnTotal, T n)
    {
        var expected = Engine.TensorDivideScalar(Engine.TensorMultiply(rowTotal, columnTotal), n);
        var positive = Engine.TensorGreaterThan(expected, NumOps.Zero);
        var safeExpected = Engine.TensorWhere(positive, expected, PreprocessingTensorOps.Filled(expected.Shape.ToArray(), NumOps.One));
        var difference = Engine.TensorSubtract(observed, expected);
        return SelectorTensorOps.ZeroUnless(positive, Engine.TensorDivide(Engine.TensorMultiply(difference, difference), safeExpected));
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
            throw new InvalidOperationException("ChiSquareText has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ChiSquareText does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquareText has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
