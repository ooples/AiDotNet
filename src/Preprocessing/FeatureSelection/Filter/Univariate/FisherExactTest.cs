using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Fisher's Exact Test for categorical feature selection with small samples.
/// </summary>
/// <remarks>
/// <para>
/// Fisher's Exact Test computes the exact probability of observing the contingency
/// table assuming independence. Unlike chi-square, it's accurate for small samples.
/// </para>
/// <para><b>For Beginners:</b> When you have categorical features and binary targets,
/// this test determines if there's a significant relationship. Unlike chi-square,
/// it works well even with small sample sizes or sparse cells.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FisherExactTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _pValues;
    private Vector<T>? _oddsRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? OddsRatios => _oddsRatios;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FisherExactTest(int nFeaturesToSelect = 10, int nBins = 2, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherExactTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Binary target (find unique classes)
        var classOneHot = StatisticalTestHelper.ClassOneHot(target, out int classCount);
        if (classCount != 2)
            throw new ArgumentException("FisherExactTest requires exactly 2 classes for binary target.");

        // 2x2 table per feature: feature at or below its split value (the sorted value at n / 2) against class 0
        var x = FilterStatisticTensorOps.AsTensor(data);
        var split = FeatureScoringTensorOps.Rows(Engine.TensorSort(x, 0, false).Item1, new[] { n / 2 });
        var low = PreprocessingTensorOps.OneMinus(Engine.TensorGreaterThan(x, Engine.TensorBroadcastTo(split, new[] { n, p })));
        // the table is counted in int and the test run in double, as master did: 0/1 sums in T stop at 2048 for
        // Half, and neither the support k = 0..n nor its log-binomials are exact in Half
        var lowCounts = low.Cast<int>();
        var firstClass = PreprocessingTensorOps.ColumnVector(classOneHot, 0).Cast<int>();
        var a = Engine.ReduceSum(
            Engine.TensorMultiply(lowCounts, Engine.TensorBroadcastTo(Engine.Reshape(firstClass, new[] { n, 1 }), new[] { n, p })),
            new[] { 0 }, false);
        var b = Engine.TensorSubtract(Engine.ReduceSum(lowCounts, new[] { 0 }, false), a);
        var c = Engine.TensorNegate(Engine.TensorSubtractScalar(a, Engine.TensorSum(firstClass)));
        var d = Engine.TensorNegate(Engine.TensorSubtractScalar(Engine.TensorAdd(Engine.TensorAdd(a, b), c), n));
        var (pValues, oddsRatios) = FeatureScoringTensorOps.FisherExact(
            a.Cast<double>(), b.Cast<double>(), c.Cast<double>(), d.Cast<double>(), n);
        _pValues = pValues.Cast<T>().ToVector();
        _oddsRatios = oddsRatios.Cast<T>().ToVector();
        _selectedIndices = PreprocessingTensorOps.BottomKIndicesSorted(pValues, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("FisherExactTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FisherExactTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherExactTest has not been fitted.");

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
