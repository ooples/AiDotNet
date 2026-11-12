using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Gini Index for feature selection based on impurity reduction.
/// </summary>
/// <remarks>
/// <para>
/// The Gini Index measures impurity in classification. Features that split
/// data into purer groups (lower Gini) are more informative for prediction.
/// </para>
/// <para><b>For Beginners:</b> Gini measures how mixed a group is. A group
/// with all same-class items has Gini=0 (pure). A 50/50 split has Gini=0.5.
/// Good features create pure groups when you split by them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GiniIndex<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _giniGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GiniGains => _giniGains;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GiniIndex(int nFeaturesToSelect = 10, int nBins = 10, int[]? columnIndices = null)
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
            "GiniIndex requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);
        var classOneHot = StatisticalTestHelper.ClassOneHot(target, out _);
        T total = NumOps.FromDouble(n);
        T bins = NumOps.FromDouble(_nBins);
        T zero = NumOps.Zero;

        // parent Gini impurity 1 - sum q_c^2
        var classCounts = Engine.ReduceSum(classOneHot, new[] { 0 }, false);
        T parentGini = NumOps.Subtract(NumOps.One,
            Engine.TensorSumOfSquares(Engine.TensorDivideScalar(classCounts, total)));

        // split thresholds min + b * range / nBins (range 1 for a constant column), left side x <= threshold
        var min = FeatureSelectionTensorOps.ColumnMin(x);
        var range = Engine.TensorSubtract(FeatureSelectionTensorOps.ColumnMax(x), min);
        range = Engine.TensorWhere(Engine.TensorLessThan(range, NumOps.FromDouble(1e-10)),
            FilterStatisticTensorOps.FilledLike(range, NumOps.One), range);
        var steps = Engine.TensorArange<T>(1, _nBins, 1).ToArray();
        var best = FilterStatisticTensorOps.FilledLike(min, zero);
        foreach (T step in steps)
        {
            var threshold = Engine.TensorAdd(min, Engine.TensorDivideScalar(Engine.TensorMultiplyScalar(range, step), bins));
            var left = FilterStatisticTensorOps.OneMinus(
                Engine.TensorGreaterThan(x, FilterStatisticTensorOps.RowBroadcast(threshold, n)));
            var leftCounts = Engine.TensorMatMul(Engine.TensorTranspose(left), classOneHot);
            var rightCounts = Engine.TensorSubtract(FilterStatisticTensorOps.RowBroadcast(classCounts, p), leftCounts);
            var leftTotal = Engine.ReduceSum(leftCounts, new[] { 1 }, false);
            var rightTotal = Engine.ReduceSum(rightCounts, new[] { 1 }, false);

            // weighted child impurity: sum over sides of (total - sum count^2 / total) / n
            var weighted = Engine.TensorDivideScalar(Engine.TensorAdd(
                SideImpurity(leftCounts, leftTotal), SideImpurity(rightCounts, rightTotal)), total);
            var reduction = Engine.TensorNegate(Engine.TensorSubtractScalar(weighted, parentGini));
            var bothSides = Engine.TensorMultiply(
                Engine.TensorGreaterThan(leftTotal, zero), Engine.TensorGreaterThan(rightTotal, zero));
            best = Engine.TensorMax(best, FeatureSelectionTensorOps.KeepOrFill(reduction, bothSides, zero));
        }

        _giniGains = best.ToVector();
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(best, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // total * Gini impurity = total - sum count^2 / total for every row of [p, k] class counts (0 for an empty side)
    private Tensor<T> SideImpurity(Tensor<T> counts, Tensor<T> totals) =>
        Engine.TensorSubtract(totals, FilterStatisticTensorOps.GuardedDivide(
            Engine.ReduceSum(Engine.TensorSquare(counts), new[] { 1 }, false), totals, 0, NumOps.Zero));

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
            throw new InvalidOperationException("GiniIndex has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GiniIndex does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniIndex has not been fitted.");

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
