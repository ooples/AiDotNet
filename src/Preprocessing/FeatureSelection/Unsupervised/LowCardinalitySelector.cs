using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Low Cardinality Feature Removal.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with very low cardinality (few unique values) as they
/// may not provide enough discriminative power.
/// </para>
/// <para><b>For Beginners:</b> Cardinality is the number of unique values a
/// feature has. A feature with only 1 or 2 unique values doesn't vary much
/// and may not be useful for distinguishing between samples. This selector
/// removes such low-variety features.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LowCardinalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly int _minCardinality;

    private int[]? _cardinalities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MinCardinality => _minCardinality;
    public int[]? Cardinalities => _cardinalities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LowCardinalitySelector(
        int minCardinality = 2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minCardinality < 1)
            throw new ArgumentException("Minimum cardinality must be at least 1.", nameof(minCardinality));

        _minCardinality = minCardinality;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n == 0 || p == 0)
        {
            // No values: every cardinality is 0, below the minimum of at least 1.
            _cardinalities = new int[p];
            _selectedIndices = Array.Empty<int>();
            IsFitted = true;
            return;
        }

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Distinct non-NaN values per column, compared after rounding to 10 decimals.
        var rounded = engine.TensorDivideScalar(
            engine.TensorRound(engine.TensorMultiplyScalar(x, NumOps.FromDouble(1e10))), NumOps.FromDouble(1e10));
        var nan = engine.TensorIsNan(x);
        var lowest = PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.FromDouble(double.NegativeInfinity));
        var columnMax = engine.ReduceMax(engine.TensorWhere(nan, lowest, rounded), new[] { 0 }, true);

        // NaNs take the column's largest real value, so they never add a distinct value of their own.
        var filled = engine.TensorWhere(nan, engine.TensorBroadcastTo(columnMax, new[] { n, p }), rounded);
        var sorted = engine.TensorSort(filled, 0, false).Values;
        var distinct = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        if (n > 1)
        {
            var changes = engine.TensorNotEquals(
                engine.TensorSlice(sorted, new[] { 1, 0 }, new[] { n - 1, p }),
                engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { n - 1, p }));
            distinct = engine.TensorAdd(distinct, engine.ReduceSum(changes, new[] { 0 }, false));
        }

        // An all-NaN column has no values at all.
        var realCount = engine.ReduceSum(
            engine.TensorWhere(nan, PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.Zero),
                PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One)), new[] { 0 }, false);
        var hasReal = engine.TensorGreaterThan(realCount, NumOps.Zero);
        var cardinalities = engine.TensorMultiply(distinct, hasReal);

        _cardinalities = cardinalities.Cast<int>().ToArray();
        _selectedIndices = PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(cardinalities, NumOps.FromDouble(_minCardinality)));
        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LowCardinalitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LowCardinalitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LowCardinalitySelector has not been fitted.");

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
