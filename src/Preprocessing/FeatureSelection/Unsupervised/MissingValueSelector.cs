using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Missing Value-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features that have too many missing values (zeros, NaN, or
/// below-threshold unique values), keeping only features with sufficient data.
/// </para>
/// <para><b>For Beginners:</b> Features with lots of missing data can't be
/// reliably used for predictions. This selector removes features where too
/// much data is missing, keeping only the ones with enough information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MissingValueSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;
    private readonly double _missingValue;

    private Vector<T>? _missingRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Vector<T>? MissingRatios => _missingRatios;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MissingValueSelector(
        double threshold = 0.5,
        double missingValue = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _missingValue = missingValue;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Missing = NaN or within 1e-10 of the missing-value marker.
        var near = engine.TensorLessThan(
            engine.TensorAbs(engine.TensorSubtractScalar(x, NumOps.FromDouble(_missingValue))), NumOps.FromDouble(1e-10));
        var missing = engine.TensorWhere(engine.TensorIsNan(x), PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One), near);
        var ratios = engine.TensorDivideScalar(engine.ReduceSum(missing, new[] { 0 }, false), NumOps.FromDouble(n));
        _missingRatios = ratios.ToVector();

        // ratio <= threshold (a NaN ratio from zero rows selects nothing).
        _selectedIndices = PreprocessingTensorOps.MaskPositions(
            engine.TensorLessEqual(ratios, PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(_threshold))));
        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MissingValueSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueSelector has not been fitted.");

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
