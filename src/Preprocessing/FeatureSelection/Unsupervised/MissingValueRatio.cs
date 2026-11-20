using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Missing Value Ratio filter for removing features with too many missing values.
/// </summary>
/// <remarks>
/// <para>
/// Removes features where the proportion of missing values exceeds a threshold.
/// Missing values are identified as NaN, infinity, or optionally a specified
/// sentinel value.
/// </para>
/// <para><b>For Beginners:</b> Features with many missing values are unreliable.
/// If more than X% of a feature's values are missing, it's often better to
/// remove it entirely rather than try to impute the missing data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MissingValueRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;
    private readonly double? _missingIndicator;

    private Vector<T>? _missingRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Vector<T>? MissingRatios => _missingRatios;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MissingValueRatio(
        double threshold = 0.5,
        double? missingIndicator = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _threshold = threshold;
        _missingIndicator = missingIndicator;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Missing = NaN, +/-infinity, or within 1e-10 of the missing indicator when one is set
        var ones = PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One);
        var missing = engine.TensorEquals(engine.TensorAbs(x), NumOps.FromDouble(double.PositiveInfinity));
        if (_missingIndicator.HasValue)
        {
            var near = engine.TensorLessThan(
                engine.TensorAbs(engine.TensorSubtractScalar(x, NumOps.FromDouble(_missingIndicator.Value))),
                NumOps.FromDouble(1e-10));
            missing = engine.TensorMax(missing, near);
        }

        missing = engine.TensorWhere(engine.TensorIsNan(x), ones, missing);
        var ratios = engine.TensorDivideScalar(engine.ReduceSum(missing, new[] { 0 }, false), NumOps.FromDouble(n));
        _missingRatios = ratios.ToVector();

        // Select features below threshold
        _selectedIndices = PreprocessingTensorOps.MaskPositions(
            engine.TensorLessEqual(ratios, PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(_threshold))));

        // If nothing selected, keep lowest missing ratio
        if (_selectedIndices.Length == 0)
            _selectedIndices = new[] { PreprocessingTensorOps.FirstArgMax(engine.TensorNegate(ratios)) };

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueRatio has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MissingValueRatio does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MissingValueRatio has not been fitted.");

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
