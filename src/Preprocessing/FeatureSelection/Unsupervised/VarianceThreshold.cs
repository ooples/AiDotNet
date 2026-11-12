using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Variance Threshold for removing low-variance features.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with variance below a specified threshold. Features with
/// very low variance carry little information and are often constant or
/// near-constant.
/// </para>
/// <para><b>For Beginners:</b> If a feature has the same value for almost all
/// samples, it can't help distinguish between them. This method removes such
/// uninformative features automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class VarianceThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;

    private Vector<T>? _variances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Vector<T>? Variances => _variances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public VarianceThreshold(double threshold = 0.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold < 0)
            throw new ArgumentException("Threshold must be non-negative.", nameof(threshold));

        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Population variance of every column
        var variances = Engine.TensorDivideScalar(
            SelectorTensorOps.ColumnSumOfSquares(SelectorTensorOps.CenterColumns(Tensor<T>.FromMatrix(data))),
            NumOps.FromDouble(n));
        _variances = variances.ToVector();

        // Select features above threshold
        _selectedIndices = PreprocessingTensorOps.MaskPositions(
            Engine.TensorGreaterThan(variances, NumOps.FromDouble(_threshold)));

        // If nothing selected, keep highest variance
        if (_selectedIndices.Length == 0)
            _selectedIndices = new[] { PreprocessingTensorOps.FirstArgMax(variances) };

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
            throw new InvalidOperationException("VarianceThreshold has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("VarianceThreshold does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceThreshold has not been fitted.");

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
