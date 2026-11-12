using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Median Absolute Deviation (MAD) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses MAD as a robust measure of variability. Unlike variance, MAD is
/// resistant to outliers, making it better for selecting features in
/// datasets with extreme values.
/// </para>
/// <para><b>For Beginners:</b> Standard variance can be thrown off by outliers.
/// MAD uses the median instead of mean, making it more robust. Features with
/// higher MAD have more consistent spread in their values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;

    private Vector<T>? _madScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public Vector<T>? MADScores => _madScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MADSelector(
        int nFeaturesToSelect = -1,
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var scores = n == 0
            ? PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero)
            : FeatureSelectionTensorOps.ColumnMedianAbsoluteDeviation(Tensor<T>.FromMatrix(data));
        _madScores = scores.ToVector();

        // Select features
        if (_nFeaturesToSelect > 0)
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
        }
        else
        {
            // Use threshold
            _selectedIndices = PreprocessingTensorOps.MaskPositions(
                Engine.TensorGreaterThan(scores, NumOps.FromDouble(_threshold)));

            if (_selectedIndices.Length == 0)
            {
                // Keep highest MAD
                _selectedIndices = new[] { PreprocessingTensorOps.FirstArgMax(scores) };
            }
        }

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
            throw new InvalidOperationException("MADSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MADSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADSelector has not been fitted.");

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
