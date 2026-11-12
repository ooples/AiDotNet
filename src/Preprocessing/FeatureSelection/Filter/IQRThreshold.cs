using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Interquartile Range (IQR) threshold feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features whose IQR is below a threshold. IQR measures the spread of
/// the middle 50% of data and is robust to outliers.
/// </para>
/// <para><b>For Beginners:</b> IQR is the range between the 25th and 75th percentile.
/// It tells you how spread out the "typical" values are, ignoring extreme high or
/// low values. Features with small IQR have values clustered tightly together and
/// may not be useful for distinguishing samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IQRThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;

    private Vector<T>? _iqrValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Vector<T>? IQRValues => _iqrValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public IQRThreshold(
        double threshold = 0.0,
        int[]? columnIndices = null)
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

        // IQR = sorted[3n/4] - sorted[n/4] per feature
        var X = Tensor<T>.FromMatrix(data);
        var iqrValues = Engine.TensorSubtract(
            PreprocessingTensorOps.ColumnOrderStatistic(X, 3 * n / 4), PreprocessingTensorOps.ColumnOrderStatistic(X, n / 4));
        _iqrValues = iqrValues.ToVector();

        _selectedIndices = PreprocessingTensorOps.MaskPositions(
            Engine.TensorGreaterThan(iqrValues, NumOps.FromDouble(_threshold)));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IQRThreshold has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IQRThreshold does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IQRThreshold has not been fitted.");

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
