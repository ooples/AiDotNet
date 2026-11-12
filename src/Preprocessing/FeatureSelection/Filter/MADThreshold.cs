using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Median Absolute Deviation (MAD) threshold feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features whose Median Absolute Deviation is below a threshold. MAD
/// is a robust measure of spread that is less sensitive to outliers than variance.
/// </para>
/// <para><b>For Beginners:</b> Instead of using variance (which can be skewed by
/// extreme values), MAD looks at how far typical values are from the middle value.
/// This makes it better for data with outliers. Features with low MAD are nearly
/// constant and don't provide useful information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MADThreshold<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _threshold;

    private Vector<T>? _madValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Threshold => _threshold;
    public Vector<T>? MADValues => _madValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MADThreshold(
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
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // MAD = median |x - median(x)| per feature (scipy.stats.median_abs_deviation, scale 1); the median of an
        // even number of values is the mean of the two middle ones
        var median = FilterStatisticTensorOps.ColumnQuantile(x, 0.5);
        var deviations = engine.TensorAbs(engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(median, n)));
        var mad = FilterStatisticTensorOps.ColumnQuantile(deviations, 0.5);
        _madValues = mad.ToVector();

        _selectedIndices = FilterStatisticTensorOps.MaskPositions(engine.TensorGreaterThan(mad, NumOps.FromDouble(_threshold)));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADThreshold has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MADThreshold does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADThreshold has not been fitted.");

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
