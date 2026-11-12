using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Range;

/// <summary>
/// Percentile Range based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the range between specified percentiles,
/// allowing customizable robust measures of spread.
/// </para>
/// <para><b>For Beginners:</b> This selector measures the spread of data between
/// two percentiles you choose. For example, the 10th-90th percentile range
/// captures 80% of the data, ignoring extreme outliers at both ends.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PercentileRangeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    private Vector<T>? _rangeValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowerPercentile => _lowerPercentile;
    public double UpperPercentile => _upperPercentile;
    public Vector<T>? RangeValues => _rangeValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PercentileRangeSelector(
        int nFeaturesToSelect = 10,
        double lowerPercentile = 0.10,
        double upperPercentile = 0.90,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lowerPercentile < 0 || lowerPercentile > 1)
            throw new ArgumentException("Lower percentile must be between 0 and 1.", nameof(lowerPercentile));
        if (upperPercentile < 0 || upperPercentile > 1)
            throw new ArgumentException("Upper percentile must be between 0 and 1.", nameof(upperPercentile));
        if (lowerPercentile >= upperPercentile)
            throw new ArgumentException("Lower percentile must be less than upper percentile.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Range between the configured percentiles per column (linear interpolation, as numpy does by default)
        var x = Tensor<T>.FromMatrix(data);
        var range = Engine.TensorSubtract(
            FeatureSelectionTensorOps.ColumnQuantile(x, _upperPercentile),
            FeatureSelectionTensorOps.ColumnQuantile(x, _lowerPercentile));
        _rangeValues = range.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(range, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PercentileRangeSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PercentileRangeSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PercentileRangeSelector has not been fitted.");

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
