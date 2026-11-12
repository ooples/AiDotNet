using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Mean Absolute Deviation (MAD) for robust unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Mean Absolute Deviation measures dispersion as the average distance from the mean.
/// Unlike variance/standard deviation, MAD is more robust to outliers because it
/// doesn't square the deviations.
/// </para>
/// <para><b>For Beginners:</b> MAD tells you on average how far each value is from
/// the mean. Features with low MAD are nearly constant and probably not useful.
/// Unlike standard deviation, MAD isn't overly influenced by extreme outliers,
/// making it a safer choice for messy data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MeanAbsoluteDeviation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minMAD;
    private readonly bool _useMedian;

    private Vector<T>? _madValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MADValues => _madValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MeanAbsoluteDeviation(
        int nFeaturesToSelect = 10,
        double minMAD = 0.0,
        bool useMedian = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minMAD = minMAD;
        _useMedian = useMedian;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // Centre of every feature: the median (more robust; the mean of the two middle values for even n)
        // or the mean
        var center = _useMedian
            ? FilterStatisticTensorOps.ColumnQuantile(x, 0.5)
            : FilterStatisticTensorOps.ColumnMeans(x);

        // Mean absolute deviation from the centre
        var mad = engine.ReduceMean(
            engine.TensorAbs(engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(center, n))), new[] { 0 }, false);
        _madValues = mad.ToVector();

        // Select features above threshold or top by MAD
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(mad, NumOps.FromDouble(_minMAD), _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MeanAbsoluteDeviation has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MeanAbsoluteDeviation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MeanAbsoluteDeviation has not been fitted.");

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
