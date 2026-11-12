using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Interquartile Range (IQR) for robust unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The interquartile range is the difference between the 75th and 25th percentiles.
/// It measures the spread of the middle 50% of data and is extremely robust to outliers.
/// </para>
/// <para><b>For Beginners:</b> IQR focuses on the "typical" range of values, ignoring
/// the extreme high and low values. A feature with small IQR has most values bunched
/// together and may not be discriminative. IQR is great for data with outliers because
/// it completely ignores them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InterquartileRange<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minIQR;

    private Vector<T>? _iqrValues;
    private Vector<T>? _q1Values;
    private Vector<T>? _q3Values;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? IQRValues => _iqrValues;
    public Vector<T>? Q1Values => _q1Values;
    public Vector<T>? Q3Values => _q3Values;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InterquartileRange(
        int nFeaturesToSelect = 10,
        double minIQR = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minIQR = minIQR;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // Q1 and Q3 of every feature using linear interpolation
        var q1 = FilterStatisticTensorOps.ColumnQuantile(x, 0.25);
        var q3 = FilterStatisticTensorOps.ColumnQuantile(x, 0.75);
        var iqr = engine.TensorSubtract(q3, q1);
        _q1Values = q1.ToVector();
        _q3Values = q3.ToVector();
        _iqrValues = iqr.ToVector();

        // Select features above threshold or top by IQR
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(iqr, NumOps.FromDouble(_minIQR), _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InterquartileRange has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InterquartileRange does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InterquartileRange has not been fitted.");

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
