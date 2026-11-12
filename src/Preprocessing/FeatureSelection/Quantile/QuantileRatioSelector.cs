using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantile;

/// <summary>
/// Quantile Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the ratio between two quantiles, measuring
/// the relative spread at different parts of the distribution.
/// </para>
/// <para><b>For Beginners:</b> Quantile ratio compares values at different
/// percentiles. For example, the 90th/10th percentile ratio shows how much
/// larger the high values are compared to low values. This can detect features
/// with skewed distributions or outliers.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QuantileRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _upperQuantile;
    private readonly double _lowerQuantile;

    private Vector<T>? _ratioValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double UpperQuantile => _upperQuantile;
    public double LowerQuantile => _lowerQuantile;
    public Vector<T>? RatioValues => _ratioValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public QuantileRatioSelector(
        int nFeaturesToSelect = 10,
        double upperQuantile = 0.90,
        double lowerQuantile = 0.10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (upperQuantile <= 0 || upperQuantile > 1)
            throw new ArgumentException("Upper quantile must be between 0 and 1.", nameof(upperQuantile));
        if (lowerQuantile < 0 || lowerQuantile >= 1)
            throw new ArgumentException("Lower quantile must be between 0 and 1.", nameof(lowerQuantile));
        if (lowerQuantile >= upperQuantile)
            throw new ArgumentException("Lower quantile must be less than upper quantile.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _upperQuantile = upperQuantile;
        _lowerQuantile = lowerQuantile;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var upper = FilterStatisticTensorOps.ColumnQuantile(x, _upperQuantile);
        var lower = FilterStatisticTensorOps.ColumnQuantile(x, _lowerQuantile);

        // |upper / lower| (mixed signs give the same |upper| / |lower|); a lower quantile near zero
        // (|lower| < 1e-10) falls back to the difference upper - lower.
        var absLower = engine.TensorAbs(lower);
        var nearZero = engine.TensorLessThan(absLower, NumOps.FromDouble(1e-10));
        var safeLower = engine.TensorWhere(nearZero, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), absLower);
        var ratio = engine.TensorDivide(engine.TensorAbs(upper), safeLower);
        var ratios = engine.TensorWhere(nearZero, engine.TensorSubtract(upper, lower), ratio);
        _ratioValues = ratios.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(ratios, numToSelect);

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
            throw new InvalidOperationException("QuantileRatioSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("QuantileRatioSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantileRatioSelector has not been fitted.");

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
