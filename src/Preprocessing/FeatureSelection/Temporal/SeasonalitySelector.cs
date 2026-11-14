using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Seasonality based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the presence of seasonal (periodic) patterns,
/// identifying features with repeating cycles.
/// </para>
/// <para><b>For Beginners:</b> Seasonal features show repeating patterns (like
/// higher sales every December). This selector finds features with strong
/// periodic behavior using spectral analysis.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SeasonalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minPeriod;
    private readonly int _maxPeriod;

    private Vector<T>? _seasonalityStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MinPeriod => _minPeriod;
    public int MaxPeriod => _maxPeriod;
    public Vector<T>? SeasonalityStrengths => _seasonalityStrengths;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SeasonalitySelector(
        int nFeaturesToSelect = 10,
        int minPeriod = 2,
        int maxPeriod = 12,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minPeriod = minPeriod;
        _maxPeriod = maxPeriod;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        int effectiveMaxPeriod = Math.Min(_maxPeriod, n / 2);
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var totalVariance = SelectorTensorOps.ColumnSumOfSquares(centered);

        // Largest DFT power (|X(1/period)|^2 / n) over the candidate periods
        var maxPower = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (effectiveMaxPeriod >= _minPeriod)
        {
            var periods = engine.TensorArange<T>(_minPeriod, effectiveMaxPeriod + 1, 1);
            var frequencies = engine.TensorDivide(
                PreprocessingTensorOps.Filled(periods.Shape.ToArray(), NumOps.One), periods);
            var power = engine.TensorDivideScalar(
                SelectorTensorOps.FrequencyPower(centered, frequencies), NumOps.FromDouble(n));
            maxPower = engine.TensorMax(maxPower, engine.ReduceMax(power, new[] { 0 }, false));
        }

        // Seasonality strength as ratio of max periodic power to total variance (near-constant columns score 0)
        var strengths = SelectorTensorOps.ZeroUnless(
            PreprocessingTensorOps.AtLeast(totalVariance, NumOps.FromDouble(1e-10)),
            engine.TensorDivide(maxPower, totalVariance));
        _seasonalityStrengths = strengths.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(strengths, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("SeasonalitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SeasonalitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalitySelector has not been fitted.");

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
