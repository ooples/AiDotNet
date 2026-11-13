using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Seasonality-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Seasonality feature selection identifies features with periodic patterns at specified
/// frequencies. It uses spectral analysis to detect features with strong seasonal
/// components, which can be valuable for forecasting cyclical phenomena.
/// </para>
/// <para><b>For Beginners:</b> Many real-world patterns repeat: daily traffic peaks,
/// monthly sales cycles, yearly temperature changes. This method finds features that
/// have such repeating patterns. It uses frequency analysis (like detecting musical
/// notes in sound) to identify periodic behavior in data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SeasonalityFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[] _seasonalPeriods;

    private Vector<T>? _seasonalityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int[] SeasonalPeriods => _seasonalPeriods;
    public Vector<T>? SeasonalityScores => _seasonalityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SeasonalityFS(
        int nFeaturesToSelect = 10,
        int[]? seasonalPeriods = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _seasonalPeriods = seasonalPeriods ?? [7, 12, 24, 52]; // Common periods: weekly, monthly, hourly, yearly
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Centre every column and compute its population variance
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var variance = Engine.TensorDivideScalar(SelectorTensorOps.ColumnSumOfSquares(centered), NumOps.FromDouble(n));
        var varying = PreprocessingTensorOps.AtLeast(variance, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safeVariance = Engine.TensorWhere(varying, variance, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));

        // Spectral power |sum_t (x_t - mean) e^(-2 pi i t / period)|^2 / n^2 at each usable seasonal period
        var periods = _seasonalPeriods.Where(period => period >= 2 && period <= n / 2).ToArray();
        var maxPower = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (periods.Length > 0)
        {
            var frequencies = Engine.TensorReciprocal(new Tensor<int>(periods, new[] { periods.Length }).Cast<T>());
            var power = Engine.TensorDivideScalar(
                SelectorTensorOps.FrequencyPower(centered, frequencies), NumOps.FromDouble((double)n * n));
            maxPower = Engine.TensorMax(PreprocessingTensorOps.ColumnMax(power), NumOps.Zero);
        }

        // Normalize by variance to get proportion of variance explained
        var scores = SelectorTensorOps.ZeroUnless(varying,
            Engine.TensorDivide(Engine.TensorMultiplyScalar(maxPower, NumOps.FromDouble(2)), safeVariance));
        _seasonalityScores = scores.ToVector();

        // Select top features by seasonality score
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalityFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SeasonalityFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SeasonalityFS has not been fitted.");

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
