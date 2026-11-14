using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Autocorrelation-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Autocorrelation measures how a feature correlates with itself at different time lags.
/// Features with strong autocorrelation patterns often contain predictable structure
/// that can be useful for time series forecasting.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation asks: "How similar is today's value
/// to yesterday's, the day before, etc.?" Features with high autocorrelation have
/// patterns that repeat or persist over time, making them potentially valuable for
/// predicting future values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AutocorrelationFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;
    private readonly double _minAutocorrelation;

    private Vector<T>? _autocorrelationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public Vector<T>? AutocorrelationScores => _autocorrelationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AutocorrelationFS(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        double minAutocorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
        _minAutocorrelation = minAutocorrelation;
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

        // Compute |autocorrelation| (lag covariance over n - lag, divided by the variance) for each lag and take max
        var maxAutocorr = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        int effectiveMaxLag = Math.Min(_maxLag, n - 1);
        for (int lag = 1; lag <= effectiveMaxLag; lag++)
        {
            var covariance = Engine.TensorDivideScalar(
                SelectorTensorOps.LagProducts(centered, lag), PreprocessingTensorOps.CountAsScalar<T>(n - lag));
            maxAutocorr = Engine.TensorMax(maxAutocorr, Engine.TensorAbs(Engine.TensorDivide(covariance, safeVariance)));
        }

        var scores = SelectorTensorOps.ZeroUnless(varying, maxAutocorr);
        _autocorrelationScores = scores.ToVector();

        // Filter by minimum autocorrelation and select top features
        _selectedIndices = PreprocessingTensorOps.TopKAtLeastSorted(
            scores, NumOps.FromDouble(_minAutocorrelation), _nFeaturesToSelect);

        if (_selectedIndices.Length == 0 && p > 0)
        {
            // If no features pass threshold, take the top feature
            _selectedIndices = new[] { PreprocessingTensorOps.FirstArgMax(scores) };
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AutocorrelationFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationFS has not been fitted.");

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
