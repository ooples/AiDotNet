using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Lagged correlation selector for time series feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Computes cross-correlation between features and target at various lags
/// to find features that predict future target values.
/// </para>
/// <para><b>For Beginners:</b> This finds features whose current values
/// predict future values of the target. A feature might correlate with
/// the target 3 time steps later, making it useful for forecasting.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LaggedCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minLag;
    private readonly int _maxLag;

    private Vector<T>? _correlationScores;
    private int[]? _bestLags;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? CorrelationScores => _correlationScores;
    public int[]? BestLags => _bestLags;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LaggedCorrelationSelector(
        int nFeaturesToSelect = 10,
        int minLag = 0,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minLag < 0)
            throw new ArgumentException("Min lag must be non-negative.", nameof(minLag));
        if (maxLag < minLag)
            throw new ArgumentException("Max lag must be >= min lag.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minLag = minLag;
        _maxLag = maxLag;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LaggedCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int maxLag = Math.Min(_maxLag, n / 3);

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var maxCorr = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var bestLag = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        for (int lag = _minLag; lag <= maxLag; lag++)
        {
            // Correlate x[0:n-lag] with y[lag:n] (0 for a constant feature or target)
            int nValid = n - lag;
            if (nValid < 3) continue;

            var corr = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
                Engine.TensorSlice(x, new[] { 0, 0 }, new[] { nValid, p }),
                Engine.TensorSlice(y, new[] { lag }, new[] { nValid })));
            var better = Engine.TensorGreaterThan(corr, maxCorr);
            maxCorr = Engine.TensorWhere(better, corr, maxCorr);
            bestLag = Engine.TensorWhere(better,
                PreprocessingTensorOps.Filled(new[] { p }, PreprocessingTensorOps.CountAsScalar<T>(lag)), bestLag);
        }

        _correlationScores = maxCorr.ToVector();
        _bestLags = bestLag.Cast<int>().AsSpan().ToArray();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(maxCorr, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaggedCorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LaggedCorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaggedCorrelationSelector has not been fitted.");

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
