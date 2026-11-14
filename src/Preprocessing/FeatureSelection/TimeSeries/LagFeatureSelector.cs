using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Lag-based Feature Selection for Time Series.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their lagged correlations with the target,
/// identifying features that have predictive power at different time lags.
/// </para>
/// <para><b>For Beginners:</b> In time series, past values often predict future
/// values. This selector finds features whose past values (lags) are strongly
/// correlated with the target, helping you identify which features are most
/// useful for forecasting.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LagFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

    private Vector<T>? _bestLagCorrelations;
    private int[]? _bestLags;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public Vector<T>? BestLagCorrelations => _bestLagCorrelations;
    public int[]? BestLags => _bestLags;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LagFeatureSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LagFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var bestCorr = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        // (lags kept as int: stored in a narrow T such as Half, a lag past 2048 rounds to a neighbouring lag)
        var bestLag = new Tensor<int>(new[] { p });

        // Correlation of x at time t with y at time t + lag; keep the first lag with the largest |r|
        for (int lag = 0; lag <= Math.Min(_maxLag, n - 10); lag++)
        {
            var corr = ComputeLaggedCorrelation(x, y, lag, n);
            var better = Engine.TensorGreaterThan(Engine.TensorAbs(corr), Engine.TensorAbs(bestCorr));
            bestCorr = Engine.TensorWhere(better, corr, bestCorr);
            bestLag = Engine.TensorWhere(better.Cast<int>(), PreprocessingTensorOps.Filled(new[] { p }, lag), bestLag);
        }

        var scores = Engine.TensorAbs(bestCorr);
        _bestLagCorrelations = scores.ToVector();
        _bestLags = bestLag.AsSpan().ToArray();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// Pearson r of every feature at time t with the target at time t + lag over the overlapping rows, 0 below two
    /// rows or for a constant feature or target.
    /// </summary>
    private Tensor<T> ComputeLaggedCorrelation(Tensor<T> x, Tensor<T> y, int lag, int n)
    {
        int effectiveN = n - lag;
        int p = x.Shape[1];
        if (effectiveN < 2) return PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        return PreprocessingTensorOps.TargetCorrelations(
            Engine.TensorSlice(x, new[] { 0, 0 }, new[] { effectiveN, p }),
            Engine.TensorSlice(y, new[] { lag }, new[] { effectiveN }));
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
            throw new InvalidOperationException("LagFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LagFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LagFeatureSelector has not been fitted.");

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
