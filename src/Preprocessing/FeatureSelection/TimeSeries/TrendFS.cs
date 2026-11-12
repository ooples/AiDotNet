using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Trend-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Trend-based feature selection identifies features that exhibit significant
/// temporal trends (upward or downward movement over time). Features with strong
/// trends may be useful for long-term forecasting or detecting changes.
/// </para>
/// <para><b>For Beginners:</b> This method looks for features that are consistently
/// going up or down over time, like a stock price in a bull market. Features without
/// clear direction (just random fluctuation) score low. It's useful when you care
/// about the overall trajectory, not just short-term patterns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TrendFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _absoluteTrend;

    private Vector<T>? _trendScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool AbsoluteTrend => _absoluteTrend;
    public Vector<T>? TrendScores => _trendScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TrendFS(
        int nFeaturesToSelect = 10,
        bool absoluteTrend = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _absoluteTrend = absoluteTrend;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
        {
            _trendScores = new Vector<T>(p);
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        // Time variable (0, 1, 2, ..., n-1), centred
        double tMean = (n - 1) / 2.0;
        var time = Engine.Reshape(Engine.TensorSubtractScalar(Engine.TensorArange<T>(0, n, 1), NumOps.FromDouble(tMean)), new[] { n, 1 });
        T tVar = Engine.TensorSumOfSquares(time);

        // Compute linear regression slope (trend) of every feature on time
        var centered = SelectorTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var covariance = Engine.ReduceSum(Engine.TensorMultiply(centered, Engine.TensorBroadcastTo(time, new[] { n, p })), new[] { 0 }, false);
        var slope = NumOps.GreaterThan(tVar, NumOps.FromDouble(1e-10))
            ? Engine.TensorDivideScalar(covariance, tVar)
            : PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        // Compute R-squared to measure trend strength
        var residual = Engine.TensorSubtract(centered, Engine.TensorMultiply(
            Engine.TensorBroadcastTo(time, new[] { n, p }), Engine.TensorBroadcastTo(Engine.Reshape(slope, new[] { 1, p }), new[] { n, p })));
        var ssTotal = SelectorTensorOps.ColumnSumOfSquares(centered);
        var ssResidual = SelectorTensorOps.ColumnSumOfSquares(residual);
        var fitted = Engine.TensorGreaterThan(ssTotal, NumOps.FromDouble(1e-10));
        var safeTotal = Engine.TensorWhere(fitted, ssTotal, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var rSquared = SelectorTensorOps.ZeroUnless(fitted, PreprocessingTensorOps.OneMinus(Engine.TensorDivide(ssResidual, safeTotal)));

        // Score combines trend magnitude and fit quality
        var trendMagnitude = _absoluteTrend ? Engine.TensorAbs(slope) : slope;
        var scores = Engine.TensorMultiply(trendMagnitude, Engine.TensorSqrt(rSquared));
        _trendScores = scores.ToVector();

        // Select top features by trend score (by magnitude)
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
            _absoluteTrend ? scores : Engine.TensorAbs(scores), Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TrendFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendFS has not been fitted.");

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
