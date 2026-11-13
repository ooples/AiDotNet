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

        // Time variable (0, 1, 2, ..., n-1), centred and divided by its norm sqrt(sum (t - mean)^2) =
        // sqrt(n (n^2 - 1) / 12), known in closed form: that sum of squares passes Half's 65504 from n = 59
        double tMean = (n - 1) / 2.0;
        double tNorm = Math.Sqrt(n * ((double)n * n - 1) / 12.0);
        var unitTime = Engine.Reshape(Engine.TensorDivideScalar(Engine.TensorSubtractScalar(
            Engine.TensorArange<T>(0, n, 1), NumOps.FromDouble(tMean)), NumOps.FromDouble(tNorm)), new[] { n, 1 });
        var unitTimeWide = Engine.TensorBroadcastTo(unitTime, new[] { n, p });

        // Linear regression slope (trend) of every feature on time: sum(x_c t_c) / sum(t_c^2) = projection / tNorm
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var projection = Engine.ReduceSum(Engine.TensorMultiply(centered, unitTimeWide), new[] { 0 }, false);
        var slope = Engine.TensorDivideScalar(projection, NumOps.FromDouble(tNorm));

        // Compute R-squared to measure trend strength (the fit slope t_c = projection unitTime)
        var residual = Engine.TensorSubtract(centered, Engine.TensorMultiply(
            unitTimeWide, Engine.TensorBroadcastTo(Engine.Reshape(projection, new[] { 1, p }), new[] { n, p })));
        var ssTotal = SelectorTensorOps.ColumnSumOfSquares(centered);
        var ssResidual = SelectorTensorOps.ColumnSumOfSquares(residual);
        var fitted = Engine.TensorGreaterThan(ssTotal, PreprocessingTensorOps.Tolerance<T>(1e-10));
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
