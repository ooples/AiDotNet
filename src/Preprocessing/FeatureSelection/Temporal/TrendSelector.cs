using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Trend based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the strength of their temporal trends,
/// identifying features with clear increasing or decreasing patterns.
/// </para>
/// <para><b>For Beginners:</b> Some features have trends - they consistently
/// go up or down over time. This selector finds features with strong trends,
/// which can be useful for forecasting applications.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TrendSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _trendStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TrendStrengths => _trendStrengths;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TrendSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var strengths = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        // Time indices, centred; tVar = sum (t - tMean)^2
        var t = engine.TensorArange<T>(0, n, 1);
        var tCentered = engine.Reshape(engine.TensorSubtractScalar(t, engine.TensorMean(t)), new[] { n, 1 });
        T tVar = engine.TensorSum(engine.TensorMultiply(tCentered, tCentered));
        if (n > 0 && !NumOps.LessThan(tVar, NumOps.FromDouble(1e-10)))
        {
            var yCentered = SelectorTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
            var yVar = SelectorTensorOps.ColumnSumOfSquares(yCentered);

            // Linear regression slope and the R-squared of the fitted trend line
            var slope = engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorMultiply(yCentered, tCentered), new[] { 0 }, false), tVar);
            var residual = engine.TensorSubtract(yCentered,
                engine.TensorMultiply(tCentered, engine.Reshape(slope, new[] { 1, p })));
            var ssRes = SelectorTensorOps.ColumnSumOfSquares(residual);
            var r2 = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p }, NumOps.One),
                engine.TensorDivide(ssRes, yVar));

            strengths = SelectorTensorOps.ZeroUnless(
                PreprocessingTensorOps.AtLeast(yVar, NumOps.FromDouble(1e-10)), engine.TensorMax(r2, NumOps.Zero));
        }

        _trendStrengths = strengths.ToVector();
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
            throw new InvalidOperationException("TrendSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TrendSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendSelector has not been fitted.");

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
