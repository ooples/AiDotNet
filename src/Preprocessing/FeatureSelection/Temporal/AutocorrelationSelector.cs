using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Temporal;

/// <summary>
/// Autocorrelation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their autocorrelation properties, identifying features
/// with meaningful temporal patterns.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation measures how a signal relates to
/// delayed versions of itself. Features with strong autocorrelation have patterns
/// that repeat over time, which can be useful for prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AutocorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

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

    public AutocorrelationSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        int effectiveMaxLag = Math.Min(_maxLag, n / 3);
        var centered = FeatureSelectionTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var variance = FeatureSelectionTensorOps.ColumnSumOfSquares(centered);

        // Largest absolute autocorrelation over lags 1 .. effectiveMaxLag
        var maxAC = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        for (int lag = 1; lag <= effectiveMaxLag; lag++)
        {
            var acf = engine.TensorDivide(FeatureSelectionTensorOps.LagProducts(centered, lag), variance);
            maxAC = engine.TensorMax(maxAC, engine.TensorAbs(acf));
        }

        // Near-constant columns score 0
        var scores = FeatureSelectionTensorOps.ZeroUnless(
            PreprocessingTensorOps.AtLeast(variance, NumOps.FromDouble(1e-10)), maxAC);
        _autocorrelationScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("AutocorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AutocorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutocorrelationSelector has not been fitted.");

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
