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
/// Selects features based on their autocorrelation properties. Features with
/// significant autocorrelation contain temporal patterns useful for prediction.
/// </para>
/// <para><b>For Beginners:</b> Autocorrelation measures how much a value at one
/// time relates to values at other times. Features with strong autocorrelation
/// have predictable patterns over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AutocorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;
    private readonly double _significanceThreshold;

    private Vector<T>? _autocorrelationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AutocorrelationScores => _autocorrelationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AutocorrelationSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        double significanceThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
        _significanceThreshold = significanceThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int maxLag = Math.Min(_maxLag, n / 3);

        var engine = Engine;
        var centered = FeatureSelectionTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var variance = engine.TensorDivideScalar(
            FeatureSelectionTensorOps.ColumnSumOfSquares(centered), NumOps.FromDouble(n));

        // Largest |ACF| over the lags, and how many lags pass the approximate 95% significance bound
        T criticalValue = NumOps.FromDouble(1.96 / Math.Sqrt(n));
        var maxAcf = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var significantLags = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (maxLag >= 1)
        {
            // |ACF(lag)| = |sum_{i<n-lag} c_i c_{i+lag}| / ((n - lag) * variance), all lags as rows of [maxLag, p]
            var products = new List<Tensor<T>>();
            for (int lag = 1; lag <= maxLag; lag++)
                products.Add(engine.Reshape(FeatureSelectionTensorOps.LagProducts(centered, lag), new[] { 1, p }));
            var counts = engine.Reshape(
                engine.TensorAddScalar(engine.TensorNegate(engine.TensorArange<T>(1, maxLag + 1, 1)), NumOps.FromDouble(n)),
                new[] { maxLag, 1 });
            var acf = engine.TensorAbs(engine.TensorDivide(engine.Concat(products, 0),
                engine.TensorMultiply(counts, engine.Reshape(variance, new[] { 1, p }))));
            maxAcf = engine.TensorMax(maxAcf, engine.ReduceMax(acf, new[] { 0 }, false));
            significantLags = engine.ReduceSum(engine.TensorGreaterThan(acf, criticalValue), new[] { 0 }, false);
        }

        // Score based on max ACF and number of significant lags (near-constant columns score 0)
        var scores = FeatureSelectionTensorOps.ZeroUnless(
            PreprocessingTensorOps.AtLeast(variance, NumOps.FromDouble(1e-10)),
            engine.TensorMultiply(maxAcf,
                engine.TensorAddScalar(engine.TensorMultiplyScalar(significantLags, NumOps.FromDouble(0.1)), NumOps.One)));
        _autocorrelationScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
