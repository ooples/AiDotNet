using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Granger Causality-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Granger causality tests to select features that help predict the target
/// beyond what the target's own history can predict.
/// </para>
/// <para><b>For Beginners:</b> Granger causality asks: does knowing a feature's
/// past values help predict the target better than just knowing the target's own
/// past? Features that "Granger-cause" the target have genuine predictive value
/// for forecasting.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GrangerCausalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

    private Vector<T>? _grangerFScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public Vector<T>? GrangerFScores => _grangerFScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GrangerCausalitySelector(
        int nFeaturesToSelect = 10,
        int maxLag = 5,
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
            "GrangerCausalitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int effectiveN = n - _maxLag;

        if (effectiveN < 0)
            throw new ArgumentException("Data must have at least maxLag rows.", nameof(data));

        // Granger (1969) causality F-test as statsmodels grangercausalitytests ("ssr_ftest"): OLS with a constant of
        // y_t on its own lags (restricted) and on its own lags plus the feature's lags (unrestricted);
        // F = ((RSS_r - RSS_u) / lag) / (RSS_u / (nobs - 2 lag - 1)) with nobs = n - lag.
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        int dfUnrestricted = effectiveN - 2 * _maxLag - 1;
        if (effectiveN > 0 && p > 0)
        {
            var x = Tensor<T>.FromMatrix(data);
            var y = Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
            var response = Engine.TensorSlice(y, new[] { _maxLag, 0 }, new[] { effectiveN, 1 });

            // Compute restricted model RSS (autoregressive model on Y only)
            var restrictedDesign = Engine.TensorConcatenate(new[]
            {
                PreprocessingTensorOps.Filled(new[] { effectiveN, 1 }, NumOps.One), LagColumns(y, effectiveN)
            }, 1);
            T rssRestricted = ComputeRSS(restrictedDesign, response).GetFlat(0);

            // For each feature, compute unrestricted model RSS (Y lags + X_j lags)
            var rssUnrestricted = new Tensor<T>[p];
            for (int j = 0; j < p; j++)
            {
                var featureLags = LagColumns(Engine.Reshape(Engine.TensorSliceAxis(x, 1, j), new[] { n, 1 }), effectiveN);
                rssUnrestricted[j] = ComputeRSS(Engine.TensorConcatenate(new[] { restrictedDesign, featureLags }, 1), response);
            }

            if (dfUnrestricted > 0)
            {
                var rss = Engine.TensorConcatenate(rssUnrestricted, 0);
                var usable = Engine.TensorGreaterThan(rss, PreprocessingTensorOps.Tolerance<T>(1e-10));
                var safeRss = Engine.TensorWhere(usable, rss, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
                var numerator = Engine.TensorDivideScalar(
                    Engine.TensorAddScalar(Engine.TensorNegate(rss), rssRestricted), NumOps.FromDouble(_maxLag));
                var fStat = Engine.TensorDivide(numerator,
                    Engine.TensorDivideScalar(safeRss, NumOps.FromDouble(dfUnrestricted)));
                scores = SelectorTensorOps.ZeroUnless(usable, Engine.TensorMax(fStat, NumOps.Zero));
            }
        }

        _grangerFScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// The lags 1 .. maxLag of an [n, 1] series aligned with the targets <c>t = maxLag .. n - 1</c>:
    /// column l holds rows <c>maxLag - l - 1 .. maxLag - l - 2 + effectiveN</c>, as [effectiveN, maxLag].
    /// </summary>
    private Tensor<T> LagColumns(Tensor<T> series, int effectiveN)
    {
        var columns = new Tensor<T>[_maxLag];
        for (int l = 0; l < _maxLag; l++)
            columns[l] = Engine.TensorSlice(series, new[] { _maxLag - l - 1, 0 }, new[] { effectiveN, 1 });
        return Engine.TensorConcatenate(columns, 1);
    }

    /// <summary>
    /// Residual sum of squares of the least-squares fit of <paramref name="y"/> ([m, 1]) on
    /// <paramref name="design"/> ([m, k]) as a [1] tensor: <c>y'y - sum_i (u_i' y)^2</c> over the left singular
    /// vectors whose singular value is above 1e-10 times the largest (a pseudo-inverse fit, so collinear columns
    /// such as a constant feature are handled like numpy.linalg.lstsq).
    /// </summary>
    private Tensor<T> ComputeRSS(Tensor<T> design, Tensor<T> y)
    {
        var (u, s, _) = LinalgDispatch<T>.Svd(design, false);
        var kept = Engine.TensorGreaterThan(s,
            NumOps.Multiply(Engine.TensorMaxValue(s), PreprocessingTensorOps.Tolerance<T>(1e-10)));
        var projections = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(u), y), new[] { s.Length });
        T explained = Engine.TensorSum(Engine.TensorMultiply(kept, Engine.TensorMultiply(projections, projections)));
        T rss = NumOps.Subtract(Engine.TensorSumOfSquares(y), explained);
        return PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.GreaterThan(rss, NumOps.Zero) ? rss : NumOps.Zero);
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
            throw new InvalidOperationException("GrangerCausalitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GrangerCausalitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GrangerCausalitySelector has not been fitted.");

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
