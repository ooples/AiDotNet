using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// ANCOVA (Analysis of Covariance) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on ANCOVA, which combines ANOVA with regression
/// to control for confounding continuous covariates.
/// </para>
/// <para><b>For Beginners:</b> ANCOVA tests group differences while accounting
/// for the effect of one or more continuous covariates. This helps isolate the
/// true effect of categorical grouping by removing the influence of confounding
/// variables. Features with significant adjusted F-statistics are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ANCOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _covariateIndex;

    private Vector<T>? _adjustedFStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int CovariateIndex => _covariateIndex;
    public Vector<T>? AdjustedFStatistics => _adjustedFStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ANCOVASelector(
        int nFeaturesToSelect = 10,
        int covariateIndex = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _covariateIndex = covariateIndex;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANCOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (_covariateIndex < 0 || _covariateIndex >= p)
            throw new ArgumentException("Covariate index is out of range.");

        // One-way ANCOVA of each feature on the classes with the covariate (common within-class slope):
        // F = ((RSS_cov - RSS_full) / (k - 1)) / (RSS_full / (n - k - 1)), where RSS_full = Wyy - Wxy^2 / Wxx
        // uses the within-class sums of squares and RSS_cov = Tyy - Txy^2 / Txx the total ones.
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        int dfBetween = k - 1;
        int dfWithin = n - k - 1; // -1 for covariate

        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (dfWithin > 0 && dfBetween > 0)
        {
            var x = Tensor<T>.FromMatrix(data);
            var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
            var classMeans = SelectorTensorOps.GroupMeans(membership, x, out _);
            var within = Engine.TensorSubtract(x, Engine.TensorIndexSelect(classMeans, new Tensor<int>(classIndex, new[] { n }), 0));
            var total = FilterStatisticTensorOps.CenterColumns(x);

            var rssFull = ResidualAfterCovariate(within);
            var rssCovariate = ResidualAfterCovariate(total);
            var msBetween = Engine.TensorDivideScalar(
                Engine.TensorMax(Engine.TensorSubtract(rssCovariate, rssFull), NumOps.Zero), NumOps.FromDouble(dfBetween));
            var msWithin = Engine.TensorDivideScalar(rssFull, NumOps.FromDouble(dfWithin));
            scores = SelectorTensorOps.SeparationRatio(msBetween, msWithin, 1e-10);
        }

        // The covariate itself scores 0 and is never selected
        // (a one-hot position mask: compared as T, the column indices past 2048 are not exact in Half and
        // would match the covariate index too)
        var notCovariate = PreprocessingTensorOps.Not(Engine.Reshape(Engine.TensorOneHot<T>(
            new Tensor<int>(new[] { _covariateIndex }, new[] { 1 }), p), new[] { p }));
        scores = SelectorTensorOps.ZeroUnless(notCovariate, scores);
        _adjustedFStatistics = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p - 1); // Exclude covariate
        _selectedIndices = PreprocessingTensorOps.StableOrder(scores, descending: true)
            .Where(j => j != _covariateIndex)
            .Take(numToSelect)
            .OrderBy(j => j)
            .ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// Residual sum of squares of every column of a centred [n, p] tensor after a no-intercept regression on its
    /// covariate column: <c>Syy - Sxy^2 / Sxx</c>, or <c>Syy</c> when <c>Sxx</c> is not above 1e-10.
    /// </summary>
    private Tensor<T> ResidualAfterCovariate(Tensor<T> centered)
    {
        int n = centered.Shape[0];
        int p = centered.Shape[1];
        var covariate = Engine.TensorSliceAxis(centered, 1, _covariateIndex);
        var syy = SelectorTensorOps.ColumnSumOfSquares(centered);
        T sxx = Engine.TensorSumOfSquares(covariate);
        if (!NumOps.GreaterThan(sxx, NumOps.FromDouble(1e-10)))
            return syy;

        var sxy = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(centered),
            Engine.Reshape(covariate, new[] { n, 1 })), new[] { p });
        return Engine.TensorMax(Engine.TensorSubtract(syy,
            Engine.TensorDivideScalar(Engine.TensorMultiply(sxy, sxy), sxx)), NumOps.Zero);
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
            throw new InvalidOperationException("ANCOVASelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ANCOVASelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANCOVASelector has not been fitted.");

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
