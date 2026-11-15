using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonparametric;

/// <summary>
/// Kendall's Tau Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on Kendall's tau rank correlation coefficient with
/// the target, which measures ordinal association between features and target.
/// </para>
/// <para><b>For Beginners:</b> Kendall's tau measures how often pairs of data
/// points are in the same order for both feature and target. It's robust to
/// outliers and doesn't assume a linear relationship, just that both variables
/// tend to increase or decrease together.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KendallTauSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _tauScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? TauScores => _tauScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KendallTauSelector(
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
        throw new InvalidOperationException(
            "KendallTauSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);

        // Kendall's tau-b (scipy.stats.kendalltau's default): with s_x = sign(x_i - x_j) and s_y likewise,
        // tau_b = sum s_x s_y / sqrt(sum s_x^2 * sum s_y^2); the sums over all ordered pairs are twice the
        // sums over i < j, so the factors cancel. Ties count in neither concordant nor discordant pairs and
        // shrink the denominator.
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var targetSigns = SignDifferences(y);
        T targetUntied = engine.TensorSum(engine.TensorSquare(targetSigns));
        var sums = FeatureSelectionTensorOps.KernelWeightedSums(
            x,
            SignDifferences,
            new[] { targetSigns },
            new[] { PreprocessingTensorOps.Filled(new[] { n, n }, NumOps.One) });
        var concordance = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 0 }, new[] { p, 1 }), new[] { p });
        var featureUntied = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 1 }, new[] { p, 1 }), new[] { p });
        var denominator = engine.TensorMultiplyScalar(featureUntied, targetUntied);
        var defined = engine.TensorGreaterThan(denominator, NumOps.Zero);
        var safeDenominator = engine.TensorWhere(defined, denominator, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var tau = engine.TensorWhere(
            defined, engine.TensorDivide(concordance, engine.TensorSqrt(safeDenominator)), new Tensor<T>(new[] { p }));
        var scores = engine.TensorAbs(tau);
        _tauScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    private static Tensor<T> SignDifferences(Tensor<T> column)
    {
        var engine = AiDotNetEngine.Current;
        int n = column.Shape[0];
        return engine.TensorSign(engine.TensorSubtract(
            engine.TensorBroadcastTo(column, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(column), new[] { n, n })));
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
            throw new InvalidOperationException("KendallTauSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KendallTauSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTauSelector has not been fitted.");

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
