using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Two-Way ANOVA based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on two-way ANOVA, which analyzes the effect of
/// two categorical factors and their interaction on a continuous response.
/// </para>
/// <para><b>For Beginners:</b> Two-way ANOVA extends one-way ANOVA by considering
/// two grouping variables simultaneously. It can detect main effects of each factor
/// and interaction effects between them. Features with significant F-statistics
/// for either main effects or interactions are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TwoWayANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _fStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FStatistics => _fStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TwoWayANOVASelector(
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
            "TwoWayANOVASelector requires two factor vectors. Use Fit(Matrix<T> data, Vector<T> factor1, Vector<T> factor2) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> factor1, Vector<T> factor2)
    {
        if (data.Rows != factor1.Length || data.Rows != factor2.Length)
            throw new ArgumentException("Factor lengths must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        PreprocessingTensorOps.ClassLabels(factor1, out var levels1, out var index1);
        PreprocessingTensorOps.ClassLabels(factor2, out var levels2, out var index2);
        int l1 = levels1.Length;
        int l2 = levels2.Length;

        // Degrees of freedom
        int df1 = l1 - 1;
        int df2 = l2 - 1;
        int dfWithin = n - l1 * l2;

        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (dfWithin > 0 && df1 > 0 && df2 > 0)
        {
            var x = Tensor<T>.FromMatrix(data);
            var cells = new int[n];
            for (int i = 0; i < n; i++)
                cells[i] = index1[i] * l2 + index2[i];

            // Main-effect sums of squares from the marginal means and the error from the cell means
            var ss1 = SelectorTensorOps.ClassScatter(x, new Tensor<int>(index1, new[] { n }), l1, out _);
            var ss2 = SelectorTensorOps.ClassScatter(x, new Tensor<int>(index2, new[] { n }), l2, out _);
            SelectorTensorOps.ClassScatter(x, new Tensor<int>(cells, new[] { n }), l1 * l2, out var ssWithin);

            // F-statistics (combine main effects), 0 where the error mean square is not above 1e-10
            var msWithin = Engine.TensorDivideScalar(ssWithin, NumOps.FromDouble(dfWithin));
            var usable = Engine.TensorGreaterThan(msWithin, NumOps.FromDouble(1e-10));
            var safeMsWithin = Engine.TensorWhere(usable, msWithin, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
            var f1Stat = Engine.TensorDivide(Engine.TensorDivideScalar(ss1, NumOps.FromDouble(df1)), safeMsWithin);
            var f2Stat = Engine.TensorDivide(Engine.TensorDivideScalar(ss2, NumOps.FromDouble(df2)), safeMsWithin);

            // Use maximum F-statistic
            scores = SelectorTensorOps.ZeroUnless(usable, Engine.TensorMax(f1Stat, f2Stat));
        }

        _fStatistics = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TwoWayANOVASelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TwoWayANOVASelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TwoWayANOVASelector has not been fitted.");

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
