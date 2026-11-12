using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Variance Inflation Factor (VIF) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with high multicollinearity by computing the Variance
/// Inflation Factor for each feature and removing those above a threshold.
/// </para>
/// <para><b>For Beginners:</b> VIF measures how much a feature can be predicted
/// by other features. A high VIF means that feature is highly correlated with
/// others (redundant). We remove features with high VIF to keep only the
/// independent, non-redundant ones.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class VarianceInflationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFeatureSelector<T>
{
    private readonly double _vifThreshold;

    private Vector<T>? _vifScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double VIFThreshold => _vifThreshold;
    public Vector<T>? VIFScores => _vifScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public VarianceInflationSelector(
        double vifThreshold = 5.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (vifThreshold <= 1)
            throw new ArgumentException("VIF threshold must be greater than 1.", nameof(vifThreshold));

        _vifThreshold = vifThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;

        // Standardize features (sample std + 1e-10)
        var x = Tensor<T>.FromMatrix(data);
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var stds = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(
                SelectorTensorOps.ColumnSumOfSquares(centered), NumOps.FromDouble(n - 1))),
            NumOps.FromDouble(1e-10));
        var z = engine.TensorDivide(centered, engine.Reshape(stds, new[] { 1, p }));

        var vifScores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var selected = new List<int>(Enumerable.Range(0, p));
        T ridge = NumOps.FromDouble(1e-6);
        T threshold = NumOps.FromDouble(_vifThreshold);

        // Iteratively remove features with highest VIF above threshold
        bool changed = true;
        while (changed && selected.Count > 1)
        {
            changed = false;

            // VIF of every remaining feature, regressed (ridge 1e-6, no intercept) on the others
            var columns = new Tensor<int>(selected.ToArray(), new[] { selected.Count });
            var vif = ComputeVIF(engine.TensorIndexSelect(z, columns, 1), ridge);
            vifScores = engine.TensorIndexCopy(vifScores, 0, columns, vif);

            // Remove feature with highest VIF if above threshold
            int maxPosition = PreprocessingTensorOps.FirstArgMax(vif);
            if (PreprocessingTensorOps.IsAbove(engine.TensorMaxValue(vif), threshold))
            {
                selected.RemoveAt(maxPosition);
                changed = true;
            }
        }

        _vifScores = vifScores.ToVector();
        _selectedIndices = selected.OrderBy(index => index).ToArray();
        IsFitted = true;
    }

    /// <summary>
    /// Variance inflation factor of each column of <paramref name="z"/> ([n, k], k &gt;= 2): 1 / (1 - R^2) for the
    /// ridge regression (penalty <paramref name="ridge"/> on the predictors, no intercept) of the column on the
    /// other k - 1, with R^2 clamped to [0, 0.9999].
    /// </summary>
    /// <remarks>
    /// With H = (Z'Z + ridge I)^-1, block inversion gives the ridge coefficients of column j as
    /// beta = -H[-j, j] / H[j, j] and its residual sum of squares as
    /// 1 / H[j, j] - ridge - ridge * |beta|^2, so one k x k inverse yields every column's fit.
    /// </remarks>
    private Tensor<T> ComputeVIF(Tensor<T> z, T ridge)
    {
        var engine = Engine;
        int k = z.Shape[1];
        var gram = engine.TensorAdd(
            engine.TensorMatMul(engine.TensorTranspose(z), z), engine.TensorMultiplyScalar(engine.TensorEye<T>(k), ridge));
        var inverse = LinalgDispatch<T>.Inv(gram);
        var diagonal = engine.TensorDiagonal(inverse);
        var diagonalSquared = engine.TensorMultiply(diagonal, diagonal);
        var offDiagonalSquared = engine.TensorSubtract(
            engine.ReduceSum(engine.TensorMultiply(inverse, inverse), new[] { 0 }, false), diagonalSquared);
        var betaNormSquared = engine.TensorDivide(offDiagonalSquared, diagonalSquared);
        var ssRes = engine.TensorSubtract(
            engine.TensorSubtractScalar(engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { k }, NumOps.One), diagonal), ridge),
            engine.TensorMultiplyScalar(betaNormSquared, ridge));
        var ssTot = SelectorTensorOps.ColumnSumOfSquares(FilterStatisticTensorOps.CenterColumns(z));

        var rSquared = SelectorTensorOps.ZeroUnless(
            engine.TensorGreaterThan(ssTot, NumOps.FromDouble(1e-10)),
            engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { k }, NumOps.One), engine.TensorDivide(ssRes, ssTot)));
        rSquared = engine.TensorClamp(rSquared, NumOps.Zero, NumOps.FromDouble(0.9999));
        return engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { k }, NumOps.One),
            PreprocessingTensorOps.OneMinus(rSquared));
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceInflationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("VarianceInflationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VarianceInflationSelector has not been fitted.");

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
