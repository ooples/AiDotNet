using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Pearson Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Pearson correlation coefficient with the target,
/// measuring the linear relationship strength between feature and target.
/// </para>
/// <para><b>For Beginners:</b> Pearson correlation measures how closely two variables
/// follow a straight line together. Values near 1 or -1 indicate strong linear
/// relationships, while values near 0 indicate no linear relationship.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PearsonCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Correlations => _correlations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PearsonCorrelationSelector(
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
            "PearsonCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // |Pearson r| from population moments; 0 where either standard deviation is below 1e-10
        var xc = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var yc = SelectorTensorOps.CenteredColumn(target);
        T count = NumOps.FromDouble(n);
        T eps = PreprocessingTensorOps.Tolerance<T>(1e-10);
        T yStd = NumOps.Sqrt(NumOps.Divide(Engine.TensorSumOfSquares(yc), count));
        var xStd = Engine.TensorSqrt(Engine.TensorDivideScalar(SelectorTensorOps.ColumnSumOfSquares(xc), count));
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (!NumOps.LessThan(yStd, eps))
        {
            var covariance = Engine.TensorDivideScalar(
                Engine.ReduceSum(Engine.TensorMultiply(xc, yc), new[] { 0 }, false), count);
            var usable = PreprocessingTensorOps.AtLeast(xStd, eps);
            var safeStd = Engine.TensorWhere(usable, xStd, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
            scores = SelectorTensorOps.ZeroUnless(usable,
                Engine.TensorAbs(Engine.TensorDivide(covariance, Engine.TensorMultiplyScalar(safeStd, yStd))));
        }

        _correlations = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
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
            throw new InvalidOperationException("PearsonCorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PearsonCorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PearsonCorrelationSelector has not been fitted.");

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
