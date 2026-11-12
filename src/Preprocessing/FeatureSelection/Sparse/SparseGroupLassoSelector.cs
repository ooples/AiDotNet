using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Sparse Group Lasso Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines Lasso (individual feature sparsity) with Group Lasso (group sparsity)
/// to select features that are both individually and group-wise important.
/// </para>
/// <para><b>For Beginners:</b> Sometimes features naturally come in groups
/// (like one-hot encoded categories or polynomial terms). This method can select
/// entire groups together while also selecting individual features within groups,
/// giving you the best of both worlds.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SparseGroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _groupRatio;
    private readonly int[]? _groupIndices;

    // Proximal-gradient limits for the sparse group lasso fit
    private const int MaxIterations = 1000;
    private const double Tolerance = 1e-6;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double GroupRatio => _groupRatio;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SparseGroupLassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.01,
        double groupRatio = 0.5,
        int[]? groupIndices = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (groupRatio < 0 || groupRatio > 1)
            throw new ArgumentException("Group ratio must be between 0 and 1.", nameof(groupRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _groupRatio = groupRatio;
        _groupIndices = groupIndices;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SparseGroupLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Normalize features: centre and divide by the sample standard deviation plus 1e-10
        var centered = SelectorTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = Engine.TensorAddScalar(Engine.TensorSqrt(Engine.TensorDivideScalar(
            SelectorTensorOps.ColumnSumOfSquares(centered), NumOps.FromDouble(n - 1))), NumOps.FromDouble(1e-10));
        var x = Engine.TensorDivide(centered, Engine.Reshape(std, new[] { 1, p }));
        var y = SelectorTensorOps.CenteredColumn(target);

        // Create groups (default: each feature is its own group)
        var groups = CreateGroups(p);

        // Sparse group lasso (Simon et al., 2013):
        // (1/2n) ||y - X b||^2 + alpha (1 - groupRatio) ||b||_1 + alpha groupRatio sum_g sqrt(p_g) ||b_g||_2
        var coefficients = SparseRegressionOps.SparseGroupElasticNet(x, y, groups,
            _alpha * (1 - _groupRatio), 0, _alpha * _groupRatio, MaxIterations, Tolerance);
        _coefficients = coefficients.ToVector();

        // Features with non-zero coefficients (|b| > 1e-10) by decreasing |b|, padded with the largest of the rest
        var magnitude = Engine.TensorAbs(coefficients);
        int nonZero = PreprocessingTensorOps.CountNonZero(Engine.TensorGreaterThan(magnitude, NumOps.FromDouble(1e-10)));
        int numToSelect = Math.Min(_nFeaturesToSelect, Math.Max(nonZero, 1));
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(magnitude, numToSelect);
        IsFitted = true;
    }

    private int[] CreateGroups(int p)
    {
        if (_groupIndices is not null && _groupIndices.Length == p)
            return _groupIndices;

        // Default: each feature is its own group
        return Enumerable.Range(0, p).ToArray();
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
            throw new InvalidOperationException("SparseGroupLassoSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SparseGroupLassoSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseGroupLassoSelector has not been fitted.");

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
