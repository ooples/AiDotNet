using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparsity;

/// <summary>
/// Group Lasso based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Group Lasso regularization, which selects or
/// removes entire groups of features together.
/// </para>
/// <para><b>For Beginners:</b> Group Lasso extends Lasso by treating features
/// in groups. Instead of selecting individual features, it selects entire groups
/// at once. This is useful when features naturally belong together, like one-hot
/// encoded categories or polynomial terms of the same variable.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GroupLassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly int[]? _groupAssignments;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GroupLassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.1,
        int[]? groupAssignments = null,
        int maxIterations = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _groupAssignments = groupAssignments;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GroupLassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Use provided groups or create individual groups
        var groups = _groupAssignments ?? Enumerable.Range(0, p).ToArray();
        if (groups.Length < p)
            throw new ArgumentException("Group assignments must have one entry per feature.");
        groups = groups.Take(p).ToArray();

        // Standardize (population std, 1 for a constant column)
        var x = SelectorTensorOps.StandardizeColumns(Tensor<T>.FromMatrix(data));
        var y = Engine.Reshape(Tensor<T>.FromVector(target), new[] { data.Rows, 1 });

        // Group lasso (Yuan and Lin, 2006): (1/2n) ||y - X b||^2 + alpha sum_g sqrt(p_g) ||b_g||_2
        var coefficients = SparseRegressionOps.SparseGroupElasticNet(x, y, groups, 0, 0, _alpha, _maxIterations, 1e-6);
        _coefficients = coefficients.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
            Engine.TensorAbs(coefficients), Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GroupLassoSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GroupLassoSelector has not been fitted.");

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
