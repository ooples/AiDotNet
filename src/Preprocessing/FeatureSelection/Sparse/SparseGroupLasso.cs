using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Sparse Group Lasso for feature selection with grouped structure.
/// </summary>
/// <remarks>
/// <para>
/// Sparse Group Lasso combines L1 (Lasso) and L2 (Group Lasso) penalties to select
/// both entire groups of features and individual features within groups. This is useful
/// when features have a natural grouping structure.
/// </para>
/// <para><b>For Beginners:</b> Imagine features organized into categories (e.g., color
/// features, shape features). Sometimes you want to select entire categories, other times
/// specific features within categories. Sparse Group Lasso does both: it can eliminate
/// entire groups or pick individual features within groups.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SparseGroupLasso<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int[] _groups;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public int[] Groups => _groups;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SparseGroupLasso(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double l1Ratio = 0.5,
        int[]? groups = null,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _groups = groups ?? [];
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SparseGroupLasso requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Create default groups if not provided (each feature is its own group)
        var groups = _groups.Length == p ? _groups : Enumerable.Range(0, p).ToArray();

        // Standardize features (population std, 1 for a constant column) and centre the target
        var x = SelectorTensorOps.StandardizeColumns(Tensor<T>.FromMatrix(data));
        var y = SelectorTensorOps.CenteredColumn(target);

        // Sparse group lasso (Simon et al., 2013):
        // (1/2n) ||y - X b||^2 + alpha l1Ratio ||b||_1 + alpha (1 - l1Ratio) sum_g sqrt(p_g) ||b_g||_2
        var coefficients = SparseRegressionOps.SparseGroupElasticNet(x, y, groups,
            _alpha * _l1Ratio, 0, _alpha * (1 - _l1Ratio), _maxIterations, _tolerance);
        _coefficients = coefficients.ToVector();

        // Select features by absolute coefficient magnitude
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
            throw new InvalidOperationException("SparseGroupLasso has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SparseGroupLasso does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SparseGroupLasso has not been fitted.");

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
