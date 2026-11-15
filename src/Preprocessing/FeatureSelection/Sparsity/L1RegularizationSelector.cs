using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparsity;

/// <summary>
/// L1 Regularization (Lasso-like) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using L1-regularized regression (Lasso), which
/// automatically sets unimportant feature coefficients to zero.
/// </para>
/// <para><b>For Beginners:</b> L1 regularization adds a penalty proportional
/// to the absolute value of coefficients. This penalty pushes weak coefficients
/// to exactly zero, effectively selecting important features. The stronger the
/// regularization, the fewer features are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class L1RegularizationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;

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

    public L1RegularizationSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.1,
        int maxIterations = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "L1RegularizationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Standardize features (population std, 1 for a constant column)
        var x = SelectorTensorOps.StandardizeColumns(Tensor<T>.FromMatrix(data));
        var y = Engine.Reshape(Tensor<T>.FromVector(target), new[] { data.Rows, 1 });

        // Lasso (scikit-learn Lasso objective (1/2n) ||y - X b||^2 + alpha ||b||_1)
        var coefficients = SparseRegressionOps.SparseGroupElasticNet(
            x, y, Enumerable.Range(0, p).ToArray(), _alpha, 0, 0, _maxIterations, 1e-6);
        _coefficients = coefficients.ToVector();

        // Select features with largest absolute coefficients
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
            throw new InvalidOperationException("L1RegularizationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("L1RegularizationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("L1RegularizationSelector has not been fitted.");

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
