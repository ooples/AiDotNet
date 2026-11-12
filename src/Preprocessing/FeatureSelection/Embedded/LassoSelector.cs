using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Lasso (L1) regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses L1-regularized linear regression (Lasso) which drives some coefficients
/// exactly to zero, performing automatic feature selection. The regularization
/// parameter controls the sparsity of the solution.
/// </para>
/// <para><b>For Beginners:</b> Lasso is like a strict budget for your model. It
/// forces the model to "spend" wisely on features, often setting unimportant
/// feature weights to exactly zero. Features with non-zero weights after this
/// "budgeting" process are considered important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize data (population std; a near-zero std is replaced by 1) and center the target
        var X = FeatureSelectionTensorOps.Standardize(Tensor<T>.FromMatrix(data), out var stds);
        var y = FeatureSelectionTensorOps.Center(Tensor<T>.FromVector(target));

        // Coordinate descent for Lasso: soft-threshold rho at alpha * n and divide by ||x_j||^2
        // (a column with ||x_j||^2 not above 1e-10 keeps its zero coefficient)
        var Xj2 = engine.ReduceSum(engine.TensorSquare(X), new[] { 0 }, false);
        T threshold = NumOps.FromDouble(_alpha * n);
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var none = new Tensor<T>(new[] { 1 });
        var beta = FeatureSelectionTensorOps.CoordinateDescent(X, y, _maxIterations, _tolerance, (rho, j) =>
        {
            var squaredNorm = engine.TensorSlice(Xj2, new[] { j }, new[] { 1 });
            return engine.TensorWhere(engine.TensorGreaterThan(squaredNorm, floor),
                engine.TensorDivide(FeatureSelectionTensorOps.SoftThreshold(rho, threshold), squaredNorm), none);
        }, out _);

        // Un-standardize coefficients
        var coefficients = engine.TensorDivide(beta, stds);
        _coefficients = coefficients.ToVector();

        // Select top features by coefficient magnitude
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(coefficients), nToSelect);

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
            throw new InvalidOperationException("LassoSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LassoSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LassoSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
