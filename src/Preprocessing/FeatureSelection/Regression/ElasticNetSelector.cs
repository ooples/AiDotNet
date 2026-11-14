using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Regression;

/// <summary>
/// Elastic Net Regularization based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Elastic Net regularization (combination of L1 and L2 penalties) to select
/// features by identifying those with non-zero coefficients after regularized regression.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net combines Lasso (which sets unimportant
/// features to zero) and Ridge (which handles correlated features well). Features
/// that survive this double penalty are likely important.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ElasticNetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ElasticNetSelector(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double l1Ratio = 0.5,
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
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ElasticNetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;

        // Standardize (population std; a constant column is only centred) and centre the target
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);
        var safeStd = engine.TensorWhere(engine.TensorLessThan(std, NumOps.FromDouble(1e-10)), PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), std);
        var X = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));
        var yColumn = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var y = engine.TensorSubtract(yColumn, engine.ReduceMean(yColumn, new[] { 0 }, true));

        // Coordinate descent for Elastic Net, (1/2n)||y - Xb||^2 + alpha * l1Ratio * ||b||_1
        // + alpha * (1 - l1Ratio) / 2 * ||b||^2, with rho_j = (X'y)_j / n - sum_{k != j} (X'X)_jk b_k / n
        var nT = FeatureSelectionTensorOps.Count<T>(n);
        var gram = engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(X), X), nT); // [p, p]
        var correlation = engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(X), y), nT); // [p, 1]
        var eye = engine.TensorEye<T>(p);
        T l1Penalty = NumOps.FromDouble(_alpha * _l1Ratio);
        T negativeL1Penalty = NumOps.Negate(l1Penalty);
        T shrinkage = NumOps.FromDouble(1 + _alpha * (1 - _l1Ratio));
        T tolerance = NumOps.FromDouble(_tolerance);

        var coefficients = new Tensor<T>(new[] { p, 1 });
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var maxChange = new Tensor<T>(new[] { 1, 1 });
            for (int j = 0; j < p; j++)
            {
                var oldCoef = engine.TensorSlice(coefficients, new[] { j, 0 }, new[] { 1, 1 });
                var fit = engine.TensorMatMul(engine.TensorSlice(gram, new[] { j, 0 }, new[] { 1, p }), coefficients);
                var own = engine.TensorMultiply(engine.TensorSlice(gram, new[] { j, j }, new[] { 1, 1 }), oldCoef);
                var rhoJ = engine.TensorAdd(engine.TensorSubtract(engine.TensorSlice(correlation, new[] { j, 0 }, new[] { 1, 1 }), fit), own);

                // Soft thresholding: S(rho, l1) / (1 + l2), with S(z, l) = z - clamp(z, -l, l)
                var newCoef = engine.TensorDivideScalar(
                    engine.TensorSubtract(rhoJ, engine.TensorClamp(rhoJ, negativeL1Penalty, l1Penalty)), shrinkage);
                var change = engine.TensorSubtract(newCoef, oldCoef);
                coefficients = engine.TensorAdd(coefficients,
                    engine.TensorMultiply(engine.TensorSlice(eye, new[] { 0, j }, new[] { p, 1 }), engine.TensorBroadcastTo(change, new[] { p, 1 })));
                maxChange = engine.TensorMax(maxChange, engine.TensorAbs(change));
            }

            if (engine.TensorCountNonzero(engine.TensorLessThan(maxChange, tolerance)) > 0)
                break;
        }

        var flat = engine.Reshape(coefficients, new[] { p });
        _coefficients = flat.ToVector();

        // Select features with largest coefficients
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(flat), numToSelect);

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
            throw new InvalidOperationException("ElasticNetSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ElasticNetSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetSelector has not been fitted.");

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
