using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Fused Lasso (Total Variation) for feature selection with smoothness constraints.
/// </summary>
/// <remarks>
/// <para>
/// Fused Lasso adds a penalty on the differences between consecutive coefficients,
/// encouraging nearby features to have similar coefficients. This is useful when
/// features have a natural ordering (e.g., spectral data, time series).
/// </para>
/// <para><b>For Beginners:</b> When features are ordered (like wavelengths or time points),
/// you often expect nearby features to behave similarly. Fused Lasso encourages smooth
/// coefficient patterns while still allowing for sparse solutions where coefficients
/// jump abruptly only at important boundaries.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FusedLasso<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _fusedPenalty;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double FusedPenalty => _fusedPenalty;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FusedLasso(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double fusedPenalty = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _fusedPenalty = fusedPenalty;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FusedLasso requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Standardize features (population std, 1 for a constant column) and centre the target
        var x = SelectorTensorOps.StandardizeColumns(Tensor<T>.FromMatrix(data));
        var y = SelectorTensorOps.CenteredColumn(target);

        // Fused lasso (Tibshirani et al., 2005): minimise 1/2 ||y - X b||^2 + alpha ||b||_1
        // + fusedPenalty sum_j |b_j+1 - b_j| by scaled ADMM (Boyd et al., 2011, section 6.4.1) on z = D b,
        // where D stacks the identity and the first-difference operator.
        var difference = DifferenceOperator(p);
        var dt = Engine.TensorTranspose(difference);
        T rho = NumOps.One; // ADMM penalty parameter
        var xt = Engine.TensorTranspose(x);
        var system = Engine.TensorAdd(Engine.TensorMatMul(xt, x),
            Engine.TensorMultiplyScalar(Engine.TensorMatMul(dt, difference), rho));
        var inverse = LinalgDispatch<T>.Inv(system);
        var xty = Engine.TensorMatMul(xt, y);

        int rows = difference.Shape[0];
        var thresholds = PreprocessingTensorOps.Filled(new[] { p, 1 }, NumOps.Divide(NumOps.FromDouble(_alpha), rho));
        if (rows > p)
        {
            thresholds = Engine.TensorConcatenate(new[]
            {
                thresholds,
                PreprocessingTensorOps.Filled(new[] { rows - p, 1 }, NumOps.Divide(NumOps.FromDouble(_fusedPenalty), rho))
            }, 0);
        }
        var z = new Tensor<T>(new[] { rows, 1 });
        var u = new Tensor<T>(new[] { rows, 1 });
        var coefficients = new Tensor<T>(new[] { p, 1 });
        T tolerance = NumOps.FromDouble(_tolerance);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var oldCoef = coefficients;

            // b = (X'X + rho D'D)^-1 (X'y + rho D'(z - u))
            coefficients = Engine.TensorMatMul(inverse, Engine.TensorAdd(xty,
                Engine.TensorMultiplyScalar(Engine.TensorMatMul(dt, Engine.TensorSubtract(z, u)), rho)));

            // z = soft-threshold(D b + u): alpha / rho on the coefficients, fusedPenalty / rho on the differences
            var db = Engine.TensorMatMul(difference, coefficients);
            var shifted = Engine.TensorAdd(db, u);
            z = Engine.TensorMultiply(Engine.TensorSign(shifted),
                Engine.TensorMax(Engine.TensorSubtract(Engine.TensorAbs(shifted), thresholds), NumOps.Zero));

            // Update scaled dual variables
            u = Engine.TensorAdd(u, Engine.TensorSubtract(db, z));

            // Check convergence
            if (PreprocessingTensorOps.IsBelow(
                    Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorSubtract(coefficients, oldCoef))), tolerance))
                break;
        }

        var flat = Engine.Reshape(coefficients, new[] { p });
        _coefficients = flat.ToVector();

        // Select features by absolute coefficient magnitude
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(Engine.TensorAbs(flat), Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// The [2p - 1, p] operator D stacking the identity (rows 0 .. p - 1) over the first differences
    /// (row p + j is <c>e_{j+1} - e_j</c>).
    /// </summary>
    private Tensor<T> DifferenceOperator(int p)
    {
        var identity = Engine.TensorEye<T>(p);
        if (p < 2)
            return identity;

        var differences = Engine.TensorSubtract(
            Engine.TensorSlice(identity, new[] { 1, 0 }, new[] { p - 1, p }),
            Engine.TensorSlice(identity, new[] { 0, 0 }, new[] { p - 1, p }));
        return Engine.TensorConcatenate(new[] { identity, differences }, 0);
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
            throw new InvalidOperationException("FusedLasso has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FusedLasso does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FusedLasso has not been fitted.");

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
