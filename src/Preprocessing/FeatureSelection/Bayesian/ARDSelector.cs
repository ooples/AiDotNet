using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Automatic Relevance Determination (ARD) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Bayesian ARD to automatically determine which features are relevant
/// by learning individual precision (inverse variance) parameters for each
/// feature's coefficient. Features with high precision (low variance) are
/// effectively pruned.
/// </para>
/// <para><b>For Beginners:</b> ARD is like giving each feature its own "importance dial."
/// During training, the method automatically turns down the dial for irrelevant features
/// (making their effect nearly zero) while keeping the dial up for important ones.
/// Features whose dials get turned all the way down are removed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ARDSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alphaThreshold;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private Vector<T>? _alphas;
    private Vector<T>? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Alphas => _alphas;
    public Vector<T>? Coefficients => _coefficients;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ARDSelector(
        int nFeaturesToSelect = 10,
        double alphaThreshold = 1e8,
        int maxIterations = 300,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alphaThreshold <= 0)
            throw new ArgumentOutOfRangeException(nameof(alphaThreshold), "Alpha threshold must be positive.");
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Max iterations must be at least 1.");
        if (tolerance <= 0)
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Tolerance must be positive.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _alphaThreshold = alphaThreshold;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ARDSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0)
            throw new ArgumentException("Data must have at least one row.", nameof(data));
        if (data.Columns == 0)
            throw new ArgumentException("Data must have at least one column.", nameof(data));
        if (_nFeaturesToSelect > data.Columns)
            throw new ArgumentException(
                $"Number of features to select ({_nFeaturesToSelect}) cannot exceed number of columns ({data.Columns}).",
                nameof(data));

        _nInputFeatures = data.Columns;
        int numToSelect = Math.Min(_nFeaturesToSelect, data.Columns);
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            // diag(X^T X), and with it the posterior precision alpha + beta diag(X^T X), passes Half's 65504 once a
            // column's values pass 256 / sqrt(n) (no rescaling helps: the EM is not scale-equivariant, since alpha
            // and beta start at 1), and the 1e8 alpha threshold and 1e10 cap are not finite in Half at all. The EM
            // runs in float and its results are cast back once.
            var (floatAlphas, floatCoefficients, floatSelected) = RunEm(X.Cast<float>(), y.Cast<float>(), numToSelect);
            _alphas = floatAlphas.Cast<T>().ToVector();
            _coefficients = floatCoefficients.Cast<T>().ToVector();
            _selectedIndices = floatSelected;
        }
        else
        {
            var (alphas, coefficients, selected) = RunEm(X, y, numToSelect);
            _alphas = alphas.ToVector();
            _coefficients = coefficients.ToVector();
            _selectedIndices = selected;
        }

        IsFitted = true;
    }

    /// <summary>
    /// The ARD EM iterations on X [n, p] and y [n] in <typeparamref name="TC"/>: returns the precisions alpha, the
    /// posterior mean coefficients and the selected feature indices.
    /// </summary>
    private (Tensor<TC> Alphas, Tensor<TC> Coefficients, int[] Selected) RunEm<TC>(Tensor<TC> X, Tensor<TC> target, int numToSelect)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        int n = X.Shape[0];
        int p = X.Shape[1];
        var y = engine.Reshape(target, new[] { n, 1 });

        TC eps = PreprocessingTensorOps.Tolerance<TC>(1e-10);
        TC tolerance = ops.FromDouble(_tolerance);
        var thresholdFill = PreprocessingTensorOps.Filled(new[] { p }, ops.FromDouble(_alphaThreshold));
        var largeFill = PreprocessingTensorOps.Filled(new[] { p }, ops.FromDouble(1e10));

        // diag(X^T X) (the diagonal posterior approximation only reads the diagonal) and X^T y.
        var xtxDiag = engine.ReduceSum(engine.TensorSquare(X), new[] { 0 }, false);
        var xty = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(X), y), new[] { p });

        // Initialize alphas (precision parameters)
        var alphas = PreprocessingTensorOps.Filled(new[] { p }, ops.One);
        var coefficients = new Tensor<TC>(new[] { p });
        TC beta = ops.One; // Noise precision

        // EM iterations
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var oldAlphas = alphas;

            // Posterior covariance (diagonal approximation): 1 / (alpha + beta * diag(X^T X))
            var denom = engine.TensorAdd(alphas, engine.TensorMultiplyScalar(xtxDiag, beta));
            var sigma = engine.TensorWhere(engine.TensorGreaterThan(denom, eps), engine.TensorReciprocal(denom), largeFill);

            // Posterior mean (coefficients)
            coefficients = engine.TensorMultiplyScalar(engine.TensorMultiply(sigma, xty), beta);

            // gamma = 1 - alpha * sigma (effective number of parameters per feature)
            var gamma = engine.TensorAddScalar(engine.TensorNegate(engine.TensorMultiply(alphas, sigma)), ops.One);

            // Update alphas
            var w2 = engine.TensorSquare(coefficients);
            var valid = engine.TensorMultiply(engine.TensorGreaterThan(w2, eps), engine.TensorGreaterThan(gamma, eps));
            alphas = engine.TensorWhere(valid, engine.TensorDivide(gamma, w2), thresholdFill);

            // Update beta (noise precision)
            var residual = engine.TensorSubtract(y, engine.TensorMatMul(X, engine.Reshape(coefficients, new[] { p, 1 })));
            beta = UpdateNoisePrecision(n, engine.TensorSum(gamma), engine.TensorSumOfSquares(residual));

            // Check convergence
            var change = engine.TensorDivide(
                engine.TensorAbs(engine.TensorSubtract(alphas, oldAlphas)),
                engine.TensorAddScalar(engine.TensorAbs(oldAlphas), eps));
            if (PreprocessingTensorOps.IsBelow(engine.TensorMaxValue(change), tolerance))
                break;
        }

        // Select features with low alpha (high relevance)
        var relevant = engine.TensorLessThan(alphas, ops.FromDouble(_alphaThreshold));
        int relevantCount = PreprocessingTensorOps.CountNonZero(relevant);
        int[] selected = relevantCount >= numToSelect
            ? PreprocessingTensorOps.BottomKIndicesSorted(alphas, numToSelect)
            // Not enough features below threshold: take top by coefficient magnitude
            : PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(coefficients), numToSelect);

        return (alphas, coefficients, selected);
    }

    private static TC UpdateNoisePrecision<TC>(int n, TC gammaSum, TC residualSum)
    {
        var ops = MathHelper.GetNumericOperations<TC>();
        TC eps = PreprocessingTensorOps.Tolerance<TC>(1e-10);
        TC value = ops.Divide(ops.Subtract(ops.FromDouble(n), gammaSum), ops.Add(residualSum, eps));
        return PreprocessingTensorOps.ScalarMax(value, eps);
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
            throw new InvalidOperationException("ARDSelector has not been fitted.");
        if (data.Columns != _nInputFeatures)
            throw new ArgumentException(
                $"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ARDSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ARDSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ARDSelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, but got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
