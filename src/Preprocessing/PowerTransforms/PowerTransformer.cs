using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Preprocessing.PowerTransforms;

/// <summary>
/// Specifies the power transformation method.
/// </summary>
public enum PowerTransformMethod
{
    /// <summary>
    /// Box-Cox transformation. Requires strictly positive data.
    /// </summary>
    BoxCox,

    /// <summary>
    /// Yeo-Johnson transformation. Works with positive, negative, and zero values.
    /// </summary>
    YeoJohnson
}

/// <summary>
/// Applies power transformations (Box-Cox or Yeo-Johnson) to make data more Gaussian-like.
/// </summary>
/// <remarks>
/// <para>
/// PowerTransformer applies a power transformation to each feature to make it more Gaussian-like.
/// Box-Cox requires strictly positive data, while Yeo-Johnson works with any values.
/// The transformation can help stabilize variance and improve the fit of models that assume normality.
/// </para>
/// <para><b>For Beginners:</b> This transformer makes your data more "bell-curve shaped":
/// - Box-Cox: For strictly positive data (prices, counts)
/// - Yeo-Johnson: Works with any data including negatives and zeros
///
/// After transformation, features will have more normal (Gaussian) distributions,
/// which helps many machine learning algorithms perform better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PowerTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly PowerTransformMethod _method;
    private readonly bool _standardize;

    // Fitted parameters
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _lambdas;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _mean;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _stdDev;

    /// <summary>
    /// Gets the power transformation method used.
    /// </summary>
    public PowerTransformMethod Method => _method;

    /// <summary>
    /// Gets whether standardization is applied after transformation.
    /// </summary>
    public bool Standardize => _standardize;

    /// <summary>
    /// Gets the optimal lambda parameters for each feature.
    /// </summary>
    public Vector<T>? Lambdas => _lambdas;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="PowerTransformer{T}"/>.
    /// </summary>
    /// <param name="method">The transformation method. Defaults to YeoJohnson.</param>
    /// <param name="standardize">Whether to standardize after transformation. Defaults to true.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public PowerTransformer(
        PowerTransformMethod method = PowerTransformMethod.YeoJohnson,
        bool standardize = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _method = method;
        _standardize = standardize;
    }

    /// <summary>
    /// Learns the optimal lambda parameters for each feature.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // Columns not processed keep lambda 1, mean 0 and standard deviation 1
        var lambdas = PreprocessingTensorOps.Filled(new[] { numColumns }, NumOps.One);
        var means = new Tensor<T>(new[] { numColumns });
        var stdDevs = PreprocessingTensorOps.Filled(new[] { numColumns }, NumOps.One);

        if (data.Rows > 0 && columnsToProcess.Length > 0)
        {
            var x = Tensor<T>.FromMatrix(data);
            var transformedColumns = new List<Tensor<T>>(numColumns);
            for (int col = 0; col < numColumns; col++)
            {
                var column = PreprocessingTensorOps.ColumnVector(x, col);
                if (!columnsToProcess.Contains(col))
                {
                    transformedColumns.Add(column);
                    continue;
                }

                var (lambda, transformed) = FitColumn(column, col);
                lambdas = Engine.TensorSetSlice(lambdas, lambda, new[] { col });
                transformedColumns.Add(transformed);
            }

            if (_standardize)
            {
                // scikit-learn standardizes with StandardScaler: NaN-ignoring mean and population standard
                // deviation, with (near-)zero scales left at 1.
                var transformedData = Tensor<T>.FromMatrix(PreprocessingTensorOps.StackColumns(transformedColumns, data.Rows));
                var (mean, std, constant) = ScalerTensorOps.NanColumnMoments(transformedData);
                var scale = ScalerTensorOps.HandleZerosInScale(std, constant);
                var mask = ScalerTensorOps.ColumnMask<T>(numColumns, columnsToProcess);
                means = Tensor<T>.FromVector(ScalerTensorOps.MaskedParameters(mask, mean, NumOps.Zero));
                stdDevs = Tensor<T>.FromVector(ScalerTensorOps.MaskedParameters(mask, scale, NumOps.One));
            }
        }

        _lambdas = lambdas.ToVector();
        _mean = means.ToVector();
        _stdDev = stdDevs.ToVector();
    }

    /// <summary>
    /// Fits the power parameter of one column (over its non-NaN values) and returns it as a [1] tensor with the
    /// transformed column.
    /// </summary>
    private (Tensor<T> Lambda, Tensor<T> Transformed) FitColumn(Tensor<T> column, int col)
    {
        var observed = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(ScalerTensorOps.NanMask(column)));
        var values = Engine.TensorIndexSelect(column, new Tensor<int>(observed, new[] { observed.Length }), 0);

        // Check for Box-Cox constraint
        if (_method == PowerTransformMethod.BoxCox
            && PreprocessingTensorOps.CountNonZero(Engine.TensorGreaterThan(values, NumOps.Zero)) < values.Length)
        {
            throw new ArgumentException(
                $"Box-Cox transformation requires strictly positive data. " +
                $"Found non-positive value in column {col}. Use YeoJohnson instead.");
        }

        // scikit-learn's _fit keeps lambda 1 for a constant Yeo-Johnson feature (its likelihood has no maximum),
        // and scipy.stats.boxcox raises "Data must not be constant." for Box-Cox.
        bool constant = IsConstantFeature(values);
        if (constant && _method == PowerTransformMethod.BoxCox)
        {
            throw new ArgumentException($"Box-Cox transformation requires non-constant data. Column {col} is constant.");
        }

        double bestLambda = values.Length == 0 || constant ? 1.0 : FindOptimalLambda(values);
        T lambda = NumOps.FromDouble(bestLambda);
        return (PreprocessingTensorOps.Filled(new[] { 1 }, lambda), TransformColumn(column, lambda));
    }

    /// <summary>
    /// The maximum-likelihood power parameter, found like scikit-learn / SciPy with
    /// <c>scipy.optimize.brent(-loglike, brack=(-2, 2))</c>. The profile log-likelihood (up to constants) is
    /// <c>-n/2 log(var(y)) + (lambda - 1) J</c> with <c>J = sum log(x)</c> (Box-Cox) or
    /// <c>J = sum sign(x) log1p(|x|)</c> (Yeo-Johnson).
    /// </summary>
    private double FindOptimalLambda(Tensor<T> values)
    {
        var jacobianTerms = _method == PowerTransformMethod.BoxCox
            ? Engine.TensorLog(values)
            : Engine.TensorMultiply(Engine.TensorSign(values), Engine.TensorLog1p(Engine.TensorAbs(values)));
        // Reduced as numpy does (float32 intermediates for float16), not in Half, whose sum stops growing
        double jacobian = ScalerTensorOps.Sum(jacobianTerms);
        double n = values.Length;
        return BrentMinimizer.Minimize(lambda => -ComputeLogLikelihood(values, lambda, n, jacobian), -2.0, 2.0);
    }

    /// <summary>
    /// scikit-learn's <c>_is_constant_feature</c> on the float64 mean and variance of the observed values:
    /// <c>var &lt;= n eps var + (n mean eps)^2</c>.
    /// </summary>
    private static bool IsConstantFeature(Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0)
        {
            return false;
        }

        double mean = ScalerTensorOps.Sum(values) / n;
        double variance = ScalerTensorOps.Variance(values);
        double eps = ScalerTensorOps.Float64Epsilon;
        return variance <= n * eps * variance + (n * mean * eps) * (n * mean * eps);
    }

    /// <summary>
    /// The profile log-likelihood of <paramref name="lambda"/>; transformed data whose variance is below the
    /// smallest normal double gives negative infinity (scikit-learn returns +inf for the negated value).
    /// </summary>
    private double ComputeLogLikelihood(Tensor<T> values, double lambda, double n, double jacobian)
    {
        var transformed = TransformColumn(values, NumOps.FromDouble(lambda));
        double variance = ScalerTensorOps.Variance(transformed);
        if (!(variance >= Float64Tiny))
        {
            return double.NegativeInfinity;
        }

        return -n / 2 * Math.Log(variance) + (lambda - 1) * jacobian;
    }

    /// <summary>The Box-Cox or Yeo-Johnson transform of a column (NaN passes through).</summary>
    private Tensor<T> TransformColumn(Tensor<T> column, T lambda) =>
        _method == PowerTransformMethod.BoxCox ? BoxCoxTransform(column, lambda) : YeoJohnsonTransform(column, lambda);

    /// <summary>scikit-learn's <c>_box_cox</c> (SciPy <c>special.boxcox</c>): <c>expm1(lambda log x) / lambda</c>, or <c>log x</c> at lambda 0.</summary>
    private Tensor<T> BoxCoxTransform(Tensor<T> x, T lambda)
    {
        var logX = Engine.TensorLog(x);
        if (Math.Abs(NumOps.ToDouble(lambda)) < BoxCoxZero)
        {
            return logX;
        }

        return Engine.TensorDivideScalar(Engine.TensorExpm1(Engine.TensorMultiplyScalar(logX, lambda)), lambda);
    }

    /// <summary>
    /// scikit-learn's <c>_yeo_johnson_transform</c>: for x &gt;= 0, <c>((x + 1)^lambda - 1) / lambda</c> (or
    /// <c>log1p(x)</c> when |lambda| &lt; eps); for x &lt; 0, <c>-((1 - x)^(2 - lambda) - 1) / (2 - lambda)</c> (or
    /// <c>-log1p(-x)</c> when |lambda - 2| &lt;= eps).
    /// </summary>
    private Tensor<T> YeoJohnsonTransform(Tensor<T> x, T lambda)
    {
        double lambdaValue = NumOps.ToDouble(lambda);
        var negative = Engine.TensorLessThan(x, NumOps.Zero);
        var zeros = new Tensor<T>(x.Shape.ToArray());
        var positivePart = Engine.TensorWhere(negative, zeros, x);
        var negativePart = Engine.TensorWhere(negative, x, zeros);

        Tensor<T> positive = Math.Abs(lambdaValue) < Float64Spacing
            ? Engine.TensorLog1p(positivePart)
            : Engine.TensorDivideScalar(
                Engine.TensorSubtractScalar(Engine.TensorPower(Engine.TensorAddScalar(positivePart, NumOps.One), lambda), NumOps.One),
                lambda);

        T twoMinusLambda = NumOps.FromDouble(2 - lambdaValue);
        Tensor<T> negativeValues = Math.Abs(lambdaValue - 2) > Float64Spacing
            ? Engine.TensorNegate(Engine.TensorDivideScalar(
                Engine.TensorSubtractScalar(
                    Engine.TensorPower(Engine.TensorAddScalar(Engine.TensorNegate(negativePart), NumOps.One), twoMinusLambda), NumOps.One),
                twoMinusLambda))
            : Engine.TensorNegate(Engine.TensorLog1p(Engine.TensorNegate(negativePart)));

        return Engine.TensorWhere(negative, negativeValues, positive);
    }

    /// <summary>scikit-learn's inverse Yeo-Johnson transform.</summary>
    private Tensor<T> InverseYeoJohnsonTransform(Tensor<T> y, T lambda)
    {
        double lambdaValue = NumOps.ToDouble(lambda);
        var negative = Engine.TensorLessThan(y, NumOps.Zero);
        var zeros = new Tensor<T>(y.Shape.ToArray());
        var positivePart = Engine.TensorWhere(negative, zeros, y);
        var negativePart = Engine.TensorWhere(negative, y, zeros);

        Tensor<T> positive = Math.Abs(lambdaValue) < Float64Spacing
            ? Engine.TensorExpm1(positivePart)
            : Engine.TensorSubtractScalar(
                Engine.TensorPower(
                    Engine.TensorAddScalar(Engine.TensorMultiplyScalar(positivePart, lambda), NumOps.One),
                    NumOps.FromDouble(1 / lambdaValue)),
                NumOps.One);

        Tensor<T> negativeValues;
        if (Math.Abs(lambdaValue - 2) > Float64Spacing)
        {
            var bases = Engine.TensorAddScalar(
                Engine.TensorMultiplyScalar(negativePart, NumOps.FromDouble(-(2 - lambdaValue))), NumOps.One);
            negativeValues = PreprocessingTensorOps.OneMinus(Engine.TensorPower(bases, NumOps.FromDouble(1 / (2 - lambdaValue))));
        }
        else
        {
            negativeValues = PreprocessingTensorOps.OneMinus(Engine.TensorExp(Engine.TensorNegate(negativePart)));
        }

        return Engine.TensorWhere(negative, negativeValues, positive);
    }

    /// <summary>SciPy's <c>special.inv_boxcox</c>: <c>exp(log1p(lambda y) / lambda)</c>, or <c>exp(y)</c> at lambda 0.</summary>
    private Tensor<T> InverseBoxCoxTransform(Tensor<T> y, T lambda)
    {
        if (Math.Abs(NumOps.ToDouble(lambda)) < BoxCoxZero)
        {
            return Engine.TensorExp(y);
        }

        return Engine.TensorExp(Engine.TensorDivideScalar(Engine.TensorLog1p(Engine.TensorMultiplyScalar(y, lambda)), lambda));
    }

    /// <summary>numpy's <c>spacing(1.0)</c>, the threshold scikit-learn uses for lambda 0 / 2 in Yeo-Johnson.</summary>
    private const double Float64Spacing = 2.220446049250313e-16;

    /// <summary>SciPy's threshold for lambda 0 in <c>boxcox</c> / <c>inv_boxcox</c>.</summary>
    private const double BoxCoxZero = 1e-19;

    /// <summary>numpy's <c>finfo(float64).tiny</c>.</summary>
    private const double Float64Tiny = 2.2250738585072014e-308;

    /// <summary>
    /// Applies (or inverts) the fitted transform to every processed column; the other columns pass through.
    /// </summary>
    private Matrix<T> MapColumns(Matrix<T> data, bool inverse)
    {
        var lambdaVector = _lambdas ?? new Vector<T>(0);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        var x = Tensor<T>.FromMatrix(data);
        var lambdas = lambdaVector.ToArray();
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);
        for (int col = 0; col < numColumns; col++)
        {
            var column = PreprocessingTensorOps.ColumnVector(x, col);
            columns.Add(processSet.Contains(col) && col < lambdas.Length ? MapColumn(column, col, lambdas[col], inverse) : column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>Transforms then standardizes one column, or un-standardizes then inverts it.</summary>
    private Tensor<T> MapColumn(Tensor<T> column, int col, T lambda, bool inverse)
    {
        bool standardize = _standardize && _mean is not null && _stdDev is not null;
        T mean = standardize && _mean is not null ? _mean[col] : NumOps.Zero;
        T std = standardize && _stdDev is not null ? _stdDev[col] : NumOps.One;

        if (!inverse)
        {
            var transformed = TransformColumn(column, lambda);
            return standardize
                ? Engine.TensorDivideScalar(Engine.TensorSubtractScalar(transformed, mean), std)
                : transformed;
        }

        // Undo standardization first
        var y = standardize ? Engine.TensorAddScalar(Engine.TensorMultiplyScalar(column, std), mean) : column;
        return _method == PowerTransformMethod.BoxCox ? InverseBoxCoxTransform(y, lambda) : InverseYeoJohnsonTransform(y, lambda);
    }

    /// <summary>
    /// Transforms the data by applying the power transformation.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lambdas is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        return MapColumns(data, inverse: false);
    }

    /// <summary>
    /// Reverses the power transformation.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_lambdas is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        return MapColumns(data, inverse: true);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (PowerTransformer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
