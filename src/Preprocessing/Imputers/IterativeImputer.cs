using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Iterative imputer using the MICE algorithm (Multiple Imputation by Chained Equations).
/// </summary>
/// <remarks>
/// <para>
/// IterativeImputer imputes missing values by modeling each feature with missing values
/// as a function of other features, iterating multiple times until convergence.
/// </para>
/// <para>
/// The algorithm:
/// 1. Initial imputation (mean/median for each feature)
/// 2. For each feature with missing values:
///    - Train a regression model using other features as predictors
///    - Predict missing values using the trained model
/// 3. Repeat step 2 for multiple iterations until convergence
/// </para>
/// <para><b>For Beginners:</b> MICE creates multiple "guesses" for missing values by
/// learning relationships between features. If taller people tend to be heavier,
/// MICE can use height to predict missing weight values more accurately than
/// simply using the average weight.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IterativeImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly IterativeImputerEstimator _estimator;
    private readonly IterativeImputerInitialStrategy _initialStrategy;
    private readonly int _randomState;

    // Fitted parameters
    private Vector<T>? _initialValues;
    private List<ImputationStep>? _imputationSequence;
    private int _nFeatures;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// </summary>
    public int MaxIterations => _maxIterations;

    /// <summary>
    /// Gets the convergence tolerance.
    /// </summary>
    public double Tolerance => _tolerance;

    /// <summary>
    /// Gets the estimator type used for imputation.
    /// </summary>
    public IterativeImputerEstimator Estimator => _estimator;

    /// <summary>
    /// Gets the initial imputation strategy.
    /// </summary>
    public IterativeImputerInitialStrategy InitialStrategy => _initialStrategy;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="IterativeImputer{T}"/>.
    /// </summary>
    /// <param name="maxIterations">Maximum number of imputation iterations. Defaults to 10.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-3.</param>
    /// <param name="estimator">The estimator to use for imputation. Defaults to BayesianRidge.</param>
    /// <param name="initialStrategy">Initial imputation strategy. Defaults to Mean.</param>
    /// <param name="randomState">Random seed for reproducibility. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public IterativeImputer(
        int maxIterations = 10,
        double tolerance = 1e-3,
        IterativeImputerEstimator estimator = IterativeImputerEstimator.BayesianRidge,
        IterativeImputerInitialStrategy initialStrategy = IterativeImputerInitialStrategy.Mean,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxIterations < 1)
        {
            throw new ArgumentException("Maximum iterations must be at least 1.", nameof(maxIterations));
        }

        if (tolerance < 0)
        {
            throw new ArgumentException("Tolerance must be non-negative.", nameof(tolerance));
        }

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _estimator = estimator;
        _initialStrategy = initialStrategy;
        _randomState = randomState;
    }

    /// <summary>
    /// One fitted step of the imputation sequence: the feature it imputes and the linear model
    /// (coefficients over all features, 0 on the imputed feature itself, plus intercept) that predicts it.
    /// </summary>
    private sealed class ImputationStep
    {
        public ImputationStep(int feature, Tensor<T> coefficients, T intercept)
        {
            Feature = feature;
            // The step is fitted state replayed by every later Transform, so it owns a garbage-collected copy:
            // the engine result (a reshaped view) may live in an arena buffer that is reused after the fit.
            Coefficients = Tensor<T>.FromVector(coefficients.ToVector());
            Intercept = intercept;
        }

        public int Feature { get; }

        public Tensor<T> Coefficients { get; }

        public T Intercept { get; }
    }

    /// <summary>
    /// Fits scikit-learn's IterativeImputer: initial imputation, then <c>max_iter</c> rounds that model each
    /// feature in ascending order of missing count from all the other features (rows where the feature is
    /// observed), immediately replacing its missing entries with the prediction. Stops once the largest
    /// change of a round is below <c>tol * max|observed value|</c>. Every fitted model is kept, in order, so
    /// <see cref="TransformCore"/> replays the same sequence.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeatures = data.Columns;
        int nSamples = data.Rows;
        _imputationSequence = new List<ImputationStep>();
        if (nSamples == 0 || _nFeatures == 0)
        {
            _initialValues = new Vector<T>(_nFeatures);
            return;
        }

        var x = Tensor<T>.FromMatrix(data);
        var missing = ScalerTensorOps.NanMask(x);
        _initialValues = ComputeInitialValues(data, x);
        var xt = Engine.TensorWhere(missing, ScalerTensorOps.BroadcastRow(_initialValues, nSamples), x);

        // Like scikit-learn, the models are fitted even when the training data has no missing value (so
        // transform can impute later), except for a single feature, which has no predictor.
        var order = ImputationOrder(missing);
        if (order.Length == 0 || _nFeatures == 1)
        {
            return;
        }

        var observed = Engine.TensorWhere(missing, new Tensor<T>(x.Shape.ToArray()), Engine.TensorAbs(x));
        T normalizedTolerance = NumOps.Multiply(NumOps.FromDouble(_tolerance), Engine.TensorMaxValue(observed));

        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            var previous = xt;
            foreach (int feature in order)
            {
                var present = PreprocessingTensorOps.MaskPositions(
                    PreprocessingTensorOps.Not(PreprocessingTensorOps.ColumnVector(missing, feature)));
                if (present.Length == 0)
                {
                    continue;
                }

                var step = FitStep(xt, feature, present);
                _imputationSequence.Add(step);
                xt = ApplyStep(xt, missing, step);
            }

            var change = Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorSubtract(xt, previous)));
            if (PreprocessingTensorOps.IsBelow(change, normalizedTolerance))
            {
                break;
            }
        }
    }

    /// <summary>
    /// The processed features in ascending order of missing count (scikit-learn's default
    /// <c>imputation_order="ascending"</c>, a stable sort).
    /// </summary>
    private int[] ImputationOrder(Tensor<T> missing)
    {
        var processed = new HashSet<int>(GetColumnsToProcess(_nFeatures));
        return AscendingMissingOrder(missing).Where(processed.Contains).ToArray();
    }

    /// <summary>
    /// Every column of a 0/1 missing mask ([n, features]) in ascending order of its missing count, ties in
    /// column order. The mask is counted as int: a sum in T stops counting at 2048 for Half, which would tie
    /// unequal counts and change the imputation order.
    /// </summary>
    internal static int[] AscendingMissingOrder(Tensor<T> missing)
    {
        var counts = AiDotNetEngine.Current.ReduceSum(missing.Cast<int>(), new[] { 0 }, false).AsSpan().ToArray();
        return Enumerable.Range(0, counts.Length).OrderBy(j => counts[j]).ToArray();
    }

    /// <summary>
    /// Fits the estimator that predicts <paramref name="feature"/> from all the other features over the
    /// <paramref name="present"/> rows of the current imputation.
    /// </summary>
    private ImputationStep FitStep(Tensor<T> xt, int feature, int[] present)
    {
        var rows = Engine.TensorIndexSelect(xt, new Tensor<int>(present, new[] { present.Length }), 0);
        var y = PreprocessingTensorOps.ColumnVector(rows, feature);
        var others = Engine.TensorMultiply(rows, ScalerTensorOps.BroadcastRow(OthersMask(feature), present.Length));
        var (coefficients, intercept) = FitLinearModel(others, y);
        return new ImputationStep(feature, coefficients, intercept);
    }

    /// <summary>A 0/1 [features] row that is 0 on <paramref name="feature"/> and 1 elsewhere.</summary>
    private Tensor<T> OthersMask(int feature)
    {
        var all = Enumerable.Range(0, _nFeatures).Where(j => j != feature).ToArray();
        return ScalerTensorOps.ColumnMask<T>(_nFeatures, all);
    }

    /// <summary>Replaces the missing entries of the step's feature with the step model's predictions.</summary>
    private Tensor<T> ApplyStep(Tensor<T> xt, Tensor<T> missing, ImputationStep step)
    {
        int rows = xt.Shape[0];
        var prediction = Engine.TensorAddScalar(
            Engine.TensorMatMul(xt, Engine.Reshape(step.Coefficients, new[] { _nFeatures, 1 })), step.Intercept);
        var target = Engine.TensorMultiply(
            missing, ScalerTensorOps.BroadcastRow(ScalerTensorOps.ColumnMask<T>(_nFeatures, new[] { step.Feature }), rows));
        return Engine.TensorWhere(target, Engine.TensorBroadcastTo(prediction, new[] { rows, _nFeatures }), xt);
    }

    /// <summary>
    /// The initial imputation value of every column from its observed (non-NaN) entries: mean, median, most
    /// frequent (smallest on ties, as scikit-learn's SimpleImputer) or the constant 0; 0 for a column with no
    /// observed entry.
    /// </summary>
    private Vector<T> ComputeInitialValues(Matrix<T> data, Tensor<T> x)
    {
        int nFeatures = data.Columns;
        Tensor<T> values;
        switch (_initialStrategy)
        {
            case IterativeImputerInitialStrategy.Mean:
                values = ScalerTensorOps.NanColumnMean(x);
                break;
            case IterativeImputerInitialStrategy.Median:
                values = Engine.TensorNanQuantile(x, 0.5, 0, false, QuantileInterpolation.Linear);
                break;
            case IterativeImputerInitialStrategy.MostFrequent:
                values = ColumnModes(data);
                break;
            default:
                values = new Tensor<T>(new[] { nFeatures });
                break;
        }

        var empty = Engine.TensorEquals(ScalerTensorOps.NanColumnCount(x), NumOps.Zero);
        return Engine.TensorWhere(empty, new Tensor<T>(new[] { nFeatures }), values).ToVector();
    }

    /// <summary>The most frequent observed value of every column (smallest on ties).</summary>
    private Tensor<T> ColumnModes(Matrix<T> data)
    {
        var modes = new Tensor<T>(new[] { data.Columns });
        for (int col = 0; col < data.Columns; col++)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var observed = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(ScalerTensorOps.NanMask(column)));
            if (observed.Length == 0)
            {
                continue;
            }

            var values = Engine.TensorIndexSelect(column, new Tensor<int>(observed, new[] { observed.Length }), 0);
            var (mode, _) = Engine.TensorMode(values);
            modes = Engine.TensorSetSlice(modes, PreprocessingTensorOps.Filled(new[] { 1 }, mode), new[] { col });
        }

        return modes;
    }

    /// <summary>
    /// Fits the configured linear estimator with an intercept on centred data:
    /// <see cref="IterativeImputerEstimator.BayesianRidge"/> as scikit-learn's BayesianRidge (evidence
    /// maximisation of alpha and lambda, alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6, max_iter 300,
    /// tol 1e-3), <see cref="IterativeImputerEstimator.Ridge"/> as Ridge(alpha=1) and
    /// <see cref="IterativeImputerEstimator.LinearRegression"/> as least squares (minimum-norm, singular values
    /// below eps * max(n, p) * the largest are cut). All solve through the eigendecomposition of X^T X.
    /// </summary>
    private (Tensor<T> Coefficients, T Intercept) FitLinearModel(Tensor<T> features, Tensor<T> y)
    {
        int n = features.Shape[0];
        int p = features.Shape[1];
        var xMean = Engine.ReduceMean(features, new[] { 0 }, false);
        T yMean = Engine.TensorMean(y);
        var xc = Engine.TensorSubtract(features, ScalerTensorOps.BroadcastRow(xMean, n));
        var yc = Engine.TensorSubtractScalar(y, yMean);

        var gram = Engine.TensorMatMul(Engine.TensorTranspose(xc), xc);
        var xty = Engine.TensorMatMul(Engine.TensorTranspose(xc), Engine.Reshape(yc, new[] { n, 1 }));
        var (eigenvalues, eigenvectors) = LinalgDispatch<T>.Eigh(gram);
        eigenvalues = Engine.TensorMax(eigenvalues, NumOps.Zero);
        var projected = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(eigenvectors), xty), new[] { p });

        Tensor<T> coefficients;
        switch (_estimator)
        {
            case IterativeImputerEstimator.Ridge:
                coefficients = SolveSpectral(eigenvectors, projected,
                    Engine.TensorReciprocal(Engine.TensorAddScalar(eigenvalues, NumOps.One)));
                break;
            case IterativeImputerEstimator.LinearRegression:
                coefficients = SolveLeastSquares(eigenvalues, eigenvectors, projected, Math.Max(n, p));
                break;
            default:
                coefficients = SolveBayesianRidge(xc, yc, eigenvalues, eigenvectors, projected);
                break;
        }

        T intercept = NumOps.Subtract(yMean, Engine.TensorSum(Engine.TensorMultiply(xMean, coefficients)));
        return (coefficients, intercept);
    }

    /// <summary><c>V diag(weights) V^T X^T y</c>, with <paramref name="projected"/> = <c>V^T X^T y</c>.</summary>
    private Tensor<T> SolveSpectral(Tensor<T> eigenvectors, Tensor<T> projected, Tensor<T> weights)
    {
        int p = projected.Length;
        var scaled = Engine.Reshape(Engine.TensorMultiply(projected, weights), new[] { p, 1 });
        return Engine.Reshape(Engine.TensorMatMul(eigenvectors, scaled), new[] { p });
    }

    /// <summary>Minimum-norm least squares: eigenvalues below the lstsq cut-off are dropped.</summary>
    private Tensor<T> SolveLeastSquares(Tensor<T> eigenvalues, Tensor<T> eigenvectors, Tensor<T> projected, int size)
    {
        // singular value s is kept when s > eps * max(n, p) * s_max, i.e. eigenvalue e = s^2 > e_max * (eps * size)^2
        double relative = ScalerTensorOps.Float64Epsilon * size;
        T cutoff = NumOps.Multiply(Engine.TensorMaxValue(eigenvalues), NumOps.FromDouble(relative * relative));
        var kept = Engine.TensorGreaterThan(eigenvalues, cutoff);
        var safe = Engine.TensorWhere(kept, eigenvalues, PreprocessingTensorOps.Filled(eigenvalues.Shape.ToArray(), NumOps.One));
        var weights = Engine.TensorMultiply(kept, Engine.TensorReciprocal(safe));
        return SolveSpectral(eigenvectors, projected, weights);
    }

    /// <summary>
    /// scikit-learn BayesianRidge on centred data: alpha starts at 1 / (var(y) + eps) and lambda at 1; each
    /// round sets coef = (X^T X + lambda / alpha I)^-1 X^T y, gamma = sum(alpha e / (lambda + alpha e)),
    /// lambda = (gamma + 2 lambda_1) / (|coef|^2 + 2 lambda_2) and
    /// alpha = (n - gamma + 2 alpha_1) / (|y - X coef|^2 + 2 alpha_2), until the coefficients move by less than
    /// tol (L1); the returned coefficients use the final alpha and lambda.
    /// </summary>
    private Tensor<T> SolveBayesianRidge(Tensor<T> xc, Tensor<T> yc, Tensor<T> eigenvalues, Tensor<T> eigenvectors, Tensor<T> projected)
    {
        const double hyper = 1e-6;
        const int maxIterations = 300;
        int n = xc.Shape[0];
        int p = eigenvalues.Length;
        T twoHyper = NumOps.FromDouble(2 * hyper);
        T evidenceCount = NumOps.FromDouble(n + 2 * hyper);
        T convergence = NumOps.FromDouble(1e-3);
        var shape = new[] { p };

        T variance = Engine.TensorMean(Engine.TensorSquare(yc));
        var alpha = PreprocessingTensorOps.Filled(new[] { 1 },
            NumOps.Divide(NumOps.One, NumOps.Add(variance, NumOps.FromDouble(ScalerTensorOps.Float64Epsilon))));
        var lambda = PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.One);

        Tensor<T>? previous = null;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var coefficients = BayesianCoefficients(eigenvalues, eigenvectors, projected, alpha, lambda);
            var residual = Engine.TensorSubtract(
                yc, Engine.Reshape(Engine.TensorMatMul(xc, Engine.Reshape(coefficients, new[] { p, 1 })), new[] { n }));
            var rmse = Engine.ReduceSum(Engine.TensorSquare(residual), new[] { 0 }, true);

            var alphaE = Engine.TensorMultiply(eigenvalues, Engine.TensorBroadcastTo(alpha, shape));
            var gamma = Engine.ReduceSum(
                Engine.TensorDivide(alphaE, Engine.TensorAdd(Engine.TensorBroadcastTo(lambda, shape), alphaE)), new[] { 0 }, true);
            lambda = Engine.TensorDivide(
                Engine.TensorAddScalar(gamma, twoHyper),
                Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorSquare(coefficients), new[] { 0 }, true), twoHyper));
            alpha = Engine.TensorDivide(
                Engine.TensorAddScalar(Engine.TensorNegate(gamma), evidenceCount), Engine.TensorAddScalar(rmse, twoHyper));

            if (previous is not null &&
                PreprocessingTensorOps.IsBelow(Engine.TensorSum(Engine.TensorAbs(Engine.TensorSubtract(previous, coefficients))), convergence))
            {
                break;
            }

            previous = coefficients;
        }

        return BayesianCoefficients(eigenvalues, eigenvectors, projected, alpha, lambda);
    }

    /// <summary><c>(X^T X + lambda / alpha I)^-1 X^T y</c> in the eigenbasis.</summary>
    private Tensor<T> BayesianCoefficients(
        Tensor<T> eigenvalues, Tensor<T> eigenvectors, Tensor<T> projected, Tensor<T> alpha, Tensor<T> lambda)
    {
        var ratio = Engine.TensorBroadcastTo(Engine.TensorDivide(lambda, alpha), eigenvalues.Shape.ToArray());
        return SolveSpectral(eigenvectors, projected, Engine.TensorReciprocal(Engine.TensorAdd(eigenvalues, ratio)));
    }

    /// <summary>
    /// Imputes with the fitted initial values, then replays the fitted imputation sequence (scikit-learn's
    /// <c>imputation_sequence_</c>) on the missing entries of the processed columns.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_initialValues is null || _imputationSequence is null)
        {
            throw new InvalidOperationException("IterativeImputer has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _initialValues.Length);
        int nSamples = data.Rows;
        int nFeatures = data.Columns;
        if (nSamples == 0 || nFeatures == 0)
        {
            return new Matrix<T>(nSamples, nFeatures);
        }

        var x = Tensor<T>.FromMatrix(data);
        var missing = ScalerTensorOps.NanMask(x);
        var xt = Engine.TensorWhere(missing, ScalerTensorOps.BroadcastRow(_initialValues, nSamples), x);
        foreach (var step in _imputationSequence)
        {
            xt = ApplyStep(xt, missing, step);
        }

        // Only the processed columns are imputed; the others keep their original values.
        var mask = ScalerTensorOps.ColumnMask<T>(nFeatures, GetColumnsToProcess(nFeatures));
        return ScalerTensorOps.MergeColumns(mask, xt, data);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IterativeImputer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies the estimator to use for iterative imputation.
/// </summary>
public enum IterativeImputerEstimator
{
    /// <summary>
    /// Bayesian Ridge Regression (regularized linear regression).
    /// </summary>
    BayesianRidge,

    /// <summary>
    /// Simple linear regression (OLS).
    /// </summary>
    LinearRegression,

    /// <summary>
    /// Ridge regression with fixed regularization.
    /// </summary>
    Ridge
}

/// <summary>
/// Specifies the initial imputation strategy.
/// </summary>
public enum IterativeImputerInitialStrategy
{
    /// <summary>
    /// Use mean of non-missing values.
    /// </summary>
    Mean,

    /// <summary>
    /// Use median of non-missing values.
    /// </summary>
    Median,

    /// <summary>
    /// Use most frequent value.
    /// </summary>
    MostFrequent,

    /// <summary>
    /// Use a constant value (zero).
    /// </summary>
    Constant
}
