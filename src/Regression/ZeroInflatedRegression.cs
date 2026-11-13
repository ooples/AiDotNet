using AiDotNet.Attributes;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Zero-Inflated regression for count data with excess zeros.
/// </summary>
/// <remarks>
/// <para>
/// Zero-Inflated models handle count data where there are more zeros than a standard
/// count distribution would predict. They model the data as a mixture: with probability π,
/// the observation is a "structural zero," and with probability (1-π), it follows a count
/// distribution (Poisson or Negative Binomial).
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine counting how many times customers visit a store each month:
/// - Some people NEVER visit (structural zeros) - they live far away or shop elsewhere
/// - Some people visit sometimes but happened to visit 0 times this month (sampling zeros)
///
/// Standard Poisson regression treats all zeros the same, but Zero-Inflated models
/// recognize these two types of zeros:
///
/// 1. The "zero model" predicts WHO are structural zeros (π)
/// 2. The "count model" predicts HOW MANY for non-structural-zero people (λ)
///
/// Example interpretation:
/// - "30% of potential customers are 'never visitors' (π = 0.3)"
/// - "Among potential visitors, the average visit rate is 2.5 times/month (λ = 2.5)"
///
/// This gives better predictions and allows you to understand both processes.
/// </para>
/// <para>
/// Reference: Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application
/// to Defects in Manufacturing". Technometrics, 34(1), 1-14.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Zero-Inflated regression for count data with excess zeros
/// var options = new ZeroInflatedRegressionOptions();
///
/// // Prepare training data: 6 samples with 2 features, count targets with excess zeros
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 0, 0, 3, 0, 7, 12 });
///
/// // Train with mixture model (zero process + count process)
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ZeroInflatedRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict count for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing", "https://doi.org/10.1080/00401706.1992.10485228", Year = 1992, Authors = "Diane Lambert")]
public partial class ZeroInflatedRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Coefficients for the count model (λ).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _countCoefficients;

    /// <summary>
    /// Intercept for the count model.
    /// </summary>
    private T _countIntercept;

    /// <summary>
    /// Coefficients for the zero-inflation model (π).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _zeroCoefficients;

    /// <summary>
    /// Intercept for the zero-inflation model.
    /// </summary>
    private T _zeroIntercept;

    /// <summary>
    /// Dispersion parameter for Negative Binomial (if applicable).
    /// </summary>
    private T _dispersion;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ZeroInflatedRegressionOptions _options;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Gets the count model coefficients.
    /// </summary>
    public Vector<T>? CountCoefficients => _countCoefficients;

    /// <summary>
    /// Gets the zero-inflation model coefficients.
    /// </summary>
    public Vector<T>? ZeroCoefficients => _zeroCoefficients;

    /// <summary>
    /// Initializes a new instance of ZeroInflatedRegression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public ZeroInflatedRegression(ZeroInflatedRegressionOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new ZeroInflatedRegressionOptions();
        _countIntercept = NumOps.Zero;
        _zeroIntercept = NumOps.Zero;
        _dispersion = NumOps.One;
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Coerce response values to non-negative integers (count data)
        // Continuous values are rounded; negatives are clamped to zero
        // (round half to even, then clamp at zero; written back into y in place).
        if (n > 0)
        {
            var counts = Engine.TensorMax(Engine.TensorRound(y.AsTensor()), NumOps.Zero).ToVector();
            counts.AsSpan().CopyTo(y.AsWritableSpan());
        }

        // Initialize parameters
        InitializeParameters(y);

        T prevLogLik = NumOps.MinValue;
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        // EM algorithm for optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: compute posterior probability of being a structural zero
            var (lambdas, pis) = ComputePredictions(x);
            var posteriorZero = ComputePosteriorZero(y, lambdas, pis);

            // M-step: update parameters
            // Update count model using weighted data (weight = 1 - posterior zero)
            UpdateCountModel(x, y, posteriorZero);

            // Update zero-inflation model
            if (_options.ModelZeroInflation)
            {
                UpdateZeroModel(x, posteriorZero);
            }

            // Update dispersion for Negative Binomial
            if (_options.DistributionFamily == ZeroInflatedDistributionFamily.NegativeBinomial)
            {
                UpdateDispersion(y, lambdas, posteriorZero);
            }

            // Check convergence
            (lambdas, pis) = ComputePredictions(x);
            T logLik = ComputeLogLikelihood(y, lambdas, pis);

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(logLik, prevLogLik)), tolerance))
            {
                break;
            }
            prevLogLik = logLik;
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var (lambdas, pis) = await Task.Run(() => ComputePredictions(input));
        // Expected value = (1 - π) * λ
        return Engine.Multiply(OneMinus(pis), lambdas);
    }

    /// <summary>
    /// Predicts the probability of being a structural zero for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of zero-inflation probabilities.</returns>
    public async Task<Vector<T>> PredictZeroProbabilityAsync(Matrix<T> input)
    {
        var (_, pis) = await Task.Run(() => ComputePredictions(input));
        return pis;
    }

    /// <summary>
    /// Predicts the expected count conditional on not being a structural zero.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of conditional count expectations.</returns>
    public async Task<Vector<T>> PredictConditionalCountAsync(Matrix<T> input)
    {
        var (lambdas, _) = await Task.Run(() => ComputePredictions(input));
        return lambdas;
    }

    /// <summary>
    /// Predicts the probability mass function for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="maxCount">Maximum count value to compute probability for.</param>
    /// <returns>Matrix where [i,k] is P(Y_i = k).</returns>
    public async Task<Matrix<T>> PredictPMFAsync(Matrix<T> input, int maxCount = 20)
    {
        var (lambdas, pis) = await Task.Run(() => ComputePredictions(input));
        int rows = input.Rows;
        int columns = maxCount + 1;
        if (rows == 0 || columns <= 0)
        {
            return new Matrix<T>(rows, Math.Max(columns, 0));
        }

        // [rows, counts] grids of λ_i, k and (1 - π_i)
        var shape = new[] { rows, columns };
        var lambdaGrid = Engine.TensorBroadcastTo(Engine.Reshape(lambdas.AsTensor(), new[] { rows, 1 }), shape);
        var kGrid = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(0, columns, 1), new[] { 1, columns }), shape);
        var oneMinusPiGrid = Engine.TensorBroadcastTo(Engine.Reshape(OneMinus(pis).AsTensor(), new[] { rows, 1 }), shape);

        // P(Y=k) = (1-π) * P_count(k), and P(Y=0) additionally gets the structural-zero mass π
        var pmf = Engine.TensorMultiply(oneMinusPiGrid, CountProbability(kGrid, lambdaGrid));
        var piColumn = Engine.Reshape(pis.AsTensor(), new[] { rows, 1 });
        var zeroMass = columns == 1
            ? piColumn
            : Engine.TensorConcatenate(new[] { piColumn, new Tensor<T>(new[] { rows, columns - 1 }) }, 1);
        return Engine.TensorAdd(zeroMass, pmf).ToMatrix();
    }

    /// <summary>
    /// Initializes parameters from target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Proportion of zeros
        var yTensor = y.AsTensor();
        var isZero = Engine.TensorEquals(yTensor, NumOps.Zero);
        int numZeros = y.Length == 0 ? 0 : Engine.TensorCountNonzero(isZero);
        int countNonZero = y.Length - numZeros;
        T sumNonZero = countNonZero > 0
            ? Engine.TensorSum(Engine.TensorWhere(isZero, Engine.TensorZerosLike(yTensor), yTensor))
            : NumOps.Zero;

        T pZero = NumOps.Divide(NumOps.FromDouble(numZeros), NumOps.FromDouble(y.Length));
        T meanNonZero = countNonZero > 0
            ? NumOps.Divide(sumNonZero, NumOps.FromDouble(countNonZero))
            : NumOps.One;

        // Initialize zero model intercept (logit scale)
        // Clamp pZero to [0.01, 0.99]
        T minP = NumOps.FromDouble(0.01);
        T maxP = NumOps.FromDouble(0.99);
        if (NumOps.LessThan(pZero, minP)) pZero = minP;
        if (NumOps.GreaterThan(pZero, maxP)) pZero = maxP;
        _zeroIntercept = NumOps.Log(NumOps.Divide(pZero, NumOps.Subtract(NumOps.One, pZero)));

        // Initialize count model intercept (log scale)
        T minMean = NumOps.FromDouble(0.1);
        if (NumOps.LessThan(meanNonZero, minMean)) meanNonZero = minMean;
        _countIntercept = NumOps.Log(meanNonZero);

        // Initialize coefficients
        _countCoefficients = new Vector<T>(_numFeatures);
        if (_options.ModelZeroInflation)
        {
            _zeroCoefficients = new Vector<T>(_numFeatures);
        }

        // Initialize dispersion
        _dispersion = NumOps.One;
    }

    /// <summary>
    /// Computes predictions for all samples.
    /// </summary>
    private (Vector<T> lambdas, Vector<T> pis) ComputePredictions(Matrix<T> x)
    {
        int n = x.Rows;
        T minLambda = NumOps.FromDouble(0.001);
        if (n == 0)
        {
            return (new Vector<T>(0), new Vector<T>(0));
        }

        // Count model linear predictor
        var etaCount = LinearPredictor(x, _countCoefficients, _countIntercept);

        // Apply count link inverse
        Vector<T> lambdas = _options.CountLink switch
        {
            ZeroInflatedCountLink.Log => Engine.Exp(etaCount),
            ZeroInflatedCountLink.SquareRoot => Engine.Multiply(etaCount, etaCount),
            ZeroInflatedCountLink.Identity => RegressionEngineOps2.ClampMin(etaCount, minLambda),
            _ => Engine.Exp(etaCount)
        };

        // Zero-inflation model linear predictor
        var etaZero = LinearPredictor(
            x, _options.ModelZeroInflation ? _zeroCoefficients : null, _zeroIntercept);

        // Apply zero link inverse
        var etaZeroTensor = etaZero.AsTensor();
        Vector<T> pis = _options.ZeroLink switch
        {
            ZeroInflatedZeroLink.Logit => Logistic(etaZero),
            ZeroInflatedZeroLink.Probit => Engine.TensorNdtr(etaZeroTensor).ToVector(),
            ZeroInflatedZeroLink.CLogLog => OneMinus(Engine.Exp(Engine.Negate(Engine.Exp(etaZero)))),
            _ => Logistic(etaZero)
        };

        return (lambdas, pis);
    }

    /// <summary>
    /// Computes posterior probability of being a structural zero.
    /// </summary>
    private Vector<T> ComputePosteriorZero(Vector<T> y, Vector<T> lambdas, Vector<T> pis)
    {
        T half = NumOps.FromDouble(0.5);
        if (y.Length == 0)
        {
            return new Vector<T>(0);
        }

        // P(Z=1 | Y=0) = π / (π + (1-π) * P(Y=0|Z=0)), 1/2 when that denominator is not positive
        var piTensor = pis.AsTensor();
        var p0 = CountProbability(Engine.TensorZerosLike(piTensor), lambdas.AsTensor());
        var denom = Engine.TensorAdd(piTensor, Engine.TensorMultiply(OneMinus(pis).AsTensor(), p0));
        var zeroPosterior = Engine.TensorWhere(
            Engine.TensorGreaterThan(denom, NumOps.Zero),
            Engine.TensorDivide(piTensor, denom),
            Engine.Fill(y.Length, half).AsTensor());

        // If Y > 0, definitely not a structural zero
        var isZero = Engine.TensorEquals(y.AsTensor(), NumOps.Zero);
        return Engine.TensorWhere(isZero, zeroPosterior, Engine.TensorZerosLike(piTensor)).ToVector();
    }

    /// <summary>
    /// P_count(k | λ) element-wise for equally shaped k and λ tensors: the Poisson pmf
    /// exp(-λ + k log λ - log k!) (with P = [k == 0] when λ &lt;= 0), or the negative binomial pmf
    /// with the current dispersion r.
    /// </summary>
    private Tensor<T> CountProbability(Tensor<T> k, Tensor<T> lambda)
    {
        var logKFactorial = Engine.TensorLgamma(Engine.TensorAddScalar(k, NumOps.One));
        if (_options.DistributionFamily == ZeroInflatedDistributionFamily.Poisson)
        {
            // Poisson PMF
            var logPmf = Engine.TensorSubtract(
                Engine.TensorAdd(Engine.TensorNegate(lambda), Engine.TensorMultiply(k, Engine.TensorLog(lambda))),
                logKFactorial);
            return Engine.TensorWhere(
                Engine.TensorLessEqual(lambda, Engine.TensorZerosLike(lambda)),
                Engine.TensorEquals(k, NumOps.Zero),
                Engine.TensorExp(logPmf));
        }

        // Negative Binomial PMF
        T r = _dispersion;
        var p = Engine.TensorDivide(
            Engine.TensorAddScalar(Engine.TensorZerosLike(lambda), r), Engine.TensorAddScalar(lambda, r));
        T logGammaR = Engine.TensorLgamma(Engine.Fill(1, r).AsTensor())[0];
        var logPmfNb = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorSubtractScalar(
                    Engine.TensorSubtract(Engine.TensorLgamma(Engine.TensorAddScalar(k, r)), logKFactorial),
                    logGammaR),
                Engine.TensorMultiplyScalar(Engine.TensorLog(p), r)),
            Engine.TensorMultiply(k, Engine.TensorLog(Engine.ScalarMinusTensor(NumOps.One, p))));
        return Engine.TensorExp(logPmfNb);
    }

    /// <summary>
    /// intercept + x[:, :numFeatures] · coefficients (just the intercept when there are no coefficients).
    /// </summary>
    private Vector<T> LinearPredictor(Matrix<T> x, Vector<T>? coefficients, T intercept)
    {
        int n = x.Rows;
        if (coefficients is null || n == 0 || _numFeatures == 0)
        {
            return Engine.Fill(n, intercept);
        }

        var used = x.Columns == _numFeatures ? x : x.GetSubMatrix(0, 0, n, _numFeatures);
        return RegressionEngineOps2.AddScalar(Engine.MatrixVectorMultiply(used, coefficients), intercept);
    }

    /// <summary>1 - v element-wise.</summary>
    private Vector<T> OneMinus(Vector<T> values)
    {
        return Engine.ScalarMinusTensor(NumOps.One, values.AsTensor()).ToVector();
    }

    /// <summary>1 / (1 + exp(-eta)) element-wise.</summary>
    private Vector<T> Logistic(Vector<T> eta)
    {
        return Engine.Reciprocal(RegressionEngineOps2.AddScalar(Engine.Exp(Engine.Negate(eta)), NumOps.One));
    }


    /// <summary>
    /// Updates the count model parameters.
    /// </summary>
    private void UpdateCountModel(Matrix<T> x, Vector<T> y, Vector<T> posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minWeight = NumOps.FromDouble(1e-10);

        // Weighted IRLS for Poisson/NB regression
        var wi = RegressionEngineOps2.ClampMin(OneMinus(posteriorZero), minWeight);
        var eta = LinearPredictor(x, _countCoefficients, _countIntercept);
        var lambda = RegressionEngineOps2.ClampMin(Engine.Exp(eta), minWeight);

        // Working weight and response for log link
        var weights = Engine.Multiply(wi, lambda);
        var z = Engine.Add(eta, Engine.Divide(Engine.Subtract(y, lambda), lambda));

        if (_countCoefficients is null)
        {
            throw new InvalidOperationException("Count coefficients not initialized.");
        }
        UpdateCoefficientsWLS(x, z, weights, ref _countCoefficients, ref _countIntercept);
    }

    /// <summary>
    /// Updates the zero-inflation model parameters.
    /// </summary>
    private void UpdateZeroModel(Matrix<T> x, Vector<T> posteriorZero)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minPi = NumOps.FromDouble(1e-10);
        T maxPi = NumOps.FromDouble(1 - 1e-10);

        // Weighted logistic regression for zero model
        var eta = LinearPredictor(x, _zeroCoefficients, _zeroIntercept);

        // Logistic function: 1 / (1 + exp(-eta))
        var pi = Engine.Clamp(Logistic(eta), minPi, maxPi);

        // Working weight and response for logit link
        var weights = Engine.Multiply(pi, OneMinus(pi));
        var z = Engine.Add(eta, Engine.Divide(Engine.Subtract(posteriorZero, pi), weights));

        if (_zeroCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _zeroCoefficients, ref _zeroIntercept);
        }
    }

    /// <summary>
    /// Updates dispersion parameter for Negative Binomial.
    /// </summary>
    private void UpdateDispersion(Vector<T> y, Vector<T> lambdas, Vector<T> posteriorZero)
    {
        // Method of moments estimate
        T sumSqDev = NumOps.Zero;
        T sumLambda = NumOps.Zero;
        T totalWeight = NumOps.Zero;
        T minWeight = NumOps.FromDouble(1e-10);

        // Only rows with weight 1 - posterior >= minWeight contribute.
        var kept = KeptRows(OneMinus(posteriorZero), minWeight);
        if (kept.Length > 0)
        {
            var wi = RegressionEngineOps2.Gather(OneMinus(posteriorZero), kept);
            var keptLambdas = RegressionEngineOps2.Gather(lambdas, kept);
            var diff = Engine.Subtract(RegressionEngineOps2.Gather(y, kept), keptLambdas);
            sumSqDev = Engine.Sum(Engine.Multiply(wi, Engine.Multiply(diff, diff)));
            sumLambda = Engine.Sum(Engine.Multiply(wi, keptLambdas));
            totalWeight = Engine.Sum(wi);
        }

        if (NumOps.GreaterThan(totalWeight, NumOps.Zero) && NumOps.GreaterThan(sumLambda, NumOps.Zero))
        {
            T varianceEstimate = NumOps.Divide(sumSqDev, totalWeight);
            T meanEstimate = NumOps.Divide(sumLambda, totalWeight);

            // For NB: Var = μ + μ²/r, so r = μ² / (Var - μ)
            T excess = NumOps.Subtract(varianceEstimate, meanEstimate);
            T minExcess = NumOps.FromDouble(0.1);
            if (NumOps.GreaterThan(excess, minExcess))
            {
                T r = NumOps.Divide(NumOps.Multiply(meanEstimate, meanEstimate), excess);
                T minR = NumOps.FromDouble(0.1);
                T maxR = NumOps.FromDouble(1000);
                if (NumOps.LessThan(r, minR)) r = minR;
                if (NumOps.GreaterThan(r, maxR)) r = maxR;
                _dispersion = r;
            }
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, Vector<T> z, Vector<T> weights,
        ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;
        T minWeight = NumOps.FromDouble(1e-10);

        var xtwx = new Matrix<T>(p + 1, p + 1);
        var xtwz = new Vector<T>(p + 1);

        // Rows whose working weight is below minWeight are skipped.
        var kept = KeptRows(weights, minWeight);
        if (kept.Length > 0)
        {
            // Design [1 | x] over the kept rows, then [1 | x]ᵀ W [1 | x] and [1 | x]ᵀ W z.
            var keptX = RegressionEngineOps2.GatherRows(x, kept);
            if (keptX.Columns != p)
            {
                keptX = keptX.GetSubMatrix(0, 0, kept.Length, p);
            }

            var design = keptX.AddConstantColumn(NumOps.One);
            var w = RegressionEngineOps2.Gather(weights, kept);
            var weightedDesign = Engine.TensorMultiply(
                Tensor<T>.FromMatrix(design),
                Engine.TensorBroadcastTo(Engine.Reshape(w.AsTensor(), new[] { kept.Length, 1 }), new[] { kept.Length, p + 1 }))
                .ToMatrix();
            var designT = Engine.MatrixTranspose(design);
            xtwx = RegressionEngineOps2.SymmetricFromLower(Engine.MatrixMultiply(designT, weightedDesign));
            xtwz = Engine.MatrixVectorMultiply(Engine.MatrixTranspose(weightedDesign), RegressionEngineOps2.Gather(z, kept));
        }

        // Regularization
        if (_options.UseRegularization)
        {
            T lambda = NumOps.FromDouble(_options.RegularizationStrength);
            xtwx = RegressionEngineOps2.AddDiagonal(
                xtwx, Vector<T>.Concatenate(Engine.FillZero<T>(1), Engine.Fill(p, lambda)));
        }

        var solution = SolveSystem(xtwx, xtwz, p + 1);

        intercept = solution[0];
        coefficients = RegressionEngineOps2.Slice(solution, 1, p);
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private T ComputeLogLikelihood(Vector<T> y, Vector<T> lambdas, Vector<T> pis)
    {
        T tiny = NumOps.FromDouble(1e-300);
        if (y.Length == 0)
        {
            return NumOps.Zero;
        }

        // y has already been coerced to non-negative integer counts, so it is its own count k.
        var yTensor = y.AsTensor();
        var piTensor = pis.AsTensor();
        var countPart = Engine.TensorAddScalar(
            Engine.TensorMultiply(OneMinus(pis).AsTensor(), CountProbability(yTensor, lambdas.AsTensor())), tiny);

        // y == 0: π + ((1-π) P(0) + tiny); y > 0: (1-π) P(y) + tiny
        var likelihood = Engine.TensorWhere(
            Engine.TensorEquals(yTensor, NumOps.Zero), Engine.TensorAdd(piTensor, countPart), countPart);
        return Engine.TensorSum(Engine.TensorLog(likelihood));
    }

    private Vector<T> SolveSystem(Matrix<T> a, Vector<T> b, int n)
    {
        T pivotThreshold = NumOps.FromDouble(1e-10);
        return RegressionEngineOps2.GaussJordanSolve(a, b, n, pivotThreshold);
    }

    /// <summary>
    /// Positions whose weight is not below <paramref name="minWeight"/> (NaN is kept, as a
    /// "skip when w &lt; minWeight" scan would keep it), ascending.
    /// </summary>
    private int[] KeptRows(Vector<T> weights, T minWeight)
    {
        if (weights.Length == 0)
        {
            return Array.Empty<int>();
        }

        var below = Engine.TensorLessThan(weights.AsTensor(), minWeight);
        return Engine.TensorNonzero(Engine.ScalarMinusTensor(NumOps.One, below)).ToArray();
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = Engine.FillZero<T>(_numFeatures);
        if (_numFeatures > 0)
        {
            if (_countCoefficients != null)
                importances = Engine.Add(importances, Engine.Abs(_countCoefficients.GetSubVector(0, _numFeatures)));
            if (_zeroCoefficients != null)
                importances = Engine.Add(importances, Engine.Abs(_zeroCoefficients.GetSubVector(0, _numFeatures)));

            T sum = Engine.Sum(importances);
            if (NumOps.GreaterThan(sum, NumOps.Zero))
            {
                importances = Engine.Divide(importances, sum);
            }
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "DistributionFamily", _options.DistributionFamily.ToString() },
                { "ModelZeroInflation", _options.ModelZeroInflation },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }
}
