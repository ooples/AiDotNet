using AiDotNet.Attributes;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Beta Regression for modeling proportions and rates bounded in (0, 1).
/// </summary>
/// <remarks>
/// <para>
/// Beta Regression is the appropriate choice when your response variable is a continuous
/// proportion or rate that must fall strictly between 0 and 1. It uses the Beta distribution
/// and can model both the mean and precision as functions of covariates.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you need to predict proportions (like percentages),
/// regular regression can give impossible results (negative values or values > 1).
/// Beta Regression fixes this by:
///
/// 1. Always producing valid predictions between 0 and 1
/// 2. Naturally handling skewed proportions
/// 3. Allowing varying uncertainty (some predictions more reliable than others)
///
/// Example applications:
/// - Predicting market share (e.g., "37% market share")
/// - Modeling test pass rates
/// - Estimating probability scores
/// - Analyzing biological concentrations
///
/// The model uses a "link function" to transform proportions to a scale where linear
/// modeling works, then transforms predictions back to valid proportions.
/// </para>
/// <para>
/// Reference: Ferrari, S.L.P., Cribari-Neto, F. (2004). "Beta Regression for
/// Modelling Rates and Proportions". Journal of Applied Statistics, 31(7), 799-815.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Beta regression for modeling proportions (values in 0-1)
/// var options = new BetaRegressionOptions();
///
/// // Prepare training data: 5 samples with 2 features, targets are proportions
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 0.15, 0.35, 0.50, 0.72, 0.88 });
///
/// // Train the model (predictions will be bounded between 0 and 1)
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BetaRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict a proportion for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 11, 12 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Beta Regression for Modelling Rates and Proportions", "https://doi.org/10.1080/0266476042000214501", Year = 2004, Authors = "Silvia L. P. Ferrari, Francisco Cribari-Neto")]
public partial class BetaRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private const double MuFloor = 1e-10;
    private const double MuCeiling = 1.0 - 1e-10;
    private const double MinFisherValue = 0.1;
    private const double PivotThreshold = 1e-10;

    /// <summary>
    /// Coefficients for the mean (μ) model.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _meanCoefficients;

    /// <summary>
    /// Intercept for the mean model.
    /// </summary>

    private T _meanIntercept;


    /// <summary>
    /// Coefficients for the precision (φ) model (if variable precision).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _precisionCoefficients;

    /// <summary>
    /// Intercept for the precision model.
    /// </summary>

    private T _precisionIntercept;


    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly BetaRegressionOptions _options;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    // SupportsParameterInitialization=false is inherited from AsyncDecisionTreeRegressionBase

    /// <summary>
    /// Gets the mean model coefficients.
    /// </summary>
    public Vector<T>? MeanCoefficients => _meanCoefficients;

    /// <summary>
    /// Gets the mean model intercept.
    /// </summary>
    public T MeanIntercept => _meanIntercept;

    /// <summary>
    /// Gets the precision model coefficients (if variable precision is enabled).
    /// </summary>
    public Vector<T>? PrecisionCoefficients => _precisionCoefficients;

    /// <summary>
    /// Gets the precision (or its intercept if constant).
    /// </summary>
    public T Precision => _precisionIntercept;

    /// <summary>
    /// Initializes a new instance of BetaRegression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public BetaRegression(BetaRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new BetaRegressionOptions();
        _meanIntercept = NumOps.Zero;
        _precisionIntercept = NumOps.FromDouble(Math.Log(10));  // Initial precision = 10
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;

        // Beta regression (Ferrari & Cribari-Neto 2004) models a proportion in (0,1) DIRECTLY: mu is the
        // mean of the response on its own scale, and the logit link already maps the linear predictor into
        // (0,1). Three separate substitutions used to sit here instead, and they compounded:
        //
        //   1. A target reaching outside [0,1] set `_useOLS = true` and fitted ordinary least squares, so a
        //      caller asking for beta regression received a linear model reporting itself as one.
        //   2. A target already inside [0,1] was MIN-MAX RESCALED onto (0.01, 0.99). That is not a boundary
        //      fix, it is a distortion: a genuine proportion range of [0.3, 0.7] was stretched to span
        //      almost the whole interval, so the fitted mean described data the caller never supplied.
        //   3. Predict inverted that rescaling only `if (_needsTransform)` -- but _needsTransform was true
        //      only on the OLS path, so on the beta path it was ALWAYS false. The rescaling was therefore
        //      never undone, and both PredictAsync and PredictDistributionsAsync reported the distorted
        //      scale as if it were the caller's.
        //
        // All three are gone. The target is validated as a proportion and used as-is, so predictions come
        // back on the scale the model was trained on and no inverse transform is needed.
        y = ValidationHelper<T>.ValidateProportionTarget(
            y, nameof(BetaRegression<T>), _options.CompressBoundaryValues);

        // Initialize parameters
        InitializeParameters(y);

        T prevLogLik = NumOps.MinValue;
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        // Fisher scoring / IRLS
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // Compute current predictions
            var (mus, phis) = ComputePredictions(x);

            // Update mean model coefficients
            UpdateMeanModel(x, y, mus, phis);

            // Update precision model if variable
            if (_options.ModelVariablePrecision)
            {
                UpdatePrecisionModel(x, y, mus, phis);
            }

            // Check convergence
            (mus, phis) = ComputePredictions(x);
            T logLik = ComputeLogLikelihood(y, mus, phis);

            if (HasConverged(logLik, prevLogLik, tolerance))
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
        // mu is already the predicted proportion on the caller's own scale: the logit link maps the linear
        // predictor into (0,1), and training no longer rescales the target, so there is nothing to invert.
        var (mus, _) = await Task.Run(() => ComputePredictions(input));

        return mus;
    }

    /// <summary>
    /// Predicts full Beta distributions for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Array of predicted Beta distributions.</returns>
    public async Task<IParametricDistribution<T>[]> PredictDistributionsAsync(Matrix<T> input)
    {
        var (mus, phis) = await Task.Run(() => ComputePredictions(input));
        if (input.Rows == 0)
        {
            return new IParametricDistribution<T>[0];
        }

        // Convert (mu, phi) to (alpha, beta) parameterization
        var alphas = Engine.Multiply(mus, phis);
        var betas = Engine.Multiply(Engine.Subtract(Engine.Fill(mus.Length, NumOps.One), mus), phis);

        return alphas.Zip(betas, (alpha, beta) => (IParametricDistribution<T>)new BetaDistribution<T>(alpha, beta)).ToArray();
    }

    /// <summary>
    /// Gets prediction intervals for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="confidenceLevel">Confidence level (default 0.95).</param>
    /// <returns>Tuple of (lower bounds, upper bounds).</returns>
    public async Task<(Vector<T> Lower, Vector<T> Upper)> PredictIntervalAsync(Matrix<T> input, double confidenceLevel = 0.95)
    {
        var distributions = await PredictDistributionsAsync(input);

        double alpha = 1 - confidenceLevel;
        T alphaLower = NumOps.FromDouble(alpha / 2);
        T alphaUpper = NumOps.FromDouble(1 - alpha / 2);

        var lower = new Vector<T>(distributions.Select(distribution => distribution.InverseCdf(alphaLower)));
        var upper = new Vector<T>(distributions.Select(distribution => distribution.InverseCdf(alphaUpper)));

        return (lower, upper);
    }

    /// <summary>
    /// Initializes parameters from target values.
    /// </summary>
    private void InitializeParameters(Vector<T> y)
    {
        // Initialize mean intercept using empirical logit
        var oneMinusY = Engine.Subtract(Engine.Fill(y.Length, NumOps.One), y);
        T sumLogit = y.Length > 0 ? Engine.Sum(Engine.Log(Engine.Divide(y, oneMinusY))) : NumOps.Zero;
        _meanIntercept = NumOps.Divide(sumLogit, NumOps.FromDouble(y.Length));

        // Initialize mean coefficients to zero
        _meanCoefficients = new Vector<T>(_numFeatures);

        // Initialize precision
        if (_options.ModelVariablePrecision)
        {
            _precisionCoefficients = new Vector<T>(_numFeatures);
        }
    }

    /// <summary>
    /// Computes mean (μ) and precision (φ) predictions for all samples.
    /// </summary>
    private (Vector<T> mus, Vector<T> phis) ComputePredictions(Matrix<T> x)
    {
        int n = x.Rows;
        T minPhi = NumOps.FromDouble(0.1);

        // Linear predictor for mean
        var etaMu = LinearPredictor(x, _meanCoefficients, _meanIntercept);

        // Apply link function inverse and clamp away from beta distribution endpoints
        var mus = ClampMu(InverseLinkFunction(etaMu));

        // Linear predictor for precision
        var etaPhi = LinearPredictor(x, _options.ModelVariablePrecision ? _precisionCoefficients : null, _precisionIntercept);

        // Precision uses log link, floored at 0.1 (NaN is left as is)
        var phi = Tensor<T>.FromVector(Engine.Exp(etaPhi));
        var phis = Engine.TensorWhere(
            Engine.TensorLessThan(phi, minPhi), Tensor<T>.FromVector(Engine.Fill(n, minPhi)), phi).ToVector();

        return (mus, phis);
    }

    /// <summary>
    /// intercept + x[:, :numFeatures] * coefficients for every row (just the intercept when there are
    /// no coefficients).
    /// </summary>
    private Vector<T> LinearPredictor(Matrix<T> x, Vector<T>? coefficients, T intercept)
    {
        int n = x.Rows;
        var eta = Engine.Fill(n, intercept);
        if (coefficients is null || n == 0 || _numFeatures == 0)
        {
            return eta;
        }

        var design = x.Columns == _numFeatures ? x : x.GetSubMatrix(0, 0, n, _numFeatures);
        return Engine.Add(eta, Engine.MatrixVectorMultiply(design, coefficients.GetSubVector(0, _numFeatures)));
    }

    /// <summary>
    /// Clamps every mean into [MuFloor, MuCeiling].
    /// </summary>
    private Vector<T> ClampMu(Vector<T> mu)
    {
        return Engine.Clamp(mu, NumOps.FromDouble(MuFloor), NumOps.FromDouble(MuCeiling));
    }

    /// <summary>
    /// Whether the log-likelihood changed by less than the tolerance.
    /// </summary>
    private bool HasConverged(T logLik, T prevLogLik, T tolerance)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(logLik, prevLogLik)), tolerance);
    }

    /// <summary>
    /// Updates the mean model using Fisher scoring.
    /// </summary>
    private void UpdateMeanModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;

        // Working weights and adjusted dependent variable.
        //
        // LinkFunctionDerivative returns g'(mu) = d(eta)/d(mu) -- for the logit that is
        // 1/(mu(1-mu)), NOT mu(1-mu). The local was named `dmu`, and both formulas below used it as
        // though it were d(mu)/d(eta), which inverted the role of the link derivative in each.
        //
        // Fisher scoring for a GLM with variance function V(mu) uses
        //
        //     w = 1 / (V(mu) * g'(mu)^2)          z = eta + (y - mu) * g'(mu)
        //
        // and for the Beta mean model V(mu) = mu(1-mu) (Ferrari & Cribari-Neto 2004). The precision
        // phi scales every weight equally, so it does not change the weighted least-squares solution.
        //
        // What was here instead:
        //
        //     w = phi * V / g'^2   which for the logit is phi * V^3, not phi * V
        //     z = eta + (y - mu) * g' / V          an extra factor of 1/V
        //
        // The working response was the fatal one. At mu = 0.1, V is 0.09, so every Fisher step
        // overshot by more than tenfold; the coefficients ran away and the fitted mean saturated at
        // 0 or 1 for every observation. That went unnoticed because a target outside [0,1] was
        // silently diverted to ordinary least squares, and one inside it was first min-max rescaled,
        // so no test ever asked this path for a prediction it could check.
        var gPrime = LinkFunctionDerivative(mus);
        var v = Engine.Multiply(mus, Engine.Subtract(Engine.Fill(n, NumOps.One), mus));
        var variance = Engine.Multiply(v, Engine.Multiply(gPrime, gPrime));
        var weights = Engine.Divide(phis, variance);

        // Working response
        var eta = LinkFunction(mus);
        var z = Engine.Add(eta, Engine.Multiply(Engine.Subtract(y, mus), gPrime));

        // Weighted least squares
        if (_meanCoefficients is null)
        {
            throw new InvalidOperationException("Mean coefficients not initialized.");
        }

        UpdateCoefficientsWLS(x, z, weights, ref _meanCoefficients, ref _meanIntercept);
    }

    /// <summary>
    /// Updates the precision model using Fisher scoring.
    /// </summary>
    private void UpdatePrecisionModel(Matrix<T> x, Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        int n = x.Rows;
        var ones = Engine.Fill(n, NumOps.One);
        var oneMinusMu = Engine.Subtract(ones, mus);
        var alphas = Engine.Multiply(mus, phis);
        var betas = Engine.Multiply(oneMinusMu, phis);

        // Score for phi: d log L / d log(phi)
        var ystar = Engine.Log(Engine.Divide(y, Engine.Subtract(ones, y)));
        var mustar = Engine.Subtract(Digamma(alphas), Digamma(betas));
        var score = Engine.Multiply(mus, Engine.Subtract(ystar, mustar));

        // Fisher information for log(phi), floored at MinFisherValue (NaN is left as is)
        var trigammaTerm = Engine.Add(
            Engine.Multiply(Engine.Multiply(mus, mus), Trigamma(alphas)),
            Engine.Multiply(Engine.Multiply(oneMinusMu, oneMinusMu), Trigamma(betas)));
        var fisher = Tensor<T>.FromVector(Engine.Multiply(Engine.Multiply(phis, phis), trigammaTerm));
        T minFisher = NumOps.FromDouble(MinFisherValue);
        var fisherInfo = Engine.TensorWhere(
            Engine.TensorLessThan(fisher, minFisher), Tensor<T>.FromVector(Engine.Fill(n, minFisher)), fisher).ToVector();

        var weights = fisherInfo;
        var z = Engine.Add(Engine.Log(phis), Engine.Divide(score, fisherInfo));

        if (_precisionCoefficients != null)
        {
            UpdateCoefficientsWLS(x, z, weights, ref _precisionCoefficients, ref _precisionIntercept);
        }
    }

    /// <summary>
    /// Updates coefficients using weighted least squares.
    /// </summary>
    private void UpdateCoefficientsWLS(Matrix<T> x, Vector<T> z, Vector<T> weights, ref Vector<T> coefficients, ref T intercept)
    {
        int n = x.Rows;
        int p = _numFeatures;

        // X'WX and X'Wz, with the intercept column first
        var design = (x.Columns == p ? x : x.GetSubMatrix(0, 0, n, p)).AddConstantColumn(NumOps.One);
        var weightColumn = Engine.TensorBroadcastTo(new Tensor<T>(new[] { n, 1 }, weights), new[] { n, p + 1 });
        var weightedDesign = Engine.TensorMultiply(Tensor<T>.FromMatrix(design), weightColumn).ToMatrix();
        var designTranspose = Engine.MatrixTranspose(design);
        var xtwx = Engine.MatrixMultiply(designTranspose, weightedDesign);
        var xtwz = Engine.MatrixVectorMultiply(designTranspose, Engine.Multiply(weights, z));

        // Regularization (not on the intercept)
        if (_options.UseRegularization)
        {
            T lambda = NumOps.FromDouble(_options.RegularizationStrength);
            var penalty = Vector<T>.Concatenate(new[] { Engine.FillZero<T>(1), Engine.Fill(p, lambda) });
            xtwx = Engine.MatrixAdd(xtwx, Matrix<T>.CreateDiagonal(penalty));
        }

        // Solve
        var solution = SolveSystem(xtwx, xtwz, p + 1);

        // Update with learning rate
        T lr = NumOps.FromDouble(_options.LearningRate);
        T oneMinusLr = NumOps.Subtract(NumOps.One, lr);

        intercept = NumOps.Add(
            NumOps.Multiply(oneMinusLr, intercept),
            NumOps.Multiply(lr, solution[0]));

        coefficients = Engine.Add(
            Engine.Multiply(coefficients, oneMinusLr),
            Engine.Multiply(solution.GetSubVector(1, p), lr));
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private T ComputeLogLikelihood(Vector<T> y, Vector<T> mus, Vector<T> phis)
    {
        if (y.Length == 0)
        {
            return NumOps.Zero;
        }

        var ones = Engine.Fill(y.Length, NumOps.One);
        var alphas = Engine.Multiply(mus, phis);
        var betas = Engine.Multiply(Engine.Subtract(ones, mus), phis);

        // Log Beta PDF: lgamma(phi) - lgamma(alpha) - lgamma(beta) + (alpha - 1) log y + (beta - 1) log(1 - y)
        var logPdf = Engine.Subtract(Engine.Subtract(LogGamma(phis), LogGamma(alphas)), LogGamma(betas));
        logPdf = Engine.Add(logPdf, Engine.Multiply(Engine.Subtract(alphas, ones), Engine.Log(y)));
        logPdf = Engine.Add(logPdf, Engine.Multiply(Engine.Subtract(betas, ones), Engine.Log(Engine.Subtract(ones, y))));

        return Engine.Sum(logPdf);
    }

    /// <summary>
    /// Applies the link function.
    /// </summary>
    private Vector<T> LinkFunction(Vector<T> mu)
    {
        mu = ClampMu(mu);
        var ones = Engine.Fill(mu.Length, NumOps.One);
        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => Engine.Log(Engine.Divide(mu, Engine.Subtract(ones, mu))),
            BetaLinkFunction.Probit => InverseStandardNormalCdf(mu),
            BetaLinkFunction.CLogLog => Engine.Log(Engine.Multiply(Engine.Log(Engine.Subtract(ones, mu)), NumOps.Negate(NumOps.One))),
            BetaLinkFunction.Log => Engine.Log(mu),
            _ => Engine.Log(Engine.Divide(mu, Engine.Subtract(ones, mu)))
        };
    }

    /// <summary>
    /// Applies the inverse link function.
    /// </summary>
    private Vector<T> InverseLinkFunction(Vector<T> eta)
    {
        var ones = Engine.Fill(eta.Length, NumOps.One);
        T minusOne = NumOps.Negate(NumOps.One);
        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => Engine.Reciprocal(Engine.Add(ones, Engine.Exp(Engine.Multiply(eta, minusOne)))),
            BetaLinkFunction.Probit => StandardNormalCdf(eta),
            BetaLinkFunction.CLogLog => Engine.Subtract(ones, Engine.Exp(Engine.Multiply(Engine.Exp(eta), minusOne))),
            BetaLinkFunction.Log => Engine.Exp(eta),
            _ => Engine.Reciprocal(Engine.Add(ones, Engine.Exp(Engine.Multiply(eta, minusOne))))
        };
    }

    /// <summary>
    /// Computes the derivative of the link function.
    /// </summary>
    private Vector<T> LinkFunctionDerivative(Vector<T> mu)
    {
        mu = ClampMu(mu);
        var oneMinusMu = Engine.Subtract(Engine.Fill(mu.Length, NumOps.One), mu);
        return _options.LinkFunction switch
        {
            BetaLinkFunction.Logit => Engine.Reciprocal(Engine.Multiply(mu, oneMinusMu)),
            BetaLinkFunction.Probit => Engine.Reciprocal(StandardNormalPdf(InverseStandardNormalCdf(mu))),
            BetaLinkFunction.CLogLog => Engine.Reciprocal(Engine.Multiply(oneMinusMu, Engine.Log(oneMinusMu))),
            BetaLinkFunction.Log => Engine.Reciprocal(mu),
            _ => Engine.Reciprocal(Engine.Multiply(mu, oneMinusMu))
        };
    }

    private Vector<T> StandardNormalCdf(Vector<T> z)
    {
        // 0.5 * (1 + erf(z / sqrt(2)))
        var erf = Erf(Engine.Divide(z, NumOps.FromDouble(Math.Sqrt(2))));
        return Engine.Multiply(Engine.Add(Engine.Fill(z.Length, NumOps.One), erf), NumOps.FromDouble(0.5));
    }

    private Vector<T> StandardNormalPdf(Vector<T> z)
    {
        // exp(-0.5 * z * z) / sqrt(2 pi)
        var exponent = Engine.Multiply(Engine.Multiply(z, NumOps.FromDouble(-0.5)), z);
        return Engine.Divide(Engine.Exp(exponent), NumOps.FromDouble(Math.Sqrt(2 * Math.PI)));
    }

    private Vector<T> InverseStandardNormalCdf(Vector<T> p)
    {
        // Rational approximation (Abramowitz & Stegun 26.2.23); -inf at p <= 0, +inf at p >= 1
        int n = p.Length;
        var ones = Engine.Fill(n, NumOps.One);
        var pTensor = Tensor<T>.FromVector(p);
        var lowerTail = Engine.TensorLessThan(pTensor, NumOps.FromDouble(0.5));
        var tail = Engine.TensorWhere(lowerTail, pTensor, Tensor<T>.FromVector(Engine.Subtract(ones, p))).ToVector();
        var t = Engine.Sqrt(Engine.Multiply(Engine.Log(tail), NumOps.FromDouble(-2)));

        T c0 = NumOps.FromDouble(2.515517), c1 = NumOps.FromDouble(0.802853), c2 = NumOps.FromDouble(0.010328);
        T d1 = NumOps.FromDouble(1.432788), d2 = NumOps.FromDouble(0.189269), d3 = NumOps.FromDouble(0.001308);

        // c0 + c1 t + c2 t t
        var numerator = Engine.Add(
            Engine.Add(Engine.Fill(n, c0), Engine.Multiply(t, c1)),
            Engine.Multiply(Engine.Multiply(t, c2), t));

        // 1 + d1 t + d2 t t + d3 t t t
        var denominator = Engine.Add(
            Engine.Add(Engine.Add(ones, Engine.Multiply(t, d1)), Engine.Multiply(Engine.Multiply(t, d2), t)),
            Engine.Multiply(Engine.Multiply(Engine.Multiply(t, d3), t), t));

        var result = Tensor<T>.FromVector(Engine.Subtract(t, Engine.Divide(numerator, denominator)));
        var signed = Engine.TensorWhere(lowerTail, Engine.TensorNegate(result), result);

        // Saturate outside (0, 1)
        var negativeInfinity = Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.NegativeInfinity)));
        var positiveInfinity = Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.PositiveInfinity)));
        signed = Engine.TensorWhere(
            Engine.TensorGreaterEqual(pTensor, Tensor<T>.FromVector(ones)), positiveInfinity, signed);
        signed = Engine.TensorWhere(
            Engine.TensorLessEqual(pTensor, Tensor<T>.FromVector(Engine.FillZero<T>(n))), negativeInfinity, signed);
        return signed.ToVector();
    }

    private Vector<T> Erf(Vector<T> x)
    {
        // Abramowitz & Stegun 7.1.26
        int n = x.Length;
        var xTensor = Tensor<T>.FromVector(x);
        var negative = Engine.TensorLessThan(xTensor, NumOps.Zero);
        var ax = Engine.Abs(x);
        var t = Engine.Reciprocal(Engine.Add(Engine.Fill(n, NumOps.One), Engine.Multiply(ax, NumOps.FromDouble(0.3275911))));

        // ((((1.061405429 t - 1.453152027) t + 1.421413741) t - 0.284496736) t + 0.254829592) t exp(-x x)
        var poly = Engine.Subtract(Engine.Multiply(t, NumOps.FromDouble(1.061405429)), Engine.Fill(n, NumOps.FromDouble(1.453152027)));
        poly = Engine.Add(Engine.Multiply(poly, t), Engine.Fill(n, NumOps.FromDouble(1.421413741)));
        poly = Engine.Subtract(Engine.Multiply(poly, t), Engine.Fill(n, NumOps.FromDouble(0.284496736)));
        poly = Engine.Add(Engine.Multiply(poly, t), Engine.Fill(n, NumOps.FromDouble(0.254829592)));
        var tail = Engine.Multiply(Engine.Multiply(poly, t), Engine.Exp(Engine.Multiply(Engine.Multiply(ax, ax), NumOps.Negate(NumOps.One))));
        var y = Tensor<T>.FromVector(Engine.Subtract(Engine.Fill(n, NumOps.One), tail));

        return Engine.TensorWhere(negative, Engine.TensorNegate(y), y).ToVector();
    }

    private Vector<T> LogGamma(Vector<T> x)
    {
        // Lanczos approximation (Numerical Recipes gammln); +inf for x <= 0
        int n = x.Length;
        // Literal Lanczos coefficients, converted to T once before the recurrence.
        T[] c =
        {
            NumOps.FromDouble(76.18009172947146), NumOps.FromDouble(-86.50532032941677),
            NumOps.FromDouble(24.01409824083091), NumOps.FromDouble(-1.231739572450155),
            NumOps.FromDouble(0.1208650973866179e-2), NumOps.FromDouble(-0.5395239384953e-5)
        };

        var y = x;
        var tmp = Engine.Add(x, Engine.Fill(n, NumOps.FromDouble(5.5)));
        tmp = Engine.Subtract(tmp, Engine.Multiply(Engine.Add(x, Engine.Fill(n, NumOps.FromDouble(0.5))), Engine.Log(tmp)));

        var ones = Engine.Fill(n, NumOps.One);
        var ser = Engine.Fill(n, NumOps.FromDouble(1.000000000190015));
        foreach (T coefficient in c)
        {
            y = Engine.Add(y, ones);
            ser = Engine.Add(ser, Engine.Divide(Engine.Fill(n, coefficient), y));
        }

        var result = Tensor<T>.FromVector(Engine.Add(
            Engine.Multiply(tmp, NumOps.Negate(NumOps.One)),
            Engine.Log(Engine.Divide(Engine.Multiply(ser, NumOps.FromDouble(2.5066282746310005)), x))));

        return Engine.TensorWhere(
            RegressionEngineOps.LessOrEqualMask(x, NumOps.Zero),
            Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.PositiveInfinity))),
            result).ToVector();
    }

    private Vector<T> Digamma(Vector<T> x)
    {
        // Recurrence up to x >= 6, then the asymptotic series; NaN for x <= 0
        int n = x.Length;
        var (shifted, correction) = ShiftToSix(x, square: false);
        var result = Engine.Multiply(correction, NumOps.Negate(NumOps.One));

        // log x - 1/(2x) - 1/(12 x x) + 1/(120 x x x x)
        var series = Engine.Subtract(Engine.Log(shifted), Engine.Reciprocal(Engine.Multiply(shifted, NumOps.FromDouble(2))));
        series = Engine.Subtract(series, Engine.Reciprocal(Engine.Multiply(Engine.Multiply(shifted, NumOps.FromDouble(12)), shifted)));
        series = Engine.Add(series, Engine.Reciprocal(
            Engine.Multiply(Engine.Multiply(Engine.Multiply(Engine.Multiply(shifted, NumOps.FromDouble(120)), shifted), shifted), shifted)));
        result = Engine.Add(result, series);

        return Engine.TensorWhere(
            RegressionEngineOps.LessOrEqualMask(x, NumOps.Zero),
            Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.NaN))),
            Tensor<T>.FromVector(result)).ToVector();
    }

    /// <summary>
    /// Applies the recurrence x -> x + 1 while x &lt; 6 (at most six steps for x &gt; 0), returning
    /// the shifted argument and the accumulated sum of 1/x (or 1/x^2 when <paramref name="square"/>).
    /// </summary>
    private (Vector<T> Shifted, Vector<T> Correction) ShiftToSix(Vector<T> x, bool square)
    {
        int n = x.Length;
        var ones = Engine.Fill(n, NumOps.One);
        var zeros = Tensor<T>.FromVector(Engine.FillZero<T>(n));
        var correction = Engine.FillZero<T>(n);
        var current = x;
        T six = NumOps.FromDouble(6);
        for (int step = 0; step < 6; step++)
        {
            var active = Engine.TensorLessThan(Tensor<T>.FromVector(current), six);
            var term = Engine.Reciprocal(square ? Engine.Multiply(current, current) : current);
            correction = Engine.Add(correction, Engine.TensorWhere(active, Tensor<T>.FromVector(term), zeros).ToVector());
            current = Engine.TensorWhere(
                active, Tensor<T>.FromVector(Engine.Add(current, ones)), Tensor<T>.FromVector(current)).ToVector();
        }

        return (current, correction);
    }

    private Vector<T> Trigamma(Vector<T> x)
    {
        // Recurrence up to x >= 6, then the asymptotic series; NaN for x <= 0
        int n = x.Length;
        var (shifted, result) = ShiftToSix(x, square: true);

        // 1/x + 1/(2 x x) + 1/(6 x x x)
        var series = Engine.Add(
            Engine.Reciprocal(shifted),
            Engine.Reciprocal(Engine.Multiply(Engine.Multiply(shifted, NumOps.FromDouble(2)), shifted)));
        series = Engine.Add(series, Engine.Reciprocal(
            Engine.Multiply(Engine.Multiply(Engine.Multiply(shifted, NumOps.FromDouble(6)), shifted), shifted)));
        result = Engine.Add(result, series);

        return Engine.TensorWhere(
            RegressionEngineOps.LessOrEqualMask(x, NumOps.Zero),
            Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.NaN))),
            Tensor<T>.FromVector(result)).ToVector();
    }

    private Vector<T> SolveSystem(Matrix<T> a, Vector<T> b, int n)
    {
        // Augmented [A | b]
        var aug = Engine.TensorConcatenate(
            new[]
            {
                Tensor<T>.FromMatrix(a.GetSubMatrix(0, 0, n, n)),
                new Tensor<T>(new[] { n, 1 }, b.GetSubVector(0, n))
            }, 1).ToMatrix();

        T pivotThreshold = NumOps.FromDouble(PivotThreshold);

        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            aug = RegressionEngineOps.GaussJordanStep(aug, col, n, pivotThreshold);
        }

        return aug.GetColumn(n);
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);
        foreach (var coefficients in new[] { _meanCoefficients, _precisionCoefficients })
        {
            if (coefficients != null && _numFeatures > 0)
            {
                importances = Engine.Add(importances, Engine.Abs(coefficients.GetSubVector(0, _numFeatures)));
            }
        }

        T sum = _numFeatures > 0 ? Engine.Sum(importances) : NumOps.Zero;
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = Engine.Divide(importances, sum);
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
                { "LinkFunction", _options.LinkFunction.ToString() },
                { "ModelVariablePrecision", _options.ModelVariablePrecision },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }

    private void WriteVector(BinaryWriter w, Vector<T>? v)
    {
        w.Write(v != null);
        if (v != null)
        {
            w.Write(v.Length);
            for (int i = 0; i < v.Length; i++) w.Write(NumOps.ToDouble(v[i]));
        }
    }

    private Vector<T>? ReadVector(BinaryReader r)
    {
        if (!r.ReadBoolean()) return null;
        int len = r.ReadInt32();
        var v = new Vector<T>(len);
        for (int i = 0; i < len; i++) v[i] = NumOps.FromDouble(r.ReadDouble());
        return v;
    }
}
