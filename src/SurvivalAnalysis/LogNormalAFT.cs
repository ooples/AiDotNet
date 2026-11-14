using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

using AiDotNet.Models.Parameters;
namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements the Log-Normal Accelerated Failure Time (AFT) model for survival analysis.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Log-Normal AFT model assumes that the log of survival times
/// follows a normal distribution. This is appropriate when survival times have a bell-curve shape
/// after log-transformation, common in many biomedical and engineering applications.</para>
///
/// <para><b>The Model:</b>
/// log(T) = β₀ + β₁X₁ + ... + βₚXₚ + σε
/// where ε ~ N(0,1) is standard normal.</para>
///
/// <para><b>Interpretation:</b>
/// <list type="bullet">
/// <item>exp(β) gives the multiplicative effect on median survival time</item>
/// <item>A positive coefficient increases survival time (protective)</item>
/// <item>A negative coefficient decreases survival time (harmful)</item>
/// </list>
/// </para>
///
/// <para><b>When to use Log-Normal vs Weibull:</b>
/// <list type="bullet">
/// <item>Log-Normal: Hazard first increases then decreases (non-monotonic)</item>
/// <item>Weibull: Hazard is monotonic (increasing, decreasing, or constant)</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Lawless (2003), Statistical Models and Methods for Lifetime Data</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <example>
/// <code>
/// // age and treatment arm per patient
/// var features = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// // months each patient was observed
/// var times = new Vector&lt;double&gt;(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0 });
/// // 1 = the event happened; 0 = censored, still fine when the study ended or lost to follow-up
/// var events = new Vector&lt;int&gt;(new int[] { 1, 0, 1, 0, 1, 1 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LogNormalAFT&lt;double&gt;(maxIterations: 100, tolerance: 1e-6))
///     .Build(features, times, events);
///
/// var risk = result.Predict(features);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.SurvivalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Survival Analysis: Techniques for Censored and Truncated Data", "https://doi.org/10.1007/978-1-4757-3294-8")]
public partial class LogNormalAFT<T> : SurvivalModelBase<T>
{

    /// <inheritdoc />
    /// <remarks>Intercept, scale, then the covariate coefficients, the same layout as WeibullAFT -- the two differ in their error distribution, not in what they fit.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(
            "parameters/00000000-intercept",
            new ScalarParameterSource<T>(() => Intercept, v => Intercept = v),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);
        RegisterParameterComponent(
            "parameters/00000001-scale",
            new ScalarParameterSource<T>(() => Scale, v => Scale = v),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);
        RegisterParameterComponent(
            "parameters/00000002-coefficients",
            new VectorFieldParameterSource<T>(
                () => Coefficients,
                value => Coefficients = value),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);
    }
    /// <summary>
    /// Gets the regression coefficients (β).
    /// </summary>
    public Vector<T>? Coefficients { get; private set; }

    /// <summary>
    /// Gets the intercept term (β₀ = μ).
    /// </summary>
    public T Intercept { get; private set; }

    /// <summary>
    /// Gets the scale parameter (σ).
    /// </summary>
    public T Scale { get; private set; }

    /// <summary>
    /// Gets the maximum iterations for optimization.
    /// </summary>
    public int MaxIterations { get; }

    /// <summary>
    /// Gets the tolerance for convergence.
    /// </summary>
    public double Tolerance { get; }

    /// <summary>
    /// Creates a new Log-Normal AFT model.
    /// </summary>
    /// <param name="maxIterations">Maximum iterations for optimization (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-6).</param>
    public LogNormalAFT(int maxIterations = 100, double tolerance = 1e-6) : base()
    {
        MaxIterations = maxIterations;
        Tolerance = tolerance;
        Intercept = NumOps.Zero;
        Scale = NumOps.One;
    }

    /// <summary>
    /// Fits the Log-Normal AFT model using maximum likelihood estimation.
    /// </summary>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        int n = x.Rows;
        int p = x.Columns;
        var engine = AiDotNetEngine.Current;

        // Initialize parameters
        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;
        Scale = NumOps.One;

        // Convert times to log scale
        var logTimes = engine.TensorLog(engine.TensorClampMin(times.AsTensor(), NumOps.FromDouble(1e-10)));

        // Initialize intercept as mean of log times
        T meanLogTime = engine.TensorMean(logTimes);
        var centered = engine.TensorAddScalar(logTimes, NumOps.Negate(meanLogTime));
        T varLogTime = NumOps.Divide(engine.TensorSum(engine.TensorMultiply(centered, centered)),
            NumOps.FromDouble(Math.Max(1, n - 1)));
        Intercept = meanLogTime;
        Scale = NumOps.Sqrt(MathHelper.Max(NumOps.FromDouble(0.01), varLogTime));

        var design = Tensor<T>.FromMatrix(x);
        var eventMask = engine.TensorEquals(events.AsTensor().Cast<T>(), NumOps.One);

        // Gradient descent optimization
        T prevLogLik = NumOps.Zero;
        bool hasPrevious = false;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // Compute log-likelihood and gradients
            var (logLik, gradIntercept, gradScale, gradBeta) = ComputeLogLikelihoodAndGradient(design, logTimes, eventMask);

            // Check convergence
            if (hasPrevious && HasConverged(logLik, prevLogLik))
                break;
            prevLogLik = logLik;
            hasPrevious = true;

            // Update parameters
            ApplyGradientStep(gradIntercept, gradScale, gradBeta, n);
        }

        // Store event times for prediction
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Compute baseline survival
        if (TrainedEventTimes.Length > 0)
        {
            var z = StandardizedLogTimes(engine.TensorLog(TrainedEventTimes.AsTensor()),
                Tensor<T>.CreateDefault(new[] { TrainedEventTimes.Length }, Intercept));
            BaselineSurvivalFunction = NormalUpperTail(z).Survival.ToVector();
        }
    }

    /// <summary>
    /// Standard normal PDF.
    /// </summary>
    /// <summary>
    /// True when the log-likelihood moved by less than the tolerance.
    /// </summary>
    private bool HasConverged(T logLik, T prevLogLik)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(logLik, prevLogLik)), NumOps.FromDouble(Tolerance));
    }

    /// <summary>
    /// Takes one gradient-ascent step of size 0.1 / n, keeping the scale at or above 0.01.
    /// </summary>
    private void ApplyGradientStep(T gradIntercept, T gradScale, Vector<T> gradBeta, int n)
    {
        T learningRate = NumOps.FromDouble(0.1);
        T count = NumOps.FromDouble(n);
        Intercept = NumOps.Add(Intercept, NumOps.Divide(NumOps.Multiply(learningRate, gradIntercept), count));
        Scale = MathHelper.Max(NumOps.FromDouble(0.01),
            NumOps.Add(Scale, NumOps.Divide(NumOps.Multiply(learningRate, gradScale), count)));

        var engine = AiDotNetEngine.Current;
        var coefficients = Coefficients ?? new Vector<T>(gradBeta.Length);
        Coefficients = engine.Add(coefficients, engine.Divide(engine.Multiply(gradBeta, learningRate), count));
    }

    /// <summary>
    /// Computes z = (log t - mu) / sigma.
    /// </summary>
    private Tensor<T> StandardizedLogTimes(Tensor<T> logTimes, Tensor<T> mu)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(engine.TensorSubtract(logTimes, mu), Scale);
    }

    /// <summary>
    /// Computes the log-normal log-likelihood and its gradients with respect to the intercept, the scale
    /// and the coefficients.
    /// </summary>
    private (T LogLik, T GradIntercept, T GradScale, Vector<T> GradBeta) ComputeLogLikelihoodAndGradient(
        Tensor<T> design, Tensor<T> logTimes, Tensor<T> eventMask)
    {
        var engine = AiDotNetEngine.Current;
        int n = design.Shape[0];
        var coefficients = Coefficients ?? new Vector<T>(design.Shape[1]);

        // Standardized residual
        var z = StandardizedLogTimes(logTimes, LinearPredictor(design, coefficients, Intercept));
        var zSquared = engine.TensorMultiply(z, z);
        var (_, logSurvival, millsRatio) = NormalUpperTail(z);

        // Event: log density of log T, log f(t) = log phi(z) - log(sigma) - log(t)
        var logDensity = engine.TensorSubtract(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(zSquared, NumOps.FromDouble(-0.5)),
                NumOps.Subtract(NumOps.FromDouble(-0.5 * Math.Log(2 * Math.PI)), NumOps.Log(Scale))),
            logTimes);

        // Censored: log survival, log S(t) = log(1 - Phi(z))
        T logLik = engine.TensorSum(engine.TensorWhere(eventMask, logDensity, logSurvival));

        // Gradients: events contribute z and z^2 - 1, censored subjects the inverse Mills ratio
        // phi/(1-Phi) and z times it (dz/dmu = -1/sigma, d log(1-Phi)/dz = -phi/(1-Phi)).
        var muTerms = engine.TensorWhere(eventMask, z, millsRatio);
        var scaleTerms = engine.TensorWhere(eventMask,
            engine.TensorAddScalar(zSquared, NumOps.Negate(NumOps.One)),
            engine.TensorMultiply(z, millsRatio));

        T gradIntercept = NumOps.Divide(engine.TensorSum(muTerms), Scale);
        T gradScale = NumOps.Divide(engine.TensorSum(scaleTerms), Scale);
        var gradBeta = engine.TensorDivideScalar(
            engine.TensorMatMul(muTerms.Reshape(new[] { 1, n }), design), Scale).ToVector();

        return (logLik, gradIntercept, gradScale, gradBeta);
    }

    /// <summary>
    /// Evaluates the standard normal upper tail 1 - Phi(z), its log, and the inverse Mills ratio
    /// phi(z) / (1 - Phi(z)).
    /// </summary>
    /// <remarks>
    /// Uses the Abramowitz and Stegun 7.1.26 form erfc(x) = P(t) exp(-x^2) directly for z &gt;= 0 rather
    /// than 1 - Phi(z): the subtraction cancels to 0 once z passes about 8 in double, after which the
    /// clamped survival made the inverse Mills ratio (and every gradient) explode. On the upper side
    /// the log and the ratio are formed analytically, so neither underflows.
    /// </remarks>
    private static (Tensor<T> Survival, Tensor<T> LogSurvival, Tensor<T> MillsRatio) NormalUpperTail(Tensor<T> z)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int[] shape = z.Shape.ToArray();

        // x = |z| / sqrt(2), t = 1 / (1 + p x), P(t) = t (a1 + t (a2 + t (a3 + t (a4 + t a5))))
        var x = engine.TensorMultiplyScalar(engine.TensorAbs(z), numOps.FromDouble(1.0 / Math.Sqrt(2)));
        var t = engine.TensorReciprocal(engine.TensorAddScalar(
            engine.TensorMultiplyScalar(x, numOps.FromDouble(0.3275911)), numOps.One));
        var poly = engine.TensorMultiplyScalar(t, numOps.FromDouble(1.061405429));
        poly = engine.TensorMultiply(t, engine.TensorAddScalar(poly, numOps.FromDouble(-1.453152027)));
        poly = engine.TensorMultiply(t, engine.TensorAddScalar(poly, numOps.FromDouble(1.421413741)));
        poly = engine.TensorMultiply(t, engine.TensorAddScalar(poly, numOps.FromDouble(-0.284496736)));
        poly = engine.TensorMultiply(t, engine.TensorAddScalar(poly, numOps.FromDouble(0.254829592)));

        // Upper tail of |z|: Q(|z|) = 0.5 P(t) exp(-z^2 / 2)
        var halfPoly = engine.TensorMultiplyScalar(poly, numOps.FromDouble(0.5));
        var negHalfZSquared = engine.TensorMultiplyScalar(engine.TensorMultiply(z, z), numOps.FromDouble(-0.5));
        var tail = engine.TensorMultiply(halfPoly, engine.TensorExp(negHalfZSquared));
        var oneMinusTail = engine.TensorAddScalar(engine.TensorNegate(tail), numOps.One);

        var upper = engine.TensorAdd(
            engine.TensorGreaterThan(z, numOps.Zero),
            engine.TensorEquals(z, numOps.Zero));
        var survival = engine.TensorWhere(upper, tail, oneMinusTail);
        var logSurvival = engine.TensorWhere(upper,
            engine.TensorAdd(engine.TensorLog(halfPoly), negHalfZSquared),
            engine.TensorLog(oneMinusTail));

        // phi(z) = exp(-z^2 / 2) / sqrt(2 pi); on the upper side phi / Q = 1 / (sqrt(2 pi) * 0.5 P(t))
        T invSqrtTwoPi = numOps.FromDouble(1.0 / Math.Sqrt(2 * Math.PI));
        var phi = engine.TensorMultiplyScalar(engine.TensorExp(negHalfZSquared), invSqrtTwoPi);
        var millsRatio = engine.TensorWhere(upper,
            engine.TensorMultiplyScalar(engine.TensorReciprocal(halfPoly), invSqrtTwoPi),
            engine.TensorDivide(phi, oneMinusTail));

        return (survival.Reshape(shape), logSurvival.Reshape(shape), millsRatio.Reshape(shape));
    }

    /// <summary>
    /// Computes exp element-wise, returning MaxValue rather than infinity once the exponent reaches
    /// log(MaxValue) of <typeparamref name="T"/>.
    /// </summary>
    private Tensor<T> ExpAsFiniteNumeric(Tensor<T> exponent)
    {
        var engine = AiDotNetEngine.Current;
        T logMax = NumOps.Log(NumOps.MaxValue);
        int[] shape = exponent.Shape.ToArray();

        return engine.TensorWhere(
            engine.TensorAdd(engine.TensorGreaterThan(exponent, logMax), engine.TensorEquals(exponent, logMax)),
            Tensor<T>.CreateDefault(shape, NumOps.MaxValue),
            engine.TensorExp(exponent));
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        x = ExtractCovariates(x);

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");
        var engine = AiDotNetEngine.Current;
        var grid = new[] { x.Rows, times.Length };

        // Linear predictor (mu_i) per subject against log(max(1e-10, t)) per time
        var mu = LinearPredictor(Tensor<T>.FromMatrix(x), coefficients, Intercept).Reshape(new[] { x.Rows, 1 });
        var logTimes = engine.TensorLog(engine.TensorClampMin(times.AsTensor(), NumOps.FromDouble(1e-10)))
            .Reshape(new[] { 1, times.Length });
        var z = StandardizedLogTimes(engine.TensorBroadcastTo(logTimes, grid), engine.TensorBroadcastTo(mu, grid));

        return NormalUpperTail(z).Survival.ToMatrix();
    }

    /// <summary>
    /// Predicts acceleration factors (hazard ratios at baseline hazard).
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        x = ExtractCovariates(x);

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");
        var engine = AiDotNetEngine.Current;

        // For log-normal, the acceleration factor is exp(-eta)
        var eta = LinearPredictor(Tensor<T>.FromMatrix(x), coefficients, NumOps.Zero);
        return ExpAsFiniteNumeric(engine.TensorNegate(eta)).ToVector();
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        var engine = AiDotNetEngine.Current;
        var logTimes = engine.TensorLog(engine.TensorClampMin(times.AsTensor(), NumOps.FromDouble(1e-10)));
        var z = StandardizedLogTimes(logTimes, Tensor<T>.CreateDefault(new[] { times.Length }, Intercept));

        return NormalUpperTail(z).Survival.ToVector();
    }

    /// <summary>
    /// Predicts median survival time.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        EnsureFitted();

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        input = ExtractCovariates(input);

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");

        // Median survival time for log-normal: exp(mu)
        return ExpAsFiniteNumeric(LinearPredictor(Tensor<T>.FromMatrix(input), coefficients, Intercept)).ToVector();
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new LogNormalAFT<T>(MaxIterations, Tolerance);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />

}
