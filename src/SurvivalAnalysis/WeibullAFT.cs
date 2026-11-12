using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

using AiDotNet.Models.Parameters;
namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements the Weibull Accelerated Failure Time (AFT) model for survival analysis.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Weibull AFT model assumes survival times follow a Weibull distribution.
/// Unlike Cox models that model hazard ratios, AFT models directly model how covariates "accelerate"
/// or "decelerate" time to event. Coefficients can be interpreted as the effect on survival time.</para>
///
/// <para><b>The Model:</b>
/// log(T) = β₀ + β₁X₁ + ... + βₚXₚ + σε
/// where T is survival time, X are covariates, β are coefficients, σ is scale, and ε ~ extreme value distribution.</para>
///
/// <para><b>Interpretation:</b>
/// <list type="bullet">
/// <item>A positive coefficient means longer survival (protective effect)</item>
/// <item>A negative coefficient means shorter survival (risk factor)</item>
/// <item>exp(β) gives the multiplicative effect on survival time</item>
/// </list>
/// </para>
///
/// <para><b>Weibull distribution:</b>
/// <list type="bullet">
/// <item>Shape parameter κ controls hazard shape (κ &lt; 1: decreasing, κ = 1: constant, κ &gt; 1: increasing)</item>
/// <item>Scale parameter λ controls time scale</item>
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
///     .ConfigureModel(new WeibullAFT&lt;double&gt;(maxIterations: 100, tolerance: 1e-6))
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
public partial class WeibullAFT<T> : SurvivalModelBase<T>
{

    /// <inheritdoc />
    /// <remarks>Intercept, scale, then the covariate coefficients -- the layout the hand-written pair used. The coefficients are registered LAST because they are the piece whose width the data decides: a fresh model has none and reports just the two scalars, and a restore sizes them from whatever follows those two.</remarks>
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
    /// Gets the intercept term (β₀).
    /// </summary>
    public T Intercept { get; private set; }

    /// <summary>
    /// Gets the scale parameter (σ).
    /// </summary>
    public T Scale { get; private set; }

    /// <summary>
    /// Gets the shape parameter (κ = 1/σ).
    /// </summary>
    public T Shape => NumOps.Divide(NumOps.One, Scale);

    /// <summary>
    /// Gets the maximum iterations for optimization.
    /// </summary>
    public int MaxIterations { get; }

    /// <summary>
    /// Gets the tolerance for convergence.
    /// </summary>
    public double Tolerance { get; }

    /// <summary>
    /// Creates a new Weibull AFT model.
    /// </summary>
    /// <param name="maxIterations">Maximum iterations for optimization (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-6).</param>
    public WeibullAFT(int maxIterations = 100, double tolerance = 1e-6) : base()
    {
        MaxIterations = maxIterations;
        Tolerance = tolerance;
        Intercept = NumOps.Zero;
        Scale = NumOps.One;
    }

    /// <summary>
    /// Fits the Weibull AFT model by maximizing the log-likelihood.
    /// </summary>
    /// <remarks>
    /// Uses gradient ascent with Armijo backtracking line search to guarantee
    /// monotonic likelihood increase. Plain gradient ascent on the Weibull AFT
    /// likelihood is unstable because the gradient contains exp(z) terms that
    /// blow up when z &gt; 0 — the original lr=0.1 step diverged the parameters to
    /// ~1e4 magnitudes on standard test data (Intercept exploded from ~2 to
    /// ~9000), making S(t) collapse to 1 for every finite t. The line search
    /// guarantees the next parameter set produces a non-decreasing log-
    /// likelihood, which is the textbook stability guarantee for AFT MLE.
    /// </remarks>
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

        // Initial intercept = mean of log times (location estimate).
        // Initial scale uses the moment-matching estimate for the Gumbel
        // distribution of z = (logT - η)/σ in the Weibull AFT family:
        // Var(z) = π²/6 ⇒ σ ≈ sqrt(6/π²) * sd(log times). Clamped away from
        // zero so the first gradient evaluation doesn't blow up.
        T mean = engine.TensorMean(logTimes);
        Intercept = mean;
        var centered = engine.TensorAddScalar(logTimes, NumOps.Negate(mean));
        T sumSq = engine.TensorSum(engine.TensorMultiply(centered, centered));
        T sdLogTimes = NumOps.Sqrt(NumOps.Divide(sumSq, NumOps.FromDouble(Math.Max(1, n - 1))));
        Scale = MathHelper.Max(NumOps.FromDouble(0.1), NumOps.Multiply(sdLogTimes, NumOps.FromDouble(Math.Sqrt(6.0) / Math.PI)));

        var design = Tensor<T>.FromMatrix(x);
        var eventMask = engine.TensorEquals(events.AsTensor().Cast<T>(), NumOps.One);
        T prevLogLik = ComputeLogLikelihood(design, logTimes, eventMask, Intercept, Scale, Coefficients);

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // Compute gradients at the current parameters, averaged per sample so step sizes don't
            // scale with n.
            var (gradIntercept, gradScale, gradBeta) = ComputeGradient(design, logTimes, eventMask);

            // Armijo backtracking line search: start at step=1, halve until
            // the new log-likelihood is strictly greater than the current one
            // (or step underflows). Guarantees monotonic improvement on a
            // concave objective.
            if (GradientIsNegligible(gradIntercept, gradScale, gradBeta)) break;

            double step = 1.0;
            var trial = TryStep(design, logTimes, eventMask, step, gradIntercept, gradScale, gradBeta, prevLogLik);
            for (int backtrack = 1; backtrack < 30 && !trial.Improves; backtrack++)
            {
                step *= 0.5;
                trial = TryStep(design, logTimes, eventMask, step, gradIntercept, gradScale, gradBeta, prevLogLik);
            }

            if (!trial.Improves) break;

            Intercept = trial.Intercept;
            Scale = trial.Scale;
            Coefficients = trial.Coefficients;

            bool converged = HasConverged(trial.LogLik, prevLogLik);
            prevLogLik = trial.LogLik;
            if (converged)
            {
                break;
            }
        }

        // Store event times for prediction
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Compute baseline survival
        if (TrainedEventTimes.Length > 0)
        {
            BaselineSurvivalFunction = WeibullSurvival(TrainedEventTimes.AsTensor(), NumOps.Exp(Intercept)).ToVector();
        }
    }

    private T ComputeLogLikelihood(Tensor<T> design, Tensor<T> logTimes, Tensor<T> eventMask, T intercept, T scale, Vector<T> coeffs)
    {
        var engine = AiDotNetEngine.Current;
        var z = ClippedResidual(design, logTimes, intercept, scale, coeffs);
        var expZ = engine.TensorExp(z);

        // log density (event): z - log(sigma) - exp(z); log survival (censored): -exp(z)
        var eventTerms = engine.TensorAddScalar(engine.TensorSubtract(z, expZ), NumOps.Negate(NumOps.Log(scale)));
        return engine.TensorSum(engine.TensorWhere(eventMask, eventTerms, engine.TensorNegate(expZ)));
    }

    /// <summary>
    /// Computes z = (log t - eta) / sigma clipped to [-30, 30] so exp(z) stays in [1e-13, 1e13] and the
    /// gradient terms don't overflow on extreme outliers.
    /// </summary>
    private Tensor<T> ClippedResidual(Tensor<T> design, Tensor<T> logTimes, T intercept, T scale, Vector<T> coeffs)
    {
        var engine = AiDotNetEngine.Current;
        var eta = LinearPredictor(design, coeffs, intercept);
        return engine.TensorClamp(engine.TensorDivideScalar(engine.TensorSubtract(logTimes, eta), scale),
            NumOps.FromDouble(-30.0), NumOps.FromDouble(30.0));
    }

    /// <summary>
    /// Computes the per-sample average gradient of the log-likelihood with respect to the intercept, the
    /// scale and the coefficients.
    /// </summary>
    private (T GradIntercept, T GradScale, Vector<T> GradBeta) ComputeGradient(
        Tensor<T> design, Tensor<T> logTimes, Tensor<T> eventMask)
    {
        var engine = AiDotNetEngine.Current;
        int n = design.Shape[0];
        var coefficients = Coefficients ?? new Vector<T>(design.Shape[1]);
        var z = ClippedResidual(design, logTimes, Intercept, Scale, coefficients);
        var expZ = engine.TensorExp(z);

        // Event: d/d(eta) = -(1 - exp z)/sigma, d/d(sigma) = -(z (1 - exp z) + 1)/sigma.
        // Censored: d/d(eta) = exp z/sigma, d/d(sigma) = z exp z/sigma.
        var zExpZ = engine.TensorMultiply(z, expZ);
        var muTerms = engine.TensorWhere(eventMask, engine.TensorAddScalar(expZ, NumOps.Negate(NumOps.One)), expZ);
        var scaleTerms = engine.TensorWhere(eventMask,
            engine.TensorSubtract(zExpZ, engine.TensorAddScalar(z, NumOps.One)),
            zExpZ);

        // Average per-sample so step sizes don't scale with n.
        T denominator = NumOps.Multiply(Scale, NumOps.FromDouble(n));
        T gradIntercept = NumOps.Divide(engine.TensorSum(muTerms), denominator);
        T gradScale = NumOps.Divide(engine.TensorSum(scaleTerms), denominator);
        var gradBeta = engine.TensorDivideScalar(
            engine.TensorMatMul(muTerms.Reshape(new[] { 1, n }), design), denominator).ToVector();

        return (gradIntercept, gradScale, gradBeta);
    }

    /// <summary>
    /// True when the squared gradient norm is below Tolerance squared.
    /// </summary>
    private bool GradientIsNegligible(T gradIntercept, T gradScale, Vector<T> gradBeta)
    {
        T gradNormSq = NumOps.Add(
            NumOps.Add(NumOps.Multiply(gradIntercept, gradIntercept), NumOps.Multiply(gradScale, gradScale)),
            AiDotNetEngine.Current.DotProduct(gradBeta, gradBeta));
        return NumOps.LessThan(gradNormSq, NumOps.FromDouble(Tolerance * Tolerance));
    }

    /// <summary>
    /// True when the log-likelihood moved by less than the tolerance.
    /// </summary>
    private bool HasConverged(T logLik, T prevLogLik)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(logLik, prevLogLik)), NumOps.FromDouble(Tolerance));
    }

    /// <summary>
    /// Evaluates the parameters one gradient step of the given size away, and whether they improve the
    /// log-likelihood by more than 1e-12 relative.
    /// </summary>
    private (T Intercept, T Scale, Vector<T> Coefficients, T LogLik, bool Improves) TryStep(
        Tensor<T> design, Tensor<T> logTimes, Tensor<T> eventMask, double step,
        T gradIntercept, T gradScale, Vector<T> gradBeta, T prevLogLik)
    {
        var engine = AiDotNetEngine.Current;
        T stepT = NumOps.FromDouble(step);
        var coefficients = Coefficients ?? new Vector<T>(gradBeta.Length);

        T newIntercept = NumOps.Add(Intercept, NumOps.Multiply(stepT, gradIntercept));
        T newScale = MathHelper.Max(NumOps.FromDouble(0.01), NumOps.Add(Scale, NumOps.Multiply(stepT, gradScale)));
        var newBeta = engine.Add(coefficients, engine.Multiply(gradBeta, stepT));
        T newLogLik = ComputeLogLikelihood(design, logTimes, eventMask, newIntercept, newScale, newBeta);

        T threshold = NumOps.Add(prevLogLik, NumOps.Multiply(NumOps.FromDouble(1e-12), NumOps.Abs(prevLogLik)));
        return (newIntercept, newScale, newBeta, newLogLik, NumOps.GreaterThan(newLogLik, threshold));
    }

    /// <summary>
    /// Weibull survival S(t) = exp(-(t/lambda)^kappa) with kappa = 1/Scale, for one lambda.
    /// </summary>
    private Tensor<T> WeibullSurvival(Tensor<T> times, T lambda)
    {
        var engine = AiDotNetEngine.Current;
        T shape = NumOps.Divide(NumOps.One, Scale);
        return engine.TensorExp(engine.TensorNegate(engine.TensorPower(engine.TensorDivideScalar(times, lambda), shape)));
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
        T shape = NumOps.Divide(NumOps.One, Scale);

        // Compute linear predictor and the subject's scale exp(η)
        var subjectScale = engine.TensorExp(LinearPredictor(Tensor<T>.FromMatrix(x), coefficients, Intercept))
            .Reshape(new[] { x.Rows, 1 });

        // Weibull survival: S(t|X) = exp(-(t/exp(η))^κ)
        var ratio = engine.TensorDivide(
            engine.TensorBroadcastTo(times.AsTensor().Reshape(new[] { 1, times.Length }), grid),
            engine.TensorBroadcastTo(subjectScale, grid));
        return engine.TensorExp(engine.TensorNegate(engine.TensorPower(ratio, shape))).ToMatrix();
    }

    /// <summary>
    /// Predicts hazard ratios (acceleration factors) relative to baseline.
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        x = ExtractCovariates(x);

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");
        var engine = AiDotNetEngine.Current;
        T shape = NumOps.Divide(NumOps.One, Scale);

        // AFT acceleration factor: exp(-κ * Σβⱼxⱼ)
        // This is equivalent to hazard ratio for Weibull
        var eta = LinearPredictor(Tensor<T>.FromMatrix(x), coefficients, NumOps.Zero);
        return engine.TensorExp(engine.TensorMultiplyScalar(eta, NumOps.Negate(shape))).ToVector();
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        return WeibullSurvival(times.AsTensor(), NumOps.Exp(Intercept)).ToVector();
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
        var engine = AiDotNetEngine.Current;
        T shape = NumOps.Divide(NumOps.One, Scale);

        // Median survival time: t_median = scale * (ln(2))^(1/shape)
        T medianFactor = NumOps.Power(NumOps.FromDouble(Math.Log(2)), NumOps.Divide(NumOps.One, shape));
        var scale = engine.TensorExp(LinearPredictor(Tensor<T>.FromMatrix(input), coefficients, Intercept));
        return engine.TensorMultiplyScalar(scale, medianFactor).ToVector();
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new WeibullAFT<T>(MaxIterations, Tolerance);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />

}
