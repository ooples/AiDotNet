using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

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
/// var weibull = new WeibullAFT&lt;double&gt;(maxIterations: 100, tolerance: 1e-6);
/// weibull.Fit(times, events, features);
/// double medianSurvival = weibull.PredictMedianSurvivalTime(newPatientFeatures);
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
public class WeibullAFT<T> : SurvivalModelBase<T>
{
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

        // Initialize parameters
        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;
        Scale = NumOps.One;

        // Convert times to log scale
        var logTimes = new double[n];
        for (int i = 0; i < n; i++)
            logTimes[i] = Math.Log(Math.Max(1e-10, NumOps.ToDouble(times[i])));

        // Initial intercept = mean of log times (location estimate).
        // Initial scale uses the moment-matching estimate for the Gumbel
        // distribution of z = (logT - η)/σ in the Weibull AFT family:
        // Var(z) = π²/6 ⇒ σ ≈ sqrt(6/π²) * sd(log times). Clamped away from
        // zero so the first gradient evaluation doesn't blow up.
        double mean = logTimes.Average();
        Intercept = NumOps.FromDouble(mean);
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) { double d = logTimes[i] - mean; sumSq += d * d; }
        double sdLogTimes = Math.Sqrt(sumSq / Math.Max(1, n - 1));
        Scale = NumOps.FromDouble(Math.Max(0.1, sdLogTimes * Math.Sqrt(6.0) / Math.PI));

        double prevLogLik = ComputeLogLikelihood(x, logTimes, events, NumOps.ToDouble(Intercept), NumOps.ToDouble(Scale), Coefficients);

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            double scale = NumOps.ToDouble(Scale);
            double intercept = NumOps.ToDouble(Intercept);

            // Compute gradients at the current parameters.
            var gradBeta = new double[p];
            double gradIntercept = 0;
            double gradScale = 0;

            for (int i = 0; i < n; i++)
            {
                double eta = intercept;
                for (int j = 0; j < p; j++)
                    eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

                // Clip z to [-30, 30] so exp(z) stays in [1e-13, 1e13] and the
                // gradient terms don't overflow on extreme outliers.
                double z = Math.Max(-30.0, Math.Min(30.0, (logTimes[i] - eta) / scale));
                double expZ = Math.Exp(z);

                if (events[i] == 1)
                {
                    double dLogLik_dZ = 1 - expZ;
                    gradIntercept -= dLogLik_dZ / scale;
                    gradScale -= (z * (1 - expZ) + 1) / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] -= dLogLik_dZ * NumOps.ToDouble(x[i, j]) / scale;
                }
                else
                {
                    gradIntercept += expZ / scale;
                    gradScale += z * expZ / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] += expZ * NumOps.ToDouble(x[i, j]) / scale;
                }
            }

            // Average per-sample so step sizes don't scale with n.
            gradIntercept /= n;
            gradScale /= n;
            for (int j = 0; j < p; j++) gradBeta[j] /= n;

            // Armijo backtracking line search: start at step=1, halve until
            // the new log-likelihood is strictly greater than the current one
            // (or step underflows). Guarantees monotonic improvement on a
            // concave objective.
            double gradNormSq = gradIntercept * gradIntercept + gradScale * gradScale;
            for (int j = 0; j < p; j++) gradNormSq += gradBeta[j] * gradBeta[j];
            if (gradNormSq < Tolerance * Tolerance) break;

            double step = 1.0;
            double newLogLik = prevLogLik;
            double newIntercept = intercept;
            double newScale = scale;
            var newBeta = new double[p];
            bool accepted = false;
            for (int backtrack = 0; backtrack < 30; backtrack++)
            {
                newIntercept = intercept + step * gradIntercept;
                newScale = Math.Max(0.01, scale + step * gradScale);
                for (int j = 0; j < p; j++)
                    newBeta[j] = NumOps.ToDouble(Coefficients[j]) + step * gradBeta[j];

                var trialCoeffs = new Vector<T>(p);
                for (int j = 0; j < p; j++) trialCoeffs[j] = NumOps.FromDouble(newBeta[j]);
                newLogLik = ComputeLogLikelihood(x, logTimes, events, newIntercept, newScale, trialCoeffs);

                if (newLogLik > prevLogLik + 1e-12 * Math.Abs(prevLogLik))
                {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }

            if (!accepted) break;

            Intercept = NumOps.FromDouble(newIntercept);
            Scale = NumOps.FromDouble(newScale);
            for (int j = 0; j < p; j++) Coefficients[j] = NumOps.FromDouble(newBeta[j]);

            if (Math.Abs(newLogLik - prevLogLik) < Tolerance)
            {
                prevLogLik = newLogLik;
                break;
            }
            prevLogLik = newLogLik;
        }

        // Store event times for prediction
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Compute baseline survival
        if (TrainedEventTimes.Length > 0)
        {
            BaselineSurvivalFunction = new Vector<T>(TrainedEventTimes.Length);
            double shape = 1.0 / NumOps.ToDouble(Scale);
            double baselineScale = Math.Exp(NumOps.ToDouble(Intercept));

            for (int t = 0; t < TrainedEventTimes.Length; t++)
            {
                double time = NumOps.ToDouble(TrainedEventTimes[t]);
                // Weibull survival: S(t) = exp(-(t/λ)^κ)
                double survival = Math.Exp(-Math.Pow(time / baselineScale, shape));
                BaselineSurvivalFunction[t] = NumOps.FromDouble(survival);
            }
        }
    }

    private double ComputeLogLikelihood(Matrix<T> x, double[] logTimes, Vector<int> events, double intercept, double scale, Vector<T> coeffs)
    {
        int n = x.Rows;
        int p = x.Columns;
        double logLik = 0.0;
        double logScale = Math.Log(scale);
        for (int i = 0; i < n; i++)
        {
            double eta = intercept;
            for (int j = 0; j < p; j++)
                eta += NumOps.ToDouble(coeffs[j]) * NumOps.ToDouble(x[i, j]);
            double z = Math.Max(-30.0, Math.Min(30.0, (logTimes[i] - eta) / scale));
            double expZ = Math.Exp(z);
            logLik += events[i] == 1
                ? z - logScale - expZ      // log density (event)
                : -expZ;                    // log survival (censored)
        }
        return logLik;
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        int numSubjects = x.Rows;
        var result = new Matrix<T>(numSubjects, times.Length);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");

        for (int i = 0; i < numSubjects; i++)
        {
            // Compute linear predictor
            double eta = NumOps.ToDouble(Intercept);
            for (int j = 0; j < coefficients.Length; j++)
                eta += NumOps.ToDouble(coefficients[j]) * NumOps.ToDouble(x[i, j]);

            double scale = Math.Exp(eta);

            for (int t = 0; t < times.Length; t++)
            {
                double time = NumOps.ToDouble(times[t]);
                // Weibull survival: S(t|X) = exp(-(t/exp(η))^κ)
                double survival = Math.Exp(-Math.Pow(time / scale, shape));
                result[i, t] = NumOps.FromDouble(survival);
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts hazard ratios (acceleration factors) relative to baseline.
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");
        var result = new Vector<T>(x.Rows);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        for (int i = 0; i < x.Rows; i++)
        {
            double eta = 0;
            for (int j = 0; j < coefficients.Length; j++)
                eta += NumOps.ToDouble(coefficients[j]) * NumOps.ToDouble(x[i, j]);

            // AFT acceleration factor: exp(-κ * Σβⱼxⱼ)
            // This is equivalent to hazard ratio for Weibull
            result[i] = NumOps.FromDouble(Math.Exp(-shape * eta));
        }

        return result;
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        var result = new Vector<T>(times.Length);
        double shape = 1.0 / NumOps.ToDouble(Scale);
        double baselineScale = Math.Exp(NumOps.ToDouble(Intercept));

        for (int t = 0; t < times.Length; t++)
        {
            double time = NumOps.ToDouble(times[t]);
            double survival = Math.Exp(-Math.Pow(time / baselineScale, shape));
            result[t] = NumOps.FromDouble(survival);
        }

        return result;
    }

    /// <summary>
    /// Predicts median survival time.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        EnsureFitted();

        var coefficients = Coefficients ?? throw new InvalidOperationException("Model has not been fitted: Coefficients is null.");
        var result = new Vector<T>(input.Rows);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        for (int i = 0; i < input.Rows; i++)
        {
            double eta = NumOps.ToDouble(Intercept);
            for (int j = 0; j < coefficients.Length; j++)
                eta += NumOps.ToDouble(coefficients[j]) * NumOps.ToDouble(input[i, j]);

            double scale = Math.Exp(eta);
            // Median survival time: t_median = scale * (ln(2))^(1/shape)
            double median = scale * Math.Pow(Math.Log(2), 1.0 / shape);
            result[i] = NumOps.FromDouble(median);
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (Coefficients is null)
            return new Vector<T>(2); // Just intercept and scale

        var parameters = new Vector<T>(Coefficients.Length + 2);
        parameters[0] = Intercept;
        parameters[1] = Scale;
        for (int i = 0; i < Coefficients.Length; i++)
            parameters[i + 2] = Coefficients[i];

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < 2) return;

        Intercept = parameters[0];
        Scale = parameters[1];

        if (parameters.Length > 2)
        {
            Coefficients = new Vector<T>(parameters.Length - 2);
            for (int i = 0; i < Coefficients.Length; i++)
                Coefficients[i] = parameters[i + 2];
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new WeibullAFT<T>(MaxIterations, Tolerance);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new WeibullAFT<T>(MaxIterations, Tolerance);
    }

    /// <inheritdoc />

    /// <inheritdoc />
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "NumFeatures", NumFeatures },
            { "IsFitted", IsFitted },
            { "Intercept", NumOps.ToDouble(Intercept) },
            { "Scale", NumOps.ToDouble(Scale) },
            { "Coefficients", Coefficients?.ToArray()?.Select(NumOps.ToDouble).ToArray() ?? Array.Empty<double>() },
            { "MaxIterations", MaxIterations },
            { "Tolerance", Tolerance }
        };

        var metadata = GetModelMetadata();
        metadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(data));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(metadata));
    }

    /// <inheritdoc />
    public override void Deserialize(byte[] modelData)
    {
        var json = Encoding.UTF8.GetString(modelData);
        var metadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(json);

        if (metadata?.ModelData is null)
            throw new InvalidOperationException("Invalid model data.");

        var dataJson = Encoding.UTF8.GetString(metadata.ModelData);
        var data = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(dataJson);

        if (data is null)
            throw new InvalidOperationException("Invalid model data.");

        NumFeatures = data["NumFeatures"]?.ToObject<int>() ?? 0;
        IsFitted = data["IsFitted"]?.ToObject<bool>() ?? false;
        Intercept = NumOps.FromDouble(data["Intercept"]?.ToObject<double>() ?? 0);
        Scale = NumOps.FromDouble(data["Scale"]?.ToObject<double>() ?? 1);

        var coeffs = data["Coefficients"]?.ToObject<double[]>() ?? Array.Empty<double>();
        Coefficients = new Vector<T>(coeffs.Length);
        for (int i = 0; i < coeffs.Length; i++)
            Coefficients[i] = NumOps.FromDouble(coeffs[i]);
    }
}
