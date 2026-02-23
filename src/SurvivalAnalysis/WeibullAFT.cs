using System.Text;
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
    /// Fits the Weibull AFT model using Newton-Raphson optimization.
    /// </summary>
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

        // Initialize intercept as mean of log times
        Intercept = NumOps.FromDouble(logTimes.Average());

        // Newton-Raphson / gradient descent optimization
        double learningRate = 0.1;
        double prevLogLik = double.NegativeInfinity;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            double scale = NumOps.ToDouble(Scale);
            double intercept = NumOps.ToDouble(Intercept);

            // Compute log-likelihood and gradients
            double logLik = 0;
            var gradBeta = new double[p];
            double gradIntercept = 0;
            double gradScale = 0;

            for (int i = 0; i < n; i++)
            {
                // Compute linear predictor
                double eta = intercept;
                for (int j = 0; j < p; j++)
                    eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

                // Standardized residual
                double z = (logTimes[i] - eta) / scale;
                double expZ = Math.Exp(z);

                if (events[i] == 1)
                {
                    // Event: add log density
                    logLik += z - Math.Log(scale) - expZ;

                    // Gradients for events
                    double dLogLik_dZ = 1 - expZ;
                    gradIntercept -= dLogLik_dZ / scale;
                    gradScale -= (z * (1 - expZ) - 1) / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] -= dLogLik_dZ * NumOps.ToDouble(x[i, j]) / scale;
                }
                else
                {
                    // Censored: add log survival
                    logLik += -expZ;

                    // Gradients for censored
                    gradIntercept += expZ / scale;
                    gradScale += z * expZ / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] += expZ * NumOps.ToDouble(x[i, j]) / scale;
                }
            }

            // Check convergence
            if (Math.Abs(logLik - prevLogLik) < Tolerance)
                break;
            prevLogLik = logLik;

            // Update parameters
            Intercept = NumOps.FromDouble(intercept + learningRate * gradIntercept);
            Scale = NumOps.FromDouble(Math.Max(0.01, scale + learningRate * gradScale));

            for (int j = 0; j < p; j++)
            {
                Coefficients[j] = NumOps.FromDouble(
                    NumOps.ToDouble(Coefficients[j]) + learningRate * gradBeta[j]);
            }
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

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        int numSubjects = x.Rows;
        var result = new Matrix<T>(numSubjects, times.Length);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        for (int i = 0; i < numSubjects; i++)
        {
            // Compute linear predictor
            double eta = NumOps.ToDouble(Intercept);
            for (int j = 0; j < Coefficients!.Length; j++)
                eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

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

        var result = new Vector<T>(x.Rows);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        for (int i = 0; i < x.Rows; i++)
        {
            double eta = 0;
            for (int j = 0; j < Coefficients!.Length; j++)
                eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

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

        var result = new Vector<T>(input.Rows);
        double shape = 1.0 / NumOps.ToDouble(Scale);

        for (int i = 0; i < input.Rows; i++)
        {
            double eta = NumOps.ToDouble(Intercept);
            for (int j = 0; j < Coefficients!.Length; j++)
                eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(input[i, j]);

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
    public override ModelType GetModelType() => ModelType.WeibullAFT;

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
