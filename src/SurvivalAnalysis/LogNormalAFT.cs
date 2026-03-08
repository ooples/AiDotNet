using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

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
public class LogNormalAFT<T> : SurvivalModelBase<T>
{
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

        // Initialize parameters
        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;
        Scale = NumOps.One;

        // Convert times to log scale
        var logTimes = new double[n];
        for (int i = 0; i < n; i++)
            logTimes[i] = Math.Log(Math.Max(1e-10, NumOps.ToDouble(times[i])));

        // Initialize intercept as mean of log times
        double meanLogTime = logTimes.Average();
        double varLogTime = logTimes.Sum(t => (t - meanLogTime) * (t - meanLogTime)) / Math.Max(1, n - 1);
        Intercept = NumOps.FromDouble(meanLogTime);
        Scale = NumOps.FromDouble(Math.Sqrt(Math.Max(0.01, varLogTime)));

        // Gradient descent optimization
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
                double phi = NormalPdf(z);
                double Phi = NormalCdf(z);

                if (events[i] == 1)
                {
                    // Event: add log density
                    // f(t) = (1/(t*σ)) * φ((log(t)-μ)/σ)
                    logLik += Math.Log(Math.Max(1e-300, phi)) - Math.Log(scale) - logTimes[i];

                    // Gradients
                    gradIntercept += z / scale;
                    gradScale += (z * z - 1) / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] += z * NumOps.ToDouble(x[i, j]) / scale;
                }
                else
                {
                    // Censored: add log survival
                    // S(t) = 1 - Φ((log(t)-μ)/σ)
                    double survival = 1 - Phi;
                    logLik += Math.Log(Math.Max(1e-300, survival));

                    // Gradients using inverse Mills ratio φ/(1-Φ)
                    // dlogL/dmu = +phi/((1-Phi)*sigma) because dz/dmu = -1/sigma
                    // and d(log(1-Phi))/dz = -phi/(1-Phi), so product = +phi/((1-Phi)*sigma)
                    double hazardRatio = phi / Math.Max(1e-300, survival);
                    gradIntercept += hazardRatio / scale;
                    gradScale += z * hazardRatio / scale;

                    for (int j = 0; j < p; j++)
                        gradBeta[j] += hazardRatio * NumOps.ToDouble(x[i, j]) / scale;
                }
            }

            // Check convergence
            if (Math.Abs(logLik - prevLogLik) < Tolerance)
                break;
            prevLogLik = logLik;

            // Update parameters
            Intercept = NumOps.FromDouble(intercept + learningRate * gradIntercept / n);
            Scale = NumOps.FromDouble(Math.Max(0.01, scale + learningRate * gradScale / n));

            for (int j = 0; j < p; j++)
            {
                Coefficients[j] = NumOps.FromDouble(
                    NumOps.ToDouble(Coefficients[j]) + learningRate * gradBeta[j] / n);
            }
        }

        // Store event times for prediction
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Compute baseline survival
        if (TrainedEventTimes.Length > 0)
        {
            BaselineSurvivalFunction = new Vector<T>(TrainedEventTimes.Length);
            double mu = NumOps.ToDouble(Intercept);
            double sigma = NumOps.ToDouble(Scale);

            for (int t = 0; t < TrainedEventTimes.Length; t++)
            {
                double time = NumOps.ToDouble(TrainedEventTimes[t]);
                double z = (Math.Log(time) - mu) / sigma;
                double survival = 1 - NormalCdf(z);
                BaselineSurvivalFunction[t] = NumOps.FromDouble(survival);
            }
        }
    }

    /// <summary>
    /// Standard normal PDF.
    /// </summary>
    private static double NormalPdf(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    /// <summary>
    /// Standard normal CDF using error function approximation.
    /// </summary>
    private static double NormalCdf(double x)
    {
        // Use error function approximation
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    /// <summary>
    /// Error function approximation (Abramowitz and Stegun).
    /// </summary>
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        // Constants
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        int numSubjects = x.Rows;
        var result = new Matrix<T>(numSubjects, times.Length);
        double sigma = NumOps.ToDouble(Scale);

        for (int i = 0; i < numSubjects; i++)
        {
            // Compute linear predictor (μ_i)
            double mu = NumOps.ToDouble(Intercept);
            for (int j = 0; j < Coefficients!.Length; j++)
                mu += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

            for (int t = 0; t < times.Length; t++)
            {
                double time = NumOps.ToDouble(times[t]);
                double z = (Math.Log(Math.Max(1e-10, time)) - mu) / sigma;
                double survival = 1 - NormalCdf(z);
                result[i, t] = NumOps.FromDouble(survival);
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts acceleration factors (hazard ratios at baseline hazard).
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        var result = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            double eta = 0;
            for (int j = 0; j < Coefficients!.Length; j++)
                eta += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(x[i, j]);

            // For log-normal, the acceleration factor is exp(-η)
            result[i] = NumOps.FromDouble(Math.Exp(-eta));
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
        double mu = NumOps.ToDouble(Intercept);
        double sigma = NumOps.ToDouble(Scale);

        for (int t = 0; t < times.Length; t++)
        {
            double time = NumOps.ToDouble(times[t]);
            double z = (Math.Log(Math.Max(1e-10, time)) - mu) / sigma;
            double survival = 1 - NormalCdf(z);
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

        for (int i = 0; i < input.Rows; i++)
        {
            double mu = NumOps.ToDouble(Intercept);
            for (int j = 0; j < Coefficients!.Length; j++)
                mu += NumOps.ToDouble(Coefficients[j]) * NumOps.ToDouble(input[i, j]);

            // Median survival time for log-normal: exp(μ)
            double median = Math.Exp(mu);
            result[i] = NumOps.FromDouble(median);
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (Coefficients is null)
            return new Vector<T>(2);

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
        var copy = new LogNormalAFT<T>(MaxIterations, Tolerance);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new LogNormalAFT<T>(MaxIterations, Tolerance);
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.LogNormalAFT;

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
