using AiDotNet.Finance.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Finance.Volatility;

/// <summary>Options for the classical (econometric) volatility models. They have no tunable architecture
/// (the estimator is MLE on the return series), so this is intentionally minimal.</summary>
public sealed class ClassicalVolatilityModelOptions : ModelOptions
{
}

/// <summary>
/// Base for CLASSICAL (econometric) conditional-volatility models — GARCH and its variants — estimated by
/// <b>maximum likelihood</b>, exactly as in their original papers, rather than by neural-network training.
/// Mirrors how <c>RegressionBase</c> gives classical regressions the full model surface, but for the
/// volatility interface: it implements all of <see cref="IVolatilityModel{T}"/> generically (forecast,
/// realized-vol, covariance/correlation, serialization, parameters) and delegates only the model-specific
/// conditional-variance recurrence + parameter set to the concrete subclass.
/// </summary>
/// <remarks>
/// Estimation maximizes the Gaussian quasi-log-likelihood of the return series under the model's conditional
/// variance path, optimized with a derivative-free Nelder–Mead simplex over the (constrained, via transform)
/// parameters — the standard approach in econometric packages (e.g. <c>arch</c>, <c>rugarch</c>).
/// </remarks>
public abstract class ClassicalVolatilityModelBase<T> : IVolatilityModel<T>
{
    /// <summary>Numeric operations for <typeparamref name="T"/>.</summary>
    protected INumericOperations<T> NumOps { get; } = MathHelper.GetNumericOperations<T>();

    /// <summary>The fitted parameters in NATURAL (constrained) space; null until trained.</summary>
    protected double[]? Parameters { get; private set; }

    /// <summary>Long-run (unconditional) variance implied by the fitted parameters — the forecast anchor.</summary>
    protected double FittedUnconditionalVariance { get; private set; }

    // ---- Model-specific contract (each GARCH variant implements these) -------------------------------

    /// <summary>Human-readable model name (e.g. "GARCH(1,1)").</summary>
    public abstract string ModelName { get; }

    /// <summary>Number of free parameters.</summary>
    public abstract int ParameterCount { get; }

    /// <summary>A reasonable starting guess in natural space, given the sample variance.</summary>
    protected abstract double[] InitialGuess(double sampleVariance);

    /// <summary>Maps an UNCONSTRAINED optimizer vector to NATURAL parameters honoring the model's
    /// constraints (positivity, stationarity). This is how MLE is done with a box-free optimizer.</summary>
    protected abstract double[] ToNatural(double[] unconstrained);

    /// <summary>Inverse of <see cref="ToNatural"/> — natural → unconstrained (for the initial simplex).</summary>
    protected abstract double[] ToUnconstrained(double[] natural);

    /// <summary>The conditional-variance recurrence σ²_t given the previous variance and return.</summary>
    protected abstract double NextVariance(double prevVariance, double prevReturn, double[] natural);

    /// <summary>Unconditional variance implied by the parameters (used to seed σ²_0 and long-run forecasts).</summary>
    protected abstract double UnconditionalVariance(double[] natural);

    // ---- Fitting (MLE) -------------------------------------------------------------------------------

    /// <summary>Fits the model to a return series by maximum likelihood (Nelder–Mead on the Gaussian QMLE).</summary>
    public void FitReturns(IReadOnlyList<double> returns)
    {
        if (returns is null || returns.Count < 10)
        {
            throw new ArgumentException("Need at least 10 returns to fit a GARCH-family model.", nameof(returns));
        }

        var r = returns is double[] arr ? arr : returns.ToArray();
        double sampleVar = Variance(r);
        var start = ToUnconstrained(InitialGuess(sampleVar));

        double Objective(double[] u) => NegLogLikelihood(ToNatural(u), r);
        var best = NelderMead.Minimize(Objective, start, ParameterCount);

        Parameters = ToNatural(best);
        FittedUnconditionalVariance = UnconditionalVariance(Parameters);
    }

    /// <summary>Negative Gaussian log-likelihood of the return series under the conditional-variance path.</summary>
    private double NegLogLikelihood(double[] natural, double[] r)
    {
        double var0 = UnconditionalVariance(natural);
        if (!(var0 > 0) || double.IsNaN(var0) || double.IsInfinity(var0))
        {
            return 1e12; // invalid parameters → reject
        }

        double sigma2 = var0;
        double nll = 0.0;
        const double log2Pi = 1.8378770664093453;
        for (int t = 0; t < r.Length; t++)
        {
            if (t > 0)
            {
                sigma2 = NextVariance(sigma2, r[t - 1], natural);
            }

            if (!(sigma2 > 1e-18) || double.IsNaN(sigma2) || double.IsInfinity(sigma2))
            {
                return 1e12;
            }

            nll += 0.5 * (log2Pi + Math.Log(sigma2) + (r[t] * r[t]) / sigma2);
        }

        return nll;
    }

    /// <summary>One-step-ahead conditional VARIANCE forecast given the observed return history.</summary>
    public double ForecastNextVariance(IReadOnlyList<double> returns)
    {
        if (Parameters is null)
        {
            FitReturns(returns);
        }

        var p = Parameters!;
        double sigma2 = UnconditionalVariance(p);
        for (int t = 1; t < returns.Count; t++)
        {
            sigma2 = NextVariance(sigma2, returns[t - 1], p);
        }

        // step from the last observed return to t+1
        sigma2 = NextVariance(sigma2, returns[^1], p);
        return Math.Max(sigma2, 0);
    }

    /// <summary>One-step-ahead ANNUALIZED volatility forecast = √(variance × periodsPerYear).</summary>
    public double ForecastAnnualizedVol(IReadOnlyList<double> returns, double periodsPerYear = 252)
        => Math.Sqrt(ForecastNextVariance(returns) * Math.Max(1, periodsPerYear));

    // ---- IVolatilityModel<T> (Tensor surface; delegates to the double engine above) ------------------

    /// <inheritdoc/>
    public Tensor<T> ForecastVolatility(Tensor<T> historicalReturns, int horizon)
    {
        var r = ToDoubles(historicalReturns);
        double var1 = ForecastNextVariance(r);
        var p = Parameters!;
        double uncond = UnconditionalVariance(p);
        var data = new T[Math.Max(1, horizon)];
        double v = var1;
        for (int h = 0; h < data.Length; h++)
        {
            // multi-step mean-reverts toward the unconditional variance
            data[h] = NumOps.FromDouble(Math.Sqrt(Math.Max(v, 0)));
            v = uncond + (v - uncond) * MeanReversionSpeed(p);
        }

        return new Tensor<T>(new[] { data.Length }, new Vector<T>(data));
    }

    /// <summary>Per-step persistence used for multi-step mean reversion (α+β for GARCH-type). Default 0.9.</summary>
    protected virtual double MeanReversionSpeed(double[] natural) => 0.9;

    /// <inheritdoc/>
    public Tensor<T> EstimateCurrentVolatility(Tensor<T> recentReturns)
        => CalculateRealizedVolatility(recentReturns);

    /// <inheritdoc/>
    public Tensor<T> CalculateRealizedVolatility(Tensor<T> highFrequencyReturns)
    {
        var r = ToDoubles(highFrequencyReturns);
        double sumSq = 0;
        for (int i = 0; i < r.Length; i++)
        {
            sumSq += r[i] * r[i];
        }

        double rv = Math.Sqrt(sumSq / Math.Max(1, r.Length));
        return new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(rv) }));
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeCovarianceMatrix(Tensor<T> returns)
    {
        var r = ToDoubles(returns);
        double mean = 0;
        for (int i = 0; i < r.Length; i++) mean += r[i];
        mean /= Math.Max(1, r.Length);
        double v = 0;
        for (int i = 0; i < r.Length; i++) v += (r[i] - mean) * (r[i] - mean);
        v /= Math.Max(1, r.Length - 1);
        return new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.FromDouble(v) }));
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeCorrelationMatrix(Tensor<T> returns)
        => new(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }));

    // ---- IFinancialModel / IModel ---------------------------------------------------------------------

    /// <inheritdoc/>
    public bool UseNativeMode => true;
    /// <inheritdoc/>
    public bool SupportsTraining => true;
    /// <inheritdoc/>
    public int SequenceLength { get; protected set; }
    /// <inheritdoc/>
    public int PredictionHorizon => 1;
    /// <inheritdoc/>
    public int NumFeatures => 1;

    /// <inheritdoc/>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput) => FitReturns(ToDoubles(input));

    /// <inheritdoc/>
    public Tensor<T> Predict(Tensor<T> input) => ForecastVolatility(input, 1);

    /// <inheritdoc/>
    public Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null) => ForecastVolatility(input, 1);

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelType", ModelName },
            { "Estimator", "MLE (Nelder-Mead Gaussian QMLE)" },
            { "ParameterCount", ParameterCount },
            { "Trained", Parameters is not null },
        }
    };

    /// <inheritdoc/>
    public ModelOptions GetOptions() => new ClassicalVolatilityModelOptions();

    // ---- IParameterizable -----------------------------------------------------------------------------

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        var p = Parameters ?? new double[ParameterCount];
        var v = new T[p.Length];
        for (int i = 0; i < p.Length; i++) v[i] = NumOps.FromDouble(p[i]);
        return new Vector<T>(v);
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        var p = new double[parameters.Length];
        for (int i = 0; i < p.Length; i++) p[i] = Convert.ToDouble(parameters[i]);
        Parameters = p;
        FittedUnconditionalVariance = UnconditionalVariance(p);
    }

    /// <inheritdoc/>
    public Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    // ---- Serialization (the fitted parameter vector is the whole model) -------------------------------

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var p = Parameters ?? Array.Empty<double>();
        var bytes = new byte[sizeof(int) + p.Length * sizeof(double)];
        BitConverter.GetBytes(p.Length).CopyTo(bytes, 0);
        Buffer.BlockCopy(p, 0, bytes, sizeof(int), p.Length * sizeof(double));
        return bytes;
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        int n = BitConverter.ToInt32(data, 0);
        var p = new double[n];
        Buffer.BlockCopy(data, sizeof(int), p, 0, n * sizeof(double));
        Parameters = n > 0 ? p : null;
        if (Parameters is not null) FittedUnconditionalVariance = UnconditionalVariance(Parameters);
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());
    /// <inheritdoc/>
    public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));
    /// <inheritdoc/>
    public void SaveState(Stream stream) { var b = Serialize(); stream.Write(b, 0, b.Length); }
    /// <inheritdoc/>
    public void LoadState(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        Deserialize(ms.ToArray());
    }

    // ---- Feature flags (univariate model — a single return feature) -----------------------------------

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex) => featureIndex == 0;

    // ---- Cloning --------------------------------------------------------------------------------------

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        var clone = CreateInstance();
        if (Parameters is not null) clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = CreateInstance();
        clone.SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <summary>Factory for a fresh instance of the concrete model (for cloning).</summary>
    protected abstract ClassicalVolatilityModelBase<T> CreateInstance();

    // ---- Metrics / misc interface members -------------------------------------------------------------

    /// <inheritdoc/>
    public ILossFunction<T> DefaultLossFunction { get; } = new MeanSquaredErrorLoss<T>();

    /// <inheritdoc/>
    public Dictionary<string, T> GetVolatilityMetrics()
    {
        var p = Parameters ?? new double[ParameterCount];
        var m = new Dictionary<string, T>
        {
            ["ParameterCount"] = NumOps.FromDouble(ParameterCount),
            ["Persistence"] = NumOps.FromDouble(Parameters is not null ? MeanReversionSpeed(p) : 0),
            ["UnconditionalVariance"] = NumOps.FromDouble(FittedUnconditionalVariance),
        };
        for (int i = 0; i < p.Length; i++)
        {
            m[$"theta{i}"] = NumOps.FromDouble(p[i]);
        }

        return m;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> GetFinancialMetrics() => GetVolatilityMetrics();

    /// <inheritdoc/>
    public Dictionary<string, T> GetFeatureImportance() => new() { ["returns"] = NumOps.One };

    /// <inheritdoc/>
    public void Dispose() => GC.SuppressFinalize(this);

    // ---- helpers --------------------------------------------------------------------------------------

    /// <summary>Flattens a returns tensor (1-D [n] or 2-D [n,1]) to a double array.</summary>
    protected double[] ToDoubles(Tensor<T> t)
    {
        int n = t.Shape[0];
        var r = new double[n];
        var span = t.Data.Span;
        for (int i = 0; i < n; i++) r[i] = Convert.ToDouble(span[i]);
        return r;
    }

    private static double Variance(double[] r)
    {
        double mean = 0;
        for (int i = 0; i < r.Length; i++) mean += r[i];
        mean /= r.Length;
        double v = 0;
        for (int i = 0; i < r.Length; i++) v += (r[i] - mean) * (r[i] - mean);
        return v / Math.Max(1, r.Length - 1);
    }
}
