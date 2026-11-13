using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;

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
/// <para>Estimation follows the <c>arch</c> package's defaults: a constant mean (<c>ConstantMean</c>), so the
/// parameter vector is <c>[mu, volatility parameters...]</c>; the variance recursion runs on the residuals
/// <c>r - mu</c> and is seeded with arch's backcast, an exponentially weighted average of the first
/// <c>min(75, n)</c> squared residuals with weights <c>0.94^i</c>; and the Gaussian quasi-log-likelihood is
/// maximised with a derivative-free Nelder–Mead simplex over the (constrained, via transform) parameters.</para>
/// <para>All arithmetic is generic in <typeparamref name="T"/> and runs as whole-tensor engine operations.</para>
/// </remarks>
public abstract class ClassicalVolatilityModelBase<T> :
    IVolatilityModel<T>,
    IFeatureAware,
    IParameterizable<T, Tensor<T>, Tensor<T>>
{
    private const int MinimumFitReturnCount = 10;

    /// <summary>Iteration budget for the quasi-maximum-likelihood simplex search.</summary>
    private const int MaximumFitIterations = 4000;

    /// <summary>Relative convergence tolerance on the spread of log-likelihood values.</summary>
    private const double FitTolerance = 1e-10;

    /// <summary>Number of leading squared residuals in arch's variance backcast.</summary>
    private const int BackcastLength = 75;

    /// <summary>Decay of arch's backcast weights.</summary>
    private const double BackcastDecay = 0.94;

    /// <summary>Numeric operations for <typeparamref name="T"/>.</summary>
    protected INumericOperations<T> NumOps { get; } = MathHelper.GetNumericOperations<T>();

    /// <summary>The fitted parameters in NATURAL (constrained) space, <c>[mu, volatility parameters...]</c>;
    /// null until trained.</summary>
    protected Vector<T>? Parameters { get; private set; }

    /// <summary>Long-run (unconditional) variance implied by the fitted parameters — the forecast anchor.</summary>
    protected T FittedUnconditionalVariance { get; private set; }

    /// <summary>Creates the model.</summary>
    protected ClassicalVolatilityModelBase()
    {
        FittedUnconditionalVariance = NumOps.Zero;
    }

    // ---- Model-specific contract (each GARCH variant implements these) -------------------------------

    /// <summary>Human-readable model name (e.g. "GARCH(1,1)").</summary>
    public abstract string ModelName { get; }

    /// <summary>Number of free parameters, including the constant mean.</summary>
    public abstract int ParameterCount { get; }

    /// <inheritdoc/>
    long IParameterSource<T>.ParameterCount => ParameterCount;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <summary>A reasonable starting guess for the volatility parameters in natural space, given the
    /// sample variance.</summary>
    protected abstract Vector<T> InitialGuess(T sampleVariance);

    /// <summary>Maps an UNCONSTRAINED optimizer vector to NATURAL volatility parameters honoring the model's
    /// constraints (positivity, stationarity). This is how MLE is done with a box-free optimizer.</summary>
    protected abstract Vector<T> ToNatural(Vector<T> unconstrained);

    /// <summary>Inverse of <see cref="ToNatural"/> — natural → unconstrained (for the initial simplex).</summary>
    protected abstract Vector<T> ToUnconstrained(Vector<T> natural);

    /// <summary>The conditional-variance recurrence σ²_t given the previous variance and residual.</summary>
    protected abstract T NextVariance(T prevVariance, T prevResidual, Vector<T> natural);

    /// <summary>Unconditional variance implied by the volatility parameters (used for long-run forecasts).</summary>
    protected abstract T UnconditionalVariance(Vector<T> natural);

    /// <summary>
    /// The conditional variance σ²_t of every residual (<c>[n]</c>), with the pre-sample terms replaced by
    /// <paramref name="backcast"/> as in arch's recursions.
    /// </summary>
    protected abstract Tensor<T> VariancePath(Tensor<T> residuals, T backcast, Vector<T> natural);

    /// <summary>
    /// arch's backcast: <c>sum(w_i * e_i^2)</c> over the first <c>min(75, n)</c> residuals with
    /// <c>w_i ∝ 0.94^i</c> normalised to sum to one.
    /// </summary>
    protected virtual T Backcast(Tensor<T> residuals)
    {
        var engine = AiDotNetEngine.Current;
        int tau = Math.Min(BackcastLength, residuals.Length);
        var head = engine.TensorSlice(residuals, new[] { 0 }, new[] { tau });
        return engine.TensorSum(engine.TensorMultiply(engine.TensorSquare(head), BackcastWeights(tau)));
    }

    /// <summary>
    /// arch's backcast weights <c>w_i = 0.94^i / sum_j 0.94^j</c> for <c>i &lt; tau</c>. They are a constant schedule,
    /// so they are computed and normalised in double and rounded to <typeparamref name="T"/> once; forming
    /// <c>i * ln 0.94</c> in <typeparamref name="T"/> would round the decay first and let the error grow with i.
    /// </summary>
    internal static Tensor<T> BackcastWeights(int tau)
    {
        var engine = AiDotNetEngine.Current;
        var weights = engine.TensorExp(engine.TensorMultiplyScalar(
            engine.TensorArange<double>(0, tau), Math.Log(BackcastDecay)));
        return engine.TensorDivideScalar(weights, engine.TensorSum(weights)).Cast<T>();
    }

    /// <summary>
    /// The backcast arch uses for a whole fit: computed ONCE from the residuals at the mean model's starting
    /// values (the sample mean for a constant mean), not from the residuals at the candidate mean, so it stays
    /// fixed while the likelihood is optimised and when parameters are fixed.
    /// </summary>
    private T StartingBackcast(Tensor<T> returns)
    {
        var engine = AiDotNetEngine.Current;
        return Backcast(engine.TensorSubtractScalar(returns, engine.TensorMean(returns)));
    }

    /// <summary>
    /// Evaluates the first-order linear recurrence <c>s_t = drive_t + decay * s_(t-1)</c> with
    /// <c>s_(-1) = initial</c>, as blocked lower-triangular Toeplitz products.
    /// </summary>
    protected Tensor<T> LinearRecurrence(Tensor<T> drive, T decay, T initial)
        => AiDotNet.Helpers.LinearRecurrence<T>.Solve(drive, decay, initial);

    /// <summary>
    /// The series delayed by one step, <c>[presample, values_0, ..., values_(n-2)]</c>: the "previous"
    /// value each step of the recursion sees, with arch's backcast standing in before the sample.
    /// </summary>
    protected static Tensor<T> Lagged(Tensor<T> values, T presample)
    {
        var engine = AiDotNetEngine.Current;
        var head = new Tensor<T>(new[] { presample }, new[] { 1 });
        return values.Length <= 1
            ? head
            : engine.TensorConcatenate(new[] { head, engine.TensorSlice(values, new[] { 0 }, new[] { values.Length - 1 }) }, 0);
    }

    /// <summary>Per-step persistence used for multi-step mean reversion (α+β for GARCH-type). Default 0.9.</summary>
    protected virtual T MeanReversionSpeed(Vector<T> natural) => NumOps.FromDouble(0.9);

    // ---- Fitting (MLE) -------------------------------------------------------------------------------

    /// <summary>Fits the model to a return series by maximum likelihood (Nelder–Mead on the Gaussian QMLE).</summary>
    public void FitReturns(Vector<T> returns)
    {
        if (returns is null || returns.Length < MinimumFitReturnCount)
        {
            throw new ArgumentException(
                $"Need at least {MinimumFitReturnCount} returns to fit a GARCH-family model.",
                nameof(returns));
        }

        var engine = AiDotNetEngine.Current;
        var r = Tensor<T>.FromVector(returns);
        T mean = engine.TensorMean(r);
        var centered = engine.TensorSubtractScalar(r, mean);
        T sampleVar = NumOps.Divide(engine.TensorSumOfSquares(centered), NumOps.FromDouble(Math.Max(1, r.Length - 1)));

        // The mean is searched in units of the sample standard deviation so the simplex steps scale
        // with the data.
        T scale = NumOps.Sqrt(NumOps.GreaterThan(sampleVar, NumOps.Zero) ? sampleVar : NumOps.One);
        var start = new Vector<T>(new[] { NumOps.Divide(mean, scale) }.Concat(ToUnconstrained(InitialGuess(sampleVar))));

        T Objective(Vector<T> u) => NegLogLikelihood(Natural(u, scale), r);

        // Uses the library's Nelder-Mead optimizer rather than a private copy. The options below
        // reproduce the estimation settings this module has always used: a 10% relative initial
        // simplex step, an absolute step for coordinates too small to perturb proportionally, and
        // fixed (non-adaptive) reflection/expansion/contraction/shrink coefficients.
        var optimizer = new NelderMeadOptimizer<T, Tensor<T>, Tensor<T>>(
            new NelderMeadOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialSimplexStep = 0.1,
                ZeroCoordinateSimplexStep = 0.1,
                ZeroCoordinateThreshold = 1e-6,
                UseAdaptiveParameters = false,
            });

        var best = optimizer.Minimize(start, Objective, MaximumFitIterations, NumOps.FromDouble(FitTolerance));

        Parameters = Natural(best, scale);
        FittedUnconditionalVariance = UnconditionalVariance(VolatilityParameters(Parameters));
    }

    /// <summary>Negative Gaussian log-likelihood of the return series under the conditional-variance path.</summary>
    internal T NegLogLikelihood(Vector<T> natural, Tensor<T> returns)
    {
        T invalid = NumOps.FromDouble(1e12);
        var residuals = AiDotNetEngine.Current.TensorSubtractScalar(returns, natural[0]);
        var volatility = VolatilityParameters(natural);
        var sigma2 = VariancePath(residuals, StartingBackcast(returns), volatility);

        var engine = AiDotNetEngine.Current;
        T smallest = engine.TensorMinValue(sigma2);
        if (!NumOps.GreaterThan(smallest, NumOps.FromDouble(1e-18)) || NumOps.IsNaN(smallest))
        {
            return invalid; // invalid parameters → reject
        }

        // 0.5 * sum(log 2π + log σ² + e² / σ²)
        T logDet = engine.TensorSum(engine.TensorLog(sigma2));
        T quadratic = engine.TensorSum(engine.TensorDivide(engine.TensorSquare(residuals), sigma2));
        T constant = NumOps.FromDouble(1.8378770664093453 * residuals.Length);
        T nll = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(constant, NumOps.Add(logDet, quadratic)));
        return NumOps.IsNaN(nll) || NumOps.IsInfinity(nll) ? invalid : nll;
    }

    /// <summary>One-step-ahead conditional VARIANCE forecast given the observed return history.</summary>
    public T ForecastNextVariance(Vector<T> returns)
    {
        if (Parameters is null)
        {
            FitReturns(returns);
        }

        var p = Parameters ?? throw new InvalidOperationException("The model has no fitted parameters.");
        var volatility = VolatilityParameters(p);
        var residuals = AiDotNetEngine.Current.TensorSubtractScalar(Tensor<T>.FromVector(returns), p[0]);
        // arch's forecast() backcasts from the residuals at the fitted mean (its fit and fix use the
        // starting-value backcast instead; the two differ only in the first few steps of the path).
        var sigma2 = VariancePath(residuals, Backcast(residuals), volatility);

        // step from the last observed residual to t+1
        T next = NextVariance(sigma2[sigma2.Length - 1], residuals[residuals.Length - 1], volatility);
        return NumOps.GreaterThan(next, NumOps.Zero) ? next : NumOps.Zero;
    }

    /// <summary>One-step-ahead ANNUALIZED volatility forecast = √(variance × periodsPerYear).</summary>
    public T ForecastAnnualizedVol(Vector<T> returns, double periodsPerYear = 252)
        => NumOps.Sqrt(NumOps.Multiply(ForecastNextVariance(returns), NumOps.FromDouble(Math.Max(1, periodsPerYear))));

    // ---- IVolatilityModel<T> (Tensor surface) ---------------------------------------------------------

    /// <inheritdoc/>
    public Tensor<T> ForecastVolatility(Tensor<T> historicalReturns, int horizon)
    {
        var engine = AiDotNetEngine.Current;
        T var1 = ForecastNextVariance(ReturnsVector(historicalReturns));
        var p = Parameters ?? throw new InvalidOperationException("The model has no fitted parameters.");
        var volatility = VolatilityParameters(p);
        T uncond = UnconditionalVariance(volatility);
        int steps = Math.Max(1, horizon);

        // Multi-step variance mean-reverts toward the unconditional variance:
        // v_h = uncond + (v_1 - uncond) * persistence^h.
        var persistence = new Tensor<T>(new[] { steps });
        engine.TensorFill(persistence, MeanReversionSpeed(volatility));
        var decay = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { NumOps.One }, new[] { 1 }),
            engine.TensorSlice(engine.TensorCumProd(persistence, 0), new[] { 0 }, new[] { steps - 1 })
        }, 0);
        var variance = engine.TensorAddScalar(engine.TensorMultiplyScalar(decay, NumOps.Subtract(var1, uncond)), uncond);
        return engine.TensorSqrt(engine.TensorClampMin(variance, NumOps.Zero));
    }

    /// <inheritdoc/>
    public Tensor<T> EstimateCurrentVolatility(Tensor<T> recentReturns)
        => CalculateRealizedVolatility(recentReturns);

    /// <inheritdoc/>
    public Tensor<T> CalculateRealizedVolatility(Tensor<T> highFrequencyReturns)
    {
        var r = Tensor<T>.FromVector(ReturnsVector(highFrequencyReturns));
        T meanSquare = NumOps.Divide(AiDotNetEngine.Current.TensorSumOfSquares(r), NumOps.FromDouble(Math.Max(1, r.Length)));
        return new Tensor<T>(new[] { NumOps.Sqrt(meanSquare) }, new[] { 1 });
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeCovarianceMatrix(Tensor<T> returns)
    {
        var engine = AiDotNetEngine.Current;
        var r = Tensor<T>.FromVector(ReturnsVector(returns));
        T mean = r.Length == 0 ? NumOps.Zero : engine.TensorMean(r);
        T v = NumOps.Divide(engine.TensorSumOfSquares(engine.TensorSubtractScalar(r, mean)),
            NumOps.FromDouble(Math.Max(1, r.Length - 1)));
        return new Tensor<T>(new[] { v }, new[] { 1, 1 });
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
    public int SequenceLength { get; protected set; } = MinimumFitReturnCount;
    /// <inheritdoc/>
    public int PredictionHorizon => 1;
    /// <inheritdoc/>
    public int NumFeatures => 1;

    /// <inheritdoc/>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput) => FitReturns(ReturnsVector(input));

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
        return Parameters is null ? new Vector<T>(ParameterCount) : Parameters.Clone();
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"{ModelName} has {ParameterCount} parameters (mu first), but {parameters.Length} were given.",
                nameof(parameters));
        }

        Parameters = parameters.Clone();
        FittedUnconditionalVariance = UnconditionalVariance(VolatilityParameters(Parameters));
    }

    /// <inheritdoc/>
    public Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    // ---- Serialization (the fitted parameter vector is the whole model) -------------------------------

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        return SerializationHelper<T>.SerializeVector(Parameters ?? new Vector<T>(0));
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        var p = SerializationHelper<T>.DeserializeVector(data);
        Parameters = p.Length > 0 ? p : null;
        if (Parameters is not null) FittedUnconditionalVariance = UnconditionalVariance(VolatilityParameters(Parameters));
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
    public IEnumerable<int> GetActiveFeatureIndices() => new[] { 0 };
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
        var p = Parameters ?? new Vector<T>(ParameterCount);
        var m = new Dictionary<string, T>
        {
            ["ParameterCount"] = NumOps.FromDouble(ParameterCount),
            ["Persistence"] = Parameters is not null ? MeanReversionSpeed(VolatilityParameters(p)) : NumOps.Zero,
            ["UnconditionalVariance"] = FittedUnconditionalVariance,
        };
        int index = 0;
        foreach (T value in p)
        {
            m[$"theta{index++}"] = value;
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

    /// <summary>Flattens a univariate returns tensor ([n], [n,1], or [batch,n,1]) to a vector; with several
    /// features the first one is taken.</summary>
    protected Vector<T> ReturnsVector(Tensor<T> t)
    {
        int featureCount = t.Shape.Length >= 2 ? Math.Max(1, t.Shape[t.Shape.Length - 1]) : 1;
        if (featureCount == 1)
        {
            return t.ToVector();
        }

        var engine = AiDotNetEngine.Current;
        int n = t.Length / featureCount;
        var rows = engine.Reshape(t, new[] { n, featureCount });
        return engine.TensorSlice(rows, new[] { 0, 0 }, new[] { n, 1 }).ToVector();
    }

    /// <summary>The volatility parameters (everything after the constant mean).</summary>
    protected static Vector<T> VolatilityParameters(Vector<T> parameters)
    {
        return new Vector<T>(parameters.Skip(1));
    }

    /// <summary>Optimizer vector <c>[mu / scale, unconstrained...]</c> to natural <c>[mu, natural...]</c>.</summary>
    private Vector<T> Natural(Vector<T> unconstrained, T scale)
    {
        return new Vector<T>(new[] { NumOps.Multiply(unconstrained[0], scale) }
            .Concat(ToNatural(VolatilityParameters(unconstrained))));
    }
}
