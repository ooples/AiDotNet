using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Regression;

namespace AiDotNet.Finance.Volatility;

/// <summary>
/// HAR-RV — the Heterogeneous AutoRegressive model of Realized Volatility (Corsi, 2009,
/// "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics 7(2)).
/// </summary>
/// <remarks>
/// <para>
/// HAR-RV forecasts next-period realized variance as a linear function of realized variance averaged over
/// three horizons — daily, weekly (5), and monthly (22) — capturing volatility's long-memory with a simple
/// regression:
/// <code>RV_{t+1} = c + β_d·RV_t^(d) + β_w·RV_t^(w) + β_m·RV_t^(m) + ε_{t+1}</code>
/// Corsi estimates this by <b>ordinary least squares</b>, so this model extends <see cref="RegressionBase{T}"/>
/// (AiDotNet's classical OLS base) rather than the neural model base — faithful to the paper's estimator.
/// It is the canonical baseline for the one signal the platform found rigorously predictable (volatility),
/// and produces the forecast realized vol consumed by the vol-edge options strategy.
/// </para>
/// <para><b>For Beginners:</b> Volatility clusters and has "long memory" — calm and turbulent stretches
/// persist. HAR-RV predicts tomorrow's variance from how volatile the last day, last week, and last month
/// were, fit by plain least-squares. Simple, robust, and hard to beat.</para>
/// </remarks>
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Simple Approximate Long-Memory Model of Realized Volatility", "https://doi.org/10.1093/jjfinec/nbp001", Year = 2009, Authors = "Fulvio Corsi")]
public sealed class HarRvModel<T> : RegressionBase<T>
{
    /// <summary>Weekly aggregation horizon (trading days) from Corsi (2009).</summary>
    public const int WeeklyWindow = 5;

    /// <summary>Monthly aggregation horizon (trading days) from Corsi (2009).</summary>
    public const int MonthlyWindow = 22;

    /// <summary>Creates a HAR-RV model. By default it includes the intercept term <c>c</c> from the paper.</summary>
    public HarRvModel(RegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Fits the HAR coefficients by OLS (normal equations) — the estimator in Corsi (2009). Columns of
    /// <paramref name="x"/> are the daily/weekly/monthly RV components; <paramref name="y"/> is next-period RV.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (Options.UseIntercept)
        {
            x = x.AddConstantColumn(NumOps.One);
        }

        var xTx = x.Transpose().Multiply(x);
        var regularizedXTx = xTx.Add(Regularization.Regularize(xTx));
        var xTy = x.Transpose().Multiply(y);
        var solution = SolveSystem(regularizedXTx, xTy);

        if (Options.UseIntercept)
        {
            Intercept = solution[0];
            Coefficients = new Vector<T>([.. solution.Skip(1)]);
        }
        else
        {
            Coefficients = new Vector<T>(solution);
        }
    }

    /// <summary>
    /// Builds the HAR design from a realized-variance series: each row is
    /// [RV_t, mean(RV over last 5), mean(RV over last 22)] and the target is RV_{t+1}. The first
    /// <see cref="MonthlyWindow"/>−1 points are skipped so every monthly average is fully populated.
    /// </summary>
    public (Matrix<T> X, Vector<T> Y) BuildHarDesign(IReadOnlyList<T> realizedVariance)
    {
        if (realizedVariance is null)
        {
            throw new ArgumentNullException(nameof(realizedVariance));
        }

        int first = MonthlyWindow - 1;
        int rows = Math.Max(0, realizedVariance.Count - 1 - first);
        if (rows == 0)
        {
            return (new Matrix<T>(0, 3), new Vector<T>(0));
        }

        // window[r, k] = RV_(t - 21 + k) for t = first + r: the trailing month ending at t.
        var engine = AiDotNetEngine.Current;
        var rv = new Tensor<T>(realizedVariance.ToArray(), new[] { realizedVariance.Count });
        var lag = Enumerable.Range(0, rows * MonthlyWindow)
            .Select(k => (k / MonthlyWindow) + (k % MonthlyWindow))
            .ToArray();
        var window = engine.Reshape(engine.TensorTake(rv, new Tensor<int>(lag, new[] { lag.Length })),
            new[] { rows, MonthlyWindow });

        var daily = engine.TensorSlice(window, new[] { 0, MonthlyWindow - 1 }, new[] { rows, 1 });
        var weekly = engine.ReduceMean(
            engine.TensorSlice(window, new[] { 0, MonthlyWindow - WeeklyWindow }, new[] { rows, WeeklyWindow }),
            new[] { 1 }, keepDims: true);
        var monthly = engine.ReduceMean(window, new[] { 1 }, keepDims: true);
        var x = engine.TensorConcatenate(new[] { daily, weekly, monthly }, 1);
        var y = engine.TensorSlice(rv, new[] { first + 1 }, new[] { rows });

        return (x.ToMatrix(), y.ToVector());
    }

    /// <summary>Fits directly on a realized-variance series (builds the HAR design, then OLS).</summary>
    public void FitRealizedVariance(IReadOnlyList<T> realizedVariance)
    {
        var (x, y) = BuildHarDesign(realizedVariance);
        Train(x, y);
    }

    /// <summary>Forecasts next-period realized VARIANCE from the latest history. Clamped to ≥ 0.</summary>
    public T ForecastNextVariance(IReadOnlyList<T> realizedVariance)
    {
        if (realizedVariance is null || realizedVariance.Count == 0 || Coefficients is null || Coefficients.Length < 3)
        {
            return NumOps.Zero;
        }

        // [daily, weekly, monthly] at the last point; short histories average what there is.
        var engine = AiDotNetEngine.Current;
        int n = realizedVariance.Count;
        var rv = new Tensor<T>(realizedVariance.ToArray(), new[] { n });
        int week = Math.Min(WeeklyWindow, n);
        int month = Math.Min(MonthlyWindow, n);
        var row = new Tensor<T>(new[]
        {
            rv[n - 1],
            engine.TensorMean(engine.TensorSlice(rv, new[] { n - week }, new[] { week })),
            engine.TensorMean(engine.TensorSlice(rv, new[] { n - month }, new[] { month })),
        }, new[] { 3 });
        var coefficients = engine.TensorSlice(Tensor<T>.FromVector(Coefficients), new[] { 0 }, new[] { 3 });

        T rvNext = NumOps.Add(Options.UseIntercept ? Intercept : NumOps.Zero,
            engine.TensorSum(engine.TensorMultiply(coefficients, row)));
        return NumOps.GreaterThan(rvNext, NumOps.Zero) ? rvNext : NumOps.Zero; // variance can't be negative
    }

    /// <summary>Forecasts next-period ANNUALIZED realized vol = √(variance × periodsPerYear).</summary>
    public T ForecastAnnualizedVol(IReadOnlyList<T> realizedVariance, double periodsPerYear = 252)
        => NumOps.Sqrt(NumOps.Multiply(ForecastNextVariance(realizedVariance), NumOps.FromDouble(periodsPerYear)));

    /// <summary>
    /// Convenience: fit + forecast annualized vol straight from a per-period RETURN series, using squared
    /// returns as the realized-variance proxy (RV_t = r_t²). Falls back to the sample vol when there is too
    /// little history to fit HAR.
    /// </summary>
    public static T ForecastVolFromReturns(IReadOnlyList<T> returns, double periodsPerYear = 252)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (returns is null || returns.Count == 0)
        {
            return ops.Zero;
        }

        var engine = AiDotNetEngine.Current;
        var rv = engine.TensorSquare(new Tensor<T>(returns.ToArray(), new[] { returns.Count }));

        // Too little history to populate the monthly window → sample-variance persistence fallback.
        if (returns.Count <= MonthlyWindow + 2)
        {
            return ops.Sqrt(ops.Multiply(engine.TensorMean(rv), ops.FromDouble(periodsPerYear)));
        }

        var model = new HarRvModel<T>(new RegressionOptions<T> { UseIntercept = true });
        var series = rv.ToArray();
        model.FitRealizedVariance(series);
        return model.ForecastAnnualizedVol(series, periodsPerYear);
    }
}
