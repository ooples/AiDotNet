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
        var x = new Matrix<T>(rows, 3);
        var y = new Vector<T>(rows);

        for (int r = 0; r < rows; r++)
        {
            int t = first + r;
            var feat = HarRow(realizedVariance, t);
            x[r, 0] = feat[0];
            x[r, 1] = feat[1];
            x[r, 2] = feat[2];
            y[r] = realizedVariance[t + 1];
        }

        return (x, y);
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

        var row = HarRow(realizedVariance, realizedVariance.Count - 1);
        T rv = Options.UseIntercept ? Intercept : NumOps.Zero;
        for (int i = 0; i < 3; i++)
        {
            rv = NumOps.Add(rv, NumOps.Multiply(Coefficients[i], row[i]));
        }

        return NumOps.GreaterThan(rv, NumOps.Zero) ? rv : NumOps.Zero; // variance can't be negative
    }

    /// <summary>Forecasts next-period ANNUALIZED realized vol = √(variance × periodsPerYear).</summary>
    public T ForecastAnnualizedVol(IReadOnlyList<T> realizedVariance, double periodsPerYear = 252)
        => NumOps.Sqrt(NumOps.Multiply(ForecastNextVariance(realizedVariance), NumOps.FromDouble(periodsPerYear)));

    /// <summary>
    /// Convenience: fit + forecast annualized vol straight from a per-period RETURN series, using squared
    /// returns as the realized-variance proxy (RV_t = r_t²). Returns the forecast as a double for the
    /// options vol-edge signal. Falls back to the sample vol when there is too little history to fit HAR.
    /// </summary>
    public static double ForecastVolFromReturns(IReadOnlyList<double> returns, double periodsPerYear = 252)
    {
        if (returns is null || returns.Count == 0)
        {
            return 0;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var rv = new T[returns.Count];
        for (int i = 0; i < returns.Count; i++)
        {
            rv[i] = ops.FromDouble(returns[i] * returns[i]);
        }

        // Too little history to populate the monthly window → sample-variance persistence fallback.
        if (returns.Count <= MonthlyWindow + 2)
        {
            double mean = 0;
            for (int i = 0; i < returns.Count; i++)
            {
                mean += returns[i] * returns[i];
            }

            mean /= returns.Count;
            return Math.Sqrt(mean * periodsPerYear);
        }

        var model = new HarRvModel<T>(new RegressionOptions<T> { UseIntercept = true });
        model.FitRealizedVariance(rv);
        return Convert.ToDouble(model.ForecastAnnualizedVol(rv, periodsPerYear));
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var model = new HarRvModel<T>(Options, Regularization);
        if (Coefficients is not null)
        {
            model.Coefficients = Coefficients.Clone();
        }

        model.Intercept = Intercept;
        return model;
    }

    /// <summary>The HAR feature row at time <paramref name="t"/>: [daily RV, weekly avg, monthly avg].</summary>
    private T[] HarRow(IReadOnlyList<T> rv, int t) =>
    [
        rv[t],
        TrailingMean(rv, t, WeeklyWindow),
        TrailingMean(rv, t, MonthlyWindow),
    ];

    private T TrailingMean(IReadOnlyList<T> rv, int t, int window)
    {
        int lo = Math.Max(0, t - window + 1);
        T sum = NumOps.Zero;
        for (int i = lo; i <= t; i++)
        {
            sum = NumOps.Add(sum, rv[i]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(t - lo + 1));
    }
}
