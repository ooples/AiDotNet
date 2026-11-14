using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Finance.Volatility;

/// <summary>
/// EGARCH(1,1) — Nelson (1991, "Conditional Heteroskedasticity in Asset Returns: A New Approach",
/// Econometrica 59). Models LOG conditional variance, so positivity holds automatically and the leverage
/// effect enters linearly in the standardized shock z = ε/σ:
/// <code>ln σ²_t = ω + β·ln σ²_{t-1} + α·(|z_{t-1}| − E|z|) + γ·z_{t-1}</code>
/// with E|z| = √(2/π) for Gaussian innovations, and |β| &lt; 1 for stationarity. γ &lt; 0 captures the
/// leverage effect (negative returns raise vol more). Fit by maximum likelihood.
/// </summary>
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Conditional Heteroskedasticity in Asset Returns: A New Approach", "https://doi.org/10.2307/2938260", Year = 1991, Authors = "Daniel B. Nelson")]
public sealed class EGarchModel<T> : ClassicalVolatilityModelBase<T>
{
    private const double ExpAbsZ = 0.7978845608028654; // E|z| = √(2/π) for a standard normal

    /// <summary>arch's cap on ln σ² (ln of the largest double, less 0.1).</summary>
    private const double LogVarianceMax = 709.682712;

    /// <summary>ln(1e-18): the floor <see cref="NextVariance"/> puts on the previous variance.</summary>
    private const double LogVarianceMin = -41.446531673892821;

    /// <inheritdoc/>
    public override string ModelName => "EGARCH(1,1)";

    /// <inheritdoc/>
    public override int ParameterCount => 5; // μ, ω, α, γ, β

    /// <inheritdoc/>
    protected override Vector<T> InitialGuess(T sampleVariance)
        // ω chosen so the implied unconditional variance ≈ sample variance: ω = (1−β)·ln(sampleVar).
        => new(new[]
        {
            NumOps.Multiply(NumOps.FromDouble(0.05), NumOps.Log(Garch11Model<T>.Floor(sampleVariance, 1e-8))),
            NumOps.FromDouble(0.1),
            NumOps.FromDouble(-0.05),
            NumOps.FromDouble(0.95),
        });

    /// <inheritdoc/>
    protected override Vector<T> ToNatural(Vector<T> u)
    {
        // β = tanh(u3) = 2·sigmoid(2·u3) − 1 ∈ (−1, 1); ω, α, γ free
        T two = NumOps.FromDouble(2.0);
        T beta = NumOps.Subtract(NumOps.Multiply(two, Garch11Model<T>.Sigmoid(NumOps.Multiply(two, u[3]))), NumOps.One);
        return new Vector<T>(new[] { u[0], u[1], u[2], beta });
    }

    /// <inheritdoc/>
    protected override Vector<T> ToUnconstrained(Vector<T> p)
    {
        T low = NumOps.FromDouble(-1.0 + 1e-9);
        T high = NumOps.FromDouble(1.0 - 1e-9);
        T beta = NumOps.LessThan(p[3], low) ? low : NumOps.GreaterThan(p[3], high) ? high : p[3];
        T atanh = NumOps.Multiply(NumOps.FromDouble(0.5),
            NumOps.Log(NumOps.Divide(NumOps.Add(NumOps.One, beta), NumOps.Subtract(NumOps.One, beta))));
        return new Vector<T>(new[] { p[0], p[1], p[2], atanh });
    }

    /// <inheritdoc/>
    /// <remarks>arch's EGARCH backcasts ln σ²: the log of the GARCH backcast.</remarks>
    protected override T Backcast(Tensor<T> residuals) => NumOps.Log(base.Backcast(residuals));

    /// <inheritdoc/>
    protected override T NextVariance(T prevVariance, T prevResidual, Vector<T> p)
    {
        T variance = Garch11Model<T>.Floor(prevVariance, 1e-18);
        T z = NumOps.Divide(prevResidual, NumOps.Sqrt(variance));
        T lnNext = NumOps.Add(
            NumOps.Add(p[0], NumOps.Multiply(p[3], NumOps.Log(variance))),
            NumOps.Add(NumOps.Multiply(p[1], NumOps.Subtract(NumOps.Abs(z), NumOps.FromDouble(ExpAbsZ))),
                NumOps.Multiply(p[2], z)));
        return NumOps.Exp(lnNext);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>arch's <c>egarch_recursion</c>: ln σ²_0 = ω + β·backcast and, for t ≥ 1,
    /// ln σ²_t = ω + α·(|z_(t-1)| − √(2/π)) + γ·z_(t-1) + β·ln σ²_(t-1) with z = ε/σ.</para>
    /// <para>The shock of step t depends on ln σ²_(t-1) through z = ε·exp(−ln σ²/2), so unlike the GARCH and
    /// GJR paths this recursion is not linear in its state and has no whole-tensor closed form. It is evaluated
    /// as the one-pass causal recursion it is, in double, and the path is rounded to
    /// <typeparamref name="T"/> once. Solving it as a fixed point of whole-tensor sweeps costs one full
    /// linear-recurrence solve per sweep and needs up to n + 1 sweeps wherever the map is not a contraction,
    /// which the simplex search visits: a fit of 600 returns took 22 s and one of 1500 did not finish in
    /// two minutes, against milliseconds for the single pass.</para>
    /// <para>As in <see cref="NextVariance"/>, the previous variance enters each step floored at 1e-18, in both
    /// the standardized shock and the β term. The floor keeps z = ε·exp(−ln σ²/2) finite: without it a very
    /// negative ln σ² (or a −∞ backcast from an all-zero residual series) overflowed exp, and 0·∞ made the path
    /// NaN. As in arch, a ln σ² above the cap is stored capped and the capped value feeds the next step.</para>
    /// </remarks>
    protected override Tensor<T> VariancePath(Tensor<T> residuals, T backcast, Vector<T> p)
    {
        var engine = AiDotNetEngine.Current;
        int n = residuals.Length;
        double omega = NumOps.ToDouble(p[0]);
        double alpha = NumOps.ToDouble(p[1]);
        double gamma = NumOps.ToDouble(p[2]);
        double beta = NumOps.ToDouble(p[3]);
        double centre = alpha * ExpAbsZ;
        double presample = NumOps.ToDouble(backcast);

        // Contiguous() first: Cast on a narrowed view reads from the start of the backing storage.
        double[] shocks = residuals.Contiguous().Cast<double>().ToArray();
        var logVariance = new double[n];

        // ln σ²_0 = ω + β·backcast (no pre-sample shock); ln σ²_t = ω + shock_(t-1) + β·ln σ²_(t-1).
        double previous = presample > LogVarianceMin ? presample : LogVarianceMin;
        double shock = 0.0;
        for (int t = 0; t < n; t++)
        {
            double current = Math.Min(omega + shock + (beta * previous), LogVarianceMax);
            logVariance[t] = current;
            previous = Math.Max(current, LogVarianceMin);
            double z = shocks[t] * Math.Exp(-0.5 * previous);
            shock = (alpha * Math.Abs(z)) + (gamma * z) - centre;
        }

        return engine.TensorExp(new Tensor<double>(logVariance, new[] { n })).Cast<T>();
    }

    /// <inheritdoc/>
    protected override T UnconditionalVariance(Vector<T> p)
    {
        T lnVar = NumOps.Divide(p[0], Garch11Model<T>.Floor(NumOps.Subtract(NumOps.One, p[3]), 1e-9));
        T limit = NumOps.FromDouble(50.0);
        return NumOps.Exp(NumOps.LessThan(lnVar, limit) ? lnVar : limit); // guard overflow
    }

    /// <inheritdoc/>
    protected override T MeanReversionSpeed(Vector<T> p) => p[3];

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new EGarchModel<T>();
}
