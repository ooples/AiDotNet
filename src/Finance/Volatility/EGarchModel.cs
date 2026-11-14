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
    /// <para>Given the whole ln σ² path the shocks are known, and given the shocks the path is a linear
    /// recurrence, so the path is the fixed point of that map, each sweep a whole-tensor evaluation.
    /// The recursion is causal, so sweep k fixes the first k entries exactly and at most n + 1 sweeps are
    /// needed; in practice the map is a contraction and it settles in a few.</para>
    /// </remarks>
    protected override Tensor<T> VariancePath(Tensor<T> residuals, T backcast, Vector<T> p)
    {
        var engine = AiDotNetEngine.Current;
        T omega = p[0];
        T alpha = p[1];
        T gamma = p[2];
        T beta = p[3];
        T cap = NumOps.FromDouble(LogVarianceMax);
        T minusHalf = NumOps.FromDouble(-0.5);
        T centre = NumOps.Multiply(alpha, NumOps.FromDouble(ExpAbsZ));
        int n = residuals.Length;

        // Sweep 0: no shocks. drive_0 = ω always (no pre-sample shock); drive_t = ω + shock_(t-1).
        var omegaDrive = new Tensor<T>(new[] { n });
        engine.TensorFill(omegaDrive, omega);
        var logVariance = engine.TensorClampMax(LinearRecurrence(omegaDrive, beta, backcast), cap);
        for (int sweep = 0; sweep <= n; sweep++)
        {
            var z = engine.TensorMultiply(residuals, engine.TensorExp(engine.TensorMultiplyScalar(logVariance, minusHalf)));
            var shock = engine.TensorSubtractScalar(engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorAbs(z), alpha), engine.TensorMultiplyScalar(z, gamma)), centre);
            var drive = engine.TensorAddScalar(Lagged(shock, NumOps.Zero), omega);
            var next = engine.TensorClampMax(LinearRecurrence(drive, beta, backcast), cap);
            bool settled = HasSettled(logVariance, next);
            logVariance = next;
            if (settled)
            {
                break;
            }
        }

        return engine.TensorExp(logVariance);
    }

    /// <summary>True once a sweep changes no entry by more than 1e-13 relative to the path's scale.</summary>
    private bool HasSettled(Tensor<T> previous, Tensor<T> next)
    {
        var engine = AiDotNetEngine.Current;
        T change = engine.TensorMaxValue(engine.TensorAbs(engine.TensorSubtract(next, previous)));
        T scale = engine.TensorMaxValue(engine.TensorAbs(next));
        T tolerance = NumOps.Multiply(NumOps.FromDouble(1e-13), Garch11Model<T>.Floor(scale, 1.0));
        return NumOps.LessThanOrEquals(change, tolerance) || NumOps.IsNaN(change);
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
