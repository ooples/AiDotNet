using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Finance.Volatility;

/// <summary>
/// GARCH(1,1) — Bollerslev (1986, "Generalized Autoregressive Conditional Heteroskedasticity",
/// Journal of Econometrics 31). Conditional variance:
/// <code>σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}</code>
/// with ω &gt; 0, α ≥ 0, β ≥ 0 and α + β &lt; 1 (covariance-stationary). Fit by maximum likelihood.
/// The unconditional variance is ω / (1 − α − β). The workhorse volatility model.
/// </summary>
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generalized Autoregressive Conditional Heteroskedasticity", "https://doi.org/10.1016/0304-4076(86)90063-1", Year = 1986, Authors = "Tim Bollerslev")]
public sealed class Garch11Model<T> : ClassicalVolatilityModelBase<T>
{
    /// <inheritdoc/>
    public override string ModelName => "GARCH(1,1)";

    /// <inheritdoc/>
    public override int ParameterCount => 4; // μ, ω, α, β

    /// <inheritdoc/>
    protected override Vector<T> InitialGuess(T sampleVariance)
        => new(new[]
        {
            NumOps.Multiply(NumOps.FromDouble(0.1), Floor(sampleVariance, 1e-8)),
            NumOps.FromDouble(0.08),
            NumOps.FromDouble(0.90),
        });

    /// <inheritdoc/>
    protected override Vector<T> ToNatural(Vector<T> u)
    {
        T omega = NumOps.Exp(u[0]);                    // ω > 0
        T persistence = Sigmoid(u[1]);                 // α + β ∈ (0, 1)  → stationary
        T weight = Sigmoid(u[2]);                      // split between α and β
        T alpha = NumOps.Multiply(persistence, weight);
        T beta = NumOps.Multiply(persistence, NumOps.Subtract(NumOps.One, weight));
        return new Vector<T>(new[] { omega, alpha, beta });
    }

    /// <inheritdoc/>
    protected override Vector<T> ToUnconstrained(Vector<T> p)
    {
        T persistence = NumOps.Add(p[1], p[2]);
        T weight = NumOps.GreaterThan(persistence, NumOps.FromDouble(1e-12))
            ? NumOps.Divide(p[1], persistence)
            : NumOps.FromDouble(0.5);
        return new Vector<T>(new[] { NumOps.Log(Floor(p[0], 1e-12)), Logit(persistence), Logit(weight) });
    }

    /// <inheritdoc/>
    protected override T NextVariance(T prevVariance, T prevResidual, Vector<T> p)
        => NumOps.Add(NumOps.Add(p[0], NumOps.Multiply(p[1], NumOps.Multiply(prevResidual, prevResidual))),
            NumOps.Multiply(p[2], prevVariance));

    /// <inheritdoc/>
    /// <remarks>arch's <c>garch_recursion</c>: σ²_0 = ω + (α + β)·backcast, then
    /// σ²_t = ω + α·ε²_(t-1) + β·σ²_(t-1).</remarks>
    protected override Tensor<T> VariancePath(Tensor<T> residuals, T backcast, Vector<T> p)
    {
        var engine = AiDotNetEngine.Current;
        var drive = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(Lagged(engine.TensorSquare(residuals), backcast), p[1]), p[0]);
        return LinearRecurrence(drive, p[2], backcast);
    }

    /// <inheritdoc/>
    protected override T UnconditionalVariance(Vector<T> p)
    {
        T denom = NumOps.Subtract(NumOps.Subtract(NumOps.One, p[1]), p[2]);
        return NumOps.Divide(p[0], Floor(denom, 1e-9));
    }

    /// <inheritdoc/>
    protected override T MeanReversionSpeed(Vector<T> p) => NumOps.Add(p[1], p[2]);

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new Garch11Model<T>();

    internal static T Sigmoid(T x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Divide(ops.One, ops.Add(ops.One, ops.Exp(ops.Negate(x))));
    }

    internal static T Logit(T p)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T low = ops.FromDouble(1e-6);
        T high = ops.FromDouble(1.0 - 1e-6);
        T c = ops.LessThan(p, low) ? low : ops.GreaterThan(p, high) ? high : p;
        return ops.Log(ops.Divide(c, ops.Subtract(ops.One, c)));
    }

    /// <summary>max(x, floor).</summary>
    internal static T Floor(T x, double floor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T f = ops.FromDouble(floor);
        return ops.GreaterThan(x, f) ? x : f;
    }
}
