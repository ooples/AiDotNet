using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Finance.Volatility;

/// <summary>
/// GJR-GARCH(1,1,1) — Glosten, Jagannathan &amp; Runkle (1993, "On the Relation between the Expected Value
/// and the Volatility of the Nominal Excess Return on Stocks", Journal of Finance 48). Adds a LEVERAGE term
/// so negative shocks raise volatility more than positive ones:
/// <code>σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·1[ε_{t-1} &lt; 0] + β·σ²_{t-1}</code>
/// Stationarity: α + γ/2 + β &lt; 1; unconditional variance ω / (1 − α − γ/2 − β). Fit by maximum likelihood.
/// </summary>
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks", "https://doi.org/10.1111/j.1540-6261.1993.tb05128.x", Year = 1993, Authors = "Lawrence R. Glosten, Ravi Jagannathan, David E. Runkle")]
public sealed class GjrGarchModel<T> : ClassicalVolatilityModelBase<T>
{
    /// <inheritdoc/>
    public override string ModelName => "GJR-GARCH(1,1,1)";

    /// <inheritdoc/>
    public override int ParameterCount => 5; // μ, ω, α, γ, β

    /// <inheritdoc/>
    protected override Vector<T> InitialGuess(T sampleVariance)
        => new(new[]
        {
            NumOps.Multiply(NumOps.FromDouble(0.1), Garch11Model<T>.Floor(sampleVariance, 1e-8)),
            NumOps.FromDouble(0.03),
            NumOps.FromDouble(0.08),
            NumOps.FromDouble(0.88),
        });

    /// <inheritdoc/>
    protected override Vector<T> ToNatural(Vector<T> u)
    {
        T omega = NumOps.Exp(u[0]);
        T persistence = Garch11Model<T>.Sigmoid(u[1]); // α + γ/2 + β ∈ (0,1)
        // 3-way split (α, γ/2, β) via softmax with α as reference.
        T e1 = NumOps.Exp(u[2]);
        T e2 = NumOps.Exp(u[3]);
        T sum = NumOps.Add(NumOps.One, NumOps.Add(e1, e2));
        T alpha = NumOps.Divide(persistence, sum);
        T gamma = NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(persistence, e1)), sum);
        T beta = NumOps.Divide(NumOps.Multiply(persistence, e2), sum);
        return new Vector<T>(new[] { omega, alpha, gamma, beta });
    }

    /// <inheritdoc/>
    protected override Vector<T> ToUnconstrained(Vector<T> p)
    {
        T gammaHalf = NumOps.Divide(p[2], NumOps.FromDouble(2.0));
        T persistence = NumOps.Add(NumOps.Add(p[1], gammaHalf), p[3]);
        T wAlpha = Garch11Model<T>.Floor(NumOps.Divide(p[1], persistence), 1e-9);
        T wGammaHalf = Garch11Model<T>.Floor(NumOps.Divide(gammaHalf, persistence), 1e-9);
        T wBeta = Garch11Model<T>.Floor(NumOps.Divide(p[3], persistence), 1e-9);
        return new Vector<T>(new[]
        {
            NumOps.Log(Garch11Model<T>.Floor(p[0], 1e-12)),
            Garch11Model<T>.Logit(persistence),
            NumOps.Log(NumOps.Divide(wGammaHalf, wAlpha)),
            NumOps.Log(NumOps.Divide(wBeta, wAlpha)),
        });
    }

    /// <inheritdoc/>
    protected override T NextVariance(T prevVariance, T prevResidual, Vector<T> p)
    {
        T shock = NumOps.Multiply(prevResidual, prevResidual);
        T leverage = NumOps.LessThan(prevResidual, NumOps.Zero) ? NumOps.Multiply(p[2], shock) : NumOps.Zero;
        return NumOps.Add(NumOps.Add(NumOps.Add(p[0], NumOps.Multiply(p[1], shock)), leverage),
            NumOps.Multiply(p[3], prevVariance));
    }

    /// <inheritdoc/>
    /// <remarks>arch's <c>garch_recursion</c> with o = 1: σ²_0 = ω + (α + γ/2 + β)·backcast, then
    /// σ²_t = ω + α·ε²_(t-1) + γ·ε²_(t-1)·1[ε_(t-1) &lt; 0] + β·σ²_(t-1).</remarks>
    protected override Tensor<T> VariancePath(Tensor<T> residuals, T backcast, Vector<T> p)
    {
        var engine = AiDotNetEngine.Current;
        var squared = engine.TensorSquare(residuals);
        var negative = engine.TensorLessThan(residuals, NumOps.Zero);
        var symmetric = engine.TensorMultiplyScalar(Lagged(squared, backcast), p[1]);
        var leverage = engine.TensorMultiplyScalar(
            Lagged(engine.TensorMultiply(squared, negative), NumOps.Multiply(NumOps.FromDouble(0.5), backcast)), p[2]);
        var drive = engine.TensorAddScalar(engine.TensorAdd(symmetric, leverage), p[0]);
        return LinearRecurrence(drive, p[3], backcast);
    }

    /// <inheritdoc/>
    protected override T UnconditionalVariance(Vector<T> p)
        => NumOps.Divide(p[0], Garch11Model<T>.Floor(NumOps.Subtract(NumOps.One, MeanReversionSpeed(p)), 1e-9));

    /// <inheritdoc/>
    protected override T MeanReversionSpeed(Vector<T> p)
        => NumOps.Add(NumOps.Add(p[1], NumOps.Divide(p[2], NumOps.FromDouble(2.0))), p[3]);

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new GjrGarchModel<T>();
}
