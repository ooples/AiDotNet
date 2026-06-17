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
    public override int ParameterCount => 4; // ω, α, γ, β

    /// <inheritdoc/>
    protected override double[] InitialGuess(double sampleVariance)
        => [0.1 * Math.Max(sampleVariance, 1e-8), 0.03, 0.08, 0.88];

    /// <inheritdoc/>
    protected override double[] ToNatural(double[] u)
    {
        double omega = Math.Exp(u[0]);
        double persistence = Garch11Model<T>.Sigmoid(u[1]); // α + γ/2 + β ∈ (0,1)
        // 3-way split (α, γ/2, β) via softmax with α as reference.
        double e0 = 1.0, e1 = Math.Exp(u[2]), e2 = Math.Exp(u[3]);
        double sum = e0 + e1 + e2;
        double wAlpha = e0 / sum, wGammaHalf = e1 / sum, wBeta = e2 / sum;
        double alpha = persistence * wAlpha;
        double gamma = 2.0 * persistence * wGammaHalf;
        double beta = persistence * wBeta;
        return [omega, alpha, gamma, beta];
    }

    /// <inheritdoc/>
    protected override double[] ToUnconstrained(double[] p)
    {
        double persistence = p[1] + (p[2] / 2.0) + p[3];
        double wAlpha = p[1] / persistence;
        double wGammaHalf = (p[2] / 2.0) / persistence;
        double wBeta = p[3] / persistence;
        return
        [
            Math.Log(Math.Max(p[0], 1e-12)),
            Garch11Model<T>.Logit(persistence),
            Math.Log(Math.Max(wGammaHalf, 1e-9) / Math.Max(wAlpha, 1e-9)),
            Math.Log(Math.Max(wBeta, 1e-9) / Math.Max(wAlpha, 1e-9)),
        ];
    }

    /// <inheritdoc/>
    protected override double NextVariance(double prevVariance, double prevReturn, double[] p)
    {
        double shock = prevReturn * prevReturn;
        double leverage = prevReturn < 0 ? p[2] * shock : 0.0;
        return p[0] + (p[1] * shock) + leverage + (p[3] * prevVariance);
    }

    /// <inheritdoc/>
    protected override double UnconditionalVariance(double[] p)
    {
        double denom = 1.0 - p[1] - (p[2] / 2.0) - p[3];
        return denom > 1e-9 ? p[0] / denom : p[0] / 1e-9;
    }

    /// <inheritdoc/>
    protected override double MeanReversionSpeed(double[] p) => p[1] + (p[2] / 2.0) + p[3];

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new GjrGarchModel<T>();
}
