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
    public override int ParameterCount => 3; // ω, α, β

    /// <inheritdoc/>
    protected override double[] InitialGuess(double sampleVariance)
        => [0.1 * Math.Max(sampleVariance, 1e-8), 0.08, 0.90];

    /// <inheritdoc/>
    protected override double[] ToNatural(double[] u)
    {
        double omega = Math.Exp(u[0]);                 // ω > 0
        double persistence = Sigmoid(u[1]);            // α + β ∈ (0, 1)  → stationary
        double weight = Sigmoid(u[2]);                 // split between α and β
        double alpha = persistence * weight;
        double beta = persistence * (1.0 - weight);
        return [omega, alpha, beta];
    }

    /// <inheritdoc/>
    protected override double[] ToUnconstrained(double[] p)
    {
        double persistence = p[1] + p[2];
        double weight = persistence > 1e-12 ? p[1] / persistence : 0.5;
        return [Math.Log(Math.Max(p[0], 1e-12)), Logit(persistence), Logit(weight)];
    }

    /// <inheritdoc/>
    protected override double NextVariance(double prevVariance, double prevReturn, double[] p)
        => p[0] + (p[1] * prevReturn * prevReturn) + (p[2] * prevVariance);

    /// <inheritdoc/>
    protected override double UnconditionalVariance(double[] p)
    {
        double denom = 1.0 - p[1] - p[2];
        return denom > 1e-9 ? p[0] / denom : p[0] / 1e-9;
    }

    /// <inheritdoc/>
    protected override double MeanReversionSpeed(double[] p) => p[1] + p[2];

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new Garch11Model<T>();

    internal static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    internal static double Logit(double p)
    {
        double c = Math.Min(Math.Max(p, 1e-6), 1.0 - 1e-6);
        return Math.Log(c / (1.0 - c));
    }
}
