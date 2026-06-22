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

    /// <inheritdoc/>
    public override string ModelName => "EGARCH(1,1)";
    /// <inheritdoc/>
    public override int ParameterCount => 4; // ω, α, γ, β

    /// <inheritdoc/>
    protected override double[] InitialGuess(double sampleVariance)
        // ω chosen so the implied unconditional variance ≈ sample variance: ω = (1−β)·ln(sampleVar).
        => [0.05 * Math.Log(Math.Max(sampleVariance, 1e-8)), 0.1, -0.05, 0.95];

    /// <inheritdoc/>
    protected override double[] ToNatural(double[] u)
        => [u[0], u[1], u[2], Math.Tanh(u[3])]; // β = tanh(u3) ∈ (−1, 1); ω, α, γ free in log-space

    /// <inheritdoc/>
    protected override double[] ToUnconstrained(double[] p)
    {
        double beta = Math.Min(Math.Max(p[3], -1.0 + 1e-9), 1.0 - 1e-9);
        return [p[0], p[1], p[2], Atanh(beta)];
    }

    /// <inheritdoc/>
    protected override double NextVariance(double prevVariance, double prevReturn, double[] p)
    {
        double sigma = Math.Sqrt(Math.Max(prevVariance, 1e-18));
        double z = prevReturn / sigma;
        double lnNext = p[0] + (p[3] * Math.Log(Math.Max(prevVariance, 1e-18)))
                        + (p[1] * (Math.Abs(z) - ExpAbsZ)) + (p[2] * z);
        return Math.Exp(lnNext);
    }

    /// <inheritdoc/>
    protected override double UnconditionalVariance(double[] p)
    {
        double denom = 1.0 - p[3];
        double lnVar = denom > 1e-9 ? p[0] / denom : p[0] / 1e-9;
        return Math.Exp(Math.Min(lnVar, 50.0)); // guard overflow
    }

    /// <inheritdoc/>
    protected override double MeanReversionSpeed(double[] p) => p[3];

    /// <inheritdoc/>
    protected override ClassicalVolatilityModelBase<T> CreateInstance() => new EGarchModel<T>();

    private static double Atanh(double x) => 0.5 * Math.Log((1.0 + x) / (1.0 - x));
}
