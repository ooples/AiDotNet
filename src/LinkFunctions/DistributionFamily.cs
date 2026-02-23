namespace AiDotNet.LinkFunctions;

/// <summary>
/// Distribution families for GLMs.
/// </summary>
public enum GlmDistributionFamily
{
    /// <summary>
    /// Normal (Gaussian) distribution. Canonical link: Identity.
    /// </summary>
    Normal,

    /// <summary>
    /// Binomial distribution. Canonical link: Logit.
    /// </summary>
    Binomial,

    /// <summary>
    /// Poisson distribution. Canonical link: Log.
    /// </summary>
    Poisson,

    /// <summary>
    /// Gamma distribution. Canonical link: Inverse.
    /// </summary>
    Gamma,

    /// <summary>
    /// Inverse Gaussian distribution. Canonical link: 1/μ².
    /// </summary>
    InverseGaussian,

    /// <summary>
    /// Negative Binomial distribution. Canonical link: Log.
    /// </summary>
    NegativeBinomial,

    /// <summary>
    /// Tweedie distribution. Canonical link: Power.
    /// </summary>
    Tweedie
}
