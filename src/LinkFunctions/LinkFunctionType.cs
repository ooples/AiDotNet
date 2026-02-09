namespace AiDotNet.LinkFunctions;

/// <summary>
/// Types of link functions available.
/// </summary>
public enum LinkFunctionType
{
    /// <summary>
    /// Identity link: g(μ) = μ. Use for Normal distribution.
    /// </summary>
    Identity,

    /// <summary>
    /// Logit link: g(μ) = log(μ/(1-μ)). Use for Binomial distribution.
    /// </summary>
    Logit,

    /// <summary>
    /// Log link: g(μ) = log(μ). Use for Poisson and other positive distributions.
    /// </summary>
    Log,

    /// <summary>
    /// Probit link: g(μ) = Φ⁻¹(μ). Alternative for binary outcomes.
    /// </summary>
    Probit,

    /// <summary>
    /// Inverse link: g(μ) = 1/μ. Use for Gamma distribution.
    /// </summary>
    Inverse,

    /// <summary>
    /// Complementary log-log link: g(μ) = log(-log(1-μ)). For asymmetric probabilities.
    /// </summary>
    CLogLog,

    /// <summary>
    /// Square root link: g(μ) = √μ. Alternative for count data.
    /// </summary>
    Sqrt,

    /// <summary>
    /// Inverse squared link: g(μ) = 1/μ². Canonical link for Inverse Gaussian distribution.
    /// </summary>
    InverseSquared
}
