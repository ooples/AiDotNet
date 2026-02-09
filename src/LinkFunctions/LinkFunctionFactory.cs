using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Factory for creating link function instances.
/// </summary>
/// <remarks>
/// <para>
/// Use this factory to get the appropriate link function for your GLM.
/// </para>
/// <para>
/// <b>For Beginners:</b> This factory creates link functions based on their type.
/// Choose the link function based on your data:
/// - Identity: Standard regression (any real values)
/// - Logit: Binary outcomes (yes/no, 0/1)
/// - Log: Counts or positive values
/// - Probit: Binary outcomes (alternative to logit)
/// - Inverse: Gamma-distributed responses
/// - CLogLog: Asymmetric probabilities
/// - Sqrt: Counts with variance stabilization
/// </para>
/// </remarks>
public static class LinkFunctionFactory<T>
{
    /// <summary>
    /// Creates a link function of the specified type.
    /// </summary>
    /// <param name="linkType">The type of link function to create.</param>
    /// <returns>The link function instance.</returns>
    public static ILinkFunction<T> Create(LinkFunctionType linkType)
    {
        return linkType switch
        {
            LinkFunctionType.Identity => new IdentityLink<T>(),
            LinkFunctionType.Logit => new LogitLink<T>(),
            LinkFunctionType.Log => new LogLink<T>(),
            LinkFunctionType.Probit => new ProbitLink<T>(),
            LinkFunctionType.Inverse => new ReciprocalLink<T>(),
            LinkFunctionType.CLogLog => new CLogLogLink<T>(),
            LinkFunctionType.Sqrt => new SqrtLink<T>(),
            LinkFunctionType.InverseSquared => new InverseSquaredLink<T>(),
            _ => throw new ArgumentException($"Unknown link function type: {linkType}", nameof(linkType))
        };
    }

    /// <summary>
    /// Gets the canonical link function for a distribution family.
    /// </summary>
    /// <param name="family">The distribution family.</param>
    /// <returns>The canonical link function.</returns>
    public static ILinkFunction<T> GetCanonicalLink(DistributionFamily family)
    {
        return family switch
        {
            DistributionFamily.Normal => new IdentityLink<T>(),
            DistributionFamily.Binomial => new LogitLink<T>(),
            DistributionFamily.Poisson => new LogLink<T>(),
            DistributionFamily.Gamma => new ReciprocalLink<T>(),
            DistributionFamily.InverseGaussian => new InverseSquaredLink<T>(),
            DistributionFamily.NegativeBinomial => new LogLink<T>(),
            DistributionFamily.Tweedie => throw new NotSupportedException(
                "The Tweedie canonical link is a power link g(μ)=μ^(1-p)/(1-p) that depends on the " +
                "Tweedie power parameter p. Use the appropriate link for your specific p value " +
                "(e.g., Log for p=1, Inverse for p=2)."),
            _ => throw new ArgumentOutOfRangeException(nameof(family), family,
                $"No canonical link function defined for distribution family '{family}'.")
        };
    }

    /// <summary>
    /// Gets all available link functions.
    /// </summary>
    /// <returns>Dictionary of link function names to instances.</returns>
    public static Dictionary<string, ILinkFunction<T>> GetAllLinkFunctions()
    {
        return new Dictionary<string, ILinkFunction<T>>
        {
            { "Identity", new IdentityLink<T>() },
            { "Logit", new LogitLink<T>() },
            { "Log", new LogLink<T>() },
            { "Probit", new ProbitLink<T>() },
            { "Inverse", new ReciprocalLink<T>() },
            { "CLogLog", new CLogLogLink<T>() },
            { "Sqrt", new SqrtLink<T>() },
            { "InverseSquared", new InverseSquaredLink<T>() }
        };
    }
}
