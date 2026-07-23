namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for link functions used in Generalized Linear Models (GLMs).
/// </summary>
/// <remarks>
/// <para>
/// A link function connects the linear predictor (Xβ) to the expected value of the response
/// variable (μ). The link function g satisfies: g(μ) = η = Xβ, where η is the linear predictor.
/// </para>
/// <para>
/// <b>For Beginners:</b> In regular linear regression, predictions can be any real number.
/// But many real-world quantities have constraints:
/// - Probabilities must be between 0 and 1
/// - Counts must be non-negative
/// - Positive quantities (like income) can't be negative
///
/// Link functions solve this by transforming predictions to the appropriate range:
/// - Logit link maps linear predictions to (0,1) for probabilities
/// - Log link maps linear predictions to (0,∞) for counts/positive values
/// - Identity link makes no transformation (standard regression)
///
/// <b>Example:</b> For logistic regression:
/// - Linear predictor: η = β₀ + β₁x₁ + β₂x₂ (can be any real number)
/// - Link function (logit): η = log(p/(1-p))
/// - Inverse link: p = exp(η)/(1+exp(η)) (always between 0 and 1)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("LinkFunction")]
public interface ILinkFunction<T>
{
    /// <summary>
    /// Gets the name of the link function.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Applies the link function: g(μ) = η.
    /// </summary>
    /// <param name="mu">The mean of the response distribution.</param>
    /// <returns>The linear predictor value.</returns>
    T Link(T mu);

    /// <summary>
    /// Applies the inverse link function: g⁻¹(η) = μ.
    /// </summary>
    /// <param name="eta">The linear predictor value.</param>
    /// <returns>The mean of the response distribution.</returns>
    T InverseLink(T eta);

    /// <summary>
    /// Computes the derivative of the link function: dg/dμ.
    /// </summary>
    /// <param name="mu">The mean of the response distribution.</param>
    /// <returns>The derivative value.</returns>
    T LinkDerivative(T mu);

    /// <summary>
    /// Computes the derivative of the inverse link function: dg⁻¹/dη.
    /// </summary>
    /// <param name="eta">The linear predictor value.</param>
    /// <returns>The derivative value.</returns>
    T InverseLinkDerivative(T eta);

    /// <summary>
    /// Computes the variance function: Var(Y) as a function of μ.
    /// </summary>
    /// <param name="mu">The mean of the response distribution.</param>
    /// <returns>The variance.</returns>
    /// <remarks>
    /// For canonical links, this relates to the distribution family:
    /// - Normal: V(μ) = 1
    /// - Binomial: V(μ) = μ(1-μ)
    /// - Poisson: V(μ) = μ
    /// - Gamma: V(μ) = μ²
    /// </remarks>
    T Variance(T mu);

    /// <summary>
    /// Gets whether this is the canonical link for a distribution family.
    /// </summary>
    bool IsCanonical { get; }
}
