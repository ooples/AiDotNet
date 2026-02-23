namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Copula-Based Synthesis, a statistical method that models
/// the joint distribution of features by fitting marginal distributions individually
/// and coupling them with a copula function.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Copula synthesis separates two concerns:
/// - <b>Marginals</b>: Each feature's individual distribution (fitted independently)
/// - <b>Copula</b>: The dependency structure between features (fitted via rank correlations)
/// </para>
/// <para>
/// <b>For Beginners:</b> Copula synthesis is like building a recipe in two steps:
///
/// Step 1 — Learn each feature's shape separately:
///   "Age is normally distributed around 40"
///   "Income follows a log-normal distribution"
///
/// Step 2 — Learn how features relate to each other:
///   "When Age is high, Income tends to be high too"
///   "Education and Income are strongly correlated"
///
/// This separation makes the method very flexible: you can model each feature
/// with whatever distribution fits best, and the copula captures how they move together.
///
/// Example:
/// <code>
/// var options = new CopulaSynthOptions&lt;double&gt;
/// {
///     CopulaType = "gaussian",
///     NumKDEPoints = 100
/// };
/// var copulaSynth = new CopulaSynthGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class CopulaSynthOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the copula family to use for dependency modeling.
    /// </summary>
    /// <value>Copula type string, defaulting to "gaussian". Supported: "gaussian".</value>
    public string CopulaType { get; set; } = "gaussian";

    /// <summary>
    /// Gets or sets the number of points for kernel density estimation of marginals.
    /// </summary>
    /// <value>Number of KDE points, defaulting to 100.</value>
    public int NumKDEPoints { get; set; } = 100;

    /// <summary>
    /// Gets or sets the KDE bandwidth multiplier.
    /// </summary>
    /// <value>Bandwidth multiplier, defaulting to 1.0. Higher values produce smoother marginals.</value>
    public double BandwidthMultiplier { get; set; } = 1.0;
}
