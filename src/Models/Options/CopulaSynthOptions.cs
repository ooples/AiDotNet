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
/// var options = new CopulaSynthOptions&lt;double&gt;{ Seed = 42 };
/// var copulaSynth = new CopulaSynthGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class CopulaSynthOptions<T> : RiskModelOptions<T>
{
    // CopulaType, NumKDEPoints and BandwidthMultiplier were declared here and none was ever
    // read: CopulaSynthGenerator assigns _options and then reads nothing off it but Seed.
    //
    // CopulaType was additionally a string naming a closed set of choices, which CLAUDE.md
    // bans -- but porting it to an enum would have been worse than deleting it, because its own
    // documentation listed exactly one supported value and the generator implements exactly one
    // copula. An enum with a single member advertises a choice that does not exist.
    //
    // NumKDEPoints and BandwidthMultiplier configure a kernel density estimator the model does
    // not contain. Its marginals are the sorted observed values -- a pure empirical CDF. The
    // class summary claimed "empirical CDF / kernel density estimation" and has been corrected
    // to say what is implemented, since that sentence is what made these two look meaningful.
}
