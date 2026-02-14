using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Represents the result of a statistical test.
/// </summary>
/// <typeparam name="T">The numeric type for results.</typeparam>
public class StatisticalTestResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Name of the test performed.
    /// </summary>
    public string TestName { get; init; } = "";

    /// <summary>
    /// The test statistic value.
    /// </summary>
    public T Statistic { get; init; } = default!;

    /// <summary>
    /// The p-value (probability of observing this result under null hypothesis).
    /// </summary>
    public T PValue { get; init; } = default!;

    /// <summary>
    /// Whether the result is statistically significant at the specified alpha level.
    /// </summary>
    public bool IsSignificant { get; init; }

    /// <summary>
    /// The significance level (alpha) used for the test.
    /// </summary>
    public double Alpha { get; init; } = 0.05;

    /// <summary>
    /// Degrees of freedom, if applicable.
    /// </summary>
    public double? DegreesOfFreedom { get; init; }

    /// <summary>
    /// Effect size (e.g., Cohen's d), if computed.
    /// </summary>
    public T? EffectSize { get; init; }

    /// <summary>
    /// Confidence interval for the effect, if computed.
    /// </summary>
    public (T Lower, T Upper)? ConfidenceInterval { get; init; }

    /// <summary>
    /// Description of the test and its interpretation.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Human-readable interpretation of the test result.
    /// </summary>
    public string? Interpretation { get; init; }

    /// <summary>
    /// Formats the result for display.
    /// </summary>
    public override string ToString()
    {
        var significance = IsSignificant ? "significant" : "not significant";
        return $"{TestName}: statistic={NumOps.ToDouble(Statistic):F4}, p={NumOps.ToDouble(PValue):F4} ({significance} at α={Alpha})";
    }
}

/// <summary>
/// Interface for statistical tests comparing two groups or samples.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TwoSampleTest")]
public interface ITwoSampleTest<T>
{
    /// <summary>
    /// Name of this statistical test.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of when to use this test.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Performs the statistical test comparing two samples.
    /// </summary>
    /// <param name="sample1">First sample.</param>
    /// <param name="sample2">Second sample.</param>
    /// <param name="alpha">Significance level. Default is 0.05.</param>
    /// <returns>The test result.</returns>
    StatisticalTestResult<T> Test(T[] sample1, T[] sample2, double alpha = 0.05);
}

/// <summary>
/// Interface for statistical tests comparing multiple groups.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("MultipleSampleTest")]
public interface IMultipleSampleTest<T>
{
    /// <summary>
    /// Name of this statistical test.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of when to use this test.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Performs the statistical test comparing multiple samples.
    /// </summary>
    /// <param name="samples">Array of samples to compare.</param>
    /// <param name="alpha">Significance level. Default is 0.05.</param>
    /// <returns>The test result.</returns>
    StatisticalTestResult<T> Test(T[][] samples, double alpha = 0.05);
}

/// <summary>
/// Interface for paired comparison tests (e.g., comparing same samples under different conditions).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IPairedTest<T>
{
    /// <summary>
    /// Name of this statistical test.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of when to use this test.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Performs the paired statistical test.
    /// </summary>
    /// <param name="sample1">First paired sample.</param>
    /// <param name="sample2">Second paired sample (must be same length as sample1).</param>
    /// <param name="alpha">Significance level. Default is 0.05.</param>
    /// <returns>The test result.</returns>
    StatisticalTestResult<T> Test(T[] sample1, T[] sample2, double alpha = 0.05);
}

/// <summary>
/// Interface for general statistical tests.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IStatisticalTest<T>
{
    /// <summary>
    /// Name of this statistical test.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of when to use this test.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Whether the test is for paired samples.
    /// </summary>
    bool IsPaired { get; }

    /// <summary>
    /// Whether the test is non-parametric (distribution-free).
    /// </summary>
    bool IsNonParametric { get; }
}

/// <summary>
/// Interface for tests comparing multiple classifiers or groups.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IMultipleComparisonTest<T> : IStatisticalTest<T>
{
    /// <summary>
    /// Performs the test comparing multiple groups.
    /// </summary>
    /// <param name="groups">Array of score arrays, one per group.</param>
    /// <returns>The test result.</returns>
    StatisticalTestResult<T> Test(T[][] groups);
}

/// <summary>
/// Interface for tests comparing multiple classifier predictions against ground truth.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IClassifierComparisonTest<T> : IStatisticalTest<T>
{
    /// <summary>
    /// Performs the test comparing classifier predictions.
    /// </summary>
    /// <param name="predictions">Array of prediction arrays, one per classifier.</param>
    /// <param name="actuals">True labels.</param>
    /// <returns>The test result.</returns>
    StatisticalTestResult<T> Test(T[][] predictions, T[] actuals);
}
