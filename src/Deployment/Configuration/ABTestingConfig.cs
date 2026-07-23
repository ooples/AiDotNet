using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for A/B testing - comparing multiple model versions by splitting traffic.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A/B testing lets you try out a new model version on a small percentage
/// of users before fully deploying it. This helps you:
/// - Test new models in production safely
/// - Compare performance between versions with real users
/// - Gradually roll out changes to minimize risk
/// - Make data-driven decisions about which model is better
///
/// How it works:
/// You specify how to split traffic between versions. For example:
/// - Version 1.0: 80% of traffic (current stable version)
/// - Version 2.0: 20% of traffic (new experimental version)
///
/// Then you monitor metrics like accuracy, latency, and user satisfaction to decide
/// which version is better.
///
/// Example:
/// <code>
/// var abConfig = new ABTestingConfig
/// {
///     Enabled = true,
///     TrafficSplit = new Dictionary&lt;string, double&gt;
///     {
///         { "1.0.0", 0.9 },
///         { "2.0.0", 0.1 }
///     },
///     ControlVersion = "1.0.0",
///     AssignmentStrategy = AssignmentStrategy.Sticky
/// };
/// </code>
/// </para>
/// </remarks>
public class ABTestingConfig
{
    /// <summary>
    /// Gets or sets whether A/B testing is enabled (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true to enable traffic splitting between model versions.
    /// False means all traffic goes to the default version.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the default traffic split percentage for new test versions (default: 0.5).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When adding a new version to test, this is the default
    /// percentage of traffic it will receive. 0.5 means 50% of traffic goes to new version.
    /// Adjust based on your risk tolerance for new deployments.
    /// </para>
    /// </remarks>
    public double DefaultTrafficSplit { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the traffic split configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dictionary mapping version name to traffic percentage (0.0 to 1.0).
    /// Example: { "1.0": 0.8, "2.0": 0.2 } means 80% on v1.0, 20% on v2.0.
    /// Percentages must sum to 1.0.
    /// </para>
    /// </remarks>
    public Dictionary<string, double> TrafficSplit { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of defined A/B tests.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Contains the list of configured A/B tests.
    /// Each test defines which model versions to compare and their traffic allocation.
    /// </para>
    /// </remarks>
    public List<ABTest> Tests { get; set; } = new();

    /// <summary>
    /// Gets or sets the strategy for assigning users to versions (default: Random).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to assign requests to versions:
    /// - Random: Each request randomly assigned based on traffic split
    /// - Sticky: Users consistently get the same version (based on user ID hash)
    /// - Gradual: Gradually shift traffic from old to new version over time
    /// </para>
    /// </remarks>
    public AssignmentStrategy AssignmentStrategy { get; set; } = AssignmentStrategy.Random;

    /// <summary>
    /// Gets or sets the duration in days for the A/B test (default: 7).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How long to run the test before analyzing results.
    /// 7 days is typical for gathering meaningful data. After this, choose a winner.
    /// </para>
    /// </remarks>
    public int TestDurationDays { get; set; } = 7;

    /// <summary>
    /// Gets or sets whether to track experiment assignment for each request (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records which version was used for each request.
    /// Useful for analysis but adds slight overhead. Recommended for A/B testing.
    /// </para>
    /// </remarks>
    public bool TrackAssignments { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum sample size per version before comparing results (default: 1000).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Need at least this many samples before results are statistically significant.
    /// 1000 is a good minimum. Don't make decisions with fewer samples.
    /// </para>
    /// </remarks>
    public int MinSampleSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the control group version (baseline for comparison).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The current production version to compare against.
    /// Typically your stable version. New versions are compared to this baseline.
    /// </para>
    /// </remarks>
    public string? ControlVersion { get; set; }
}
