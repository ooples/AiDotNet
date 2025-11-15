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
/// Example Use Case:
/// You've trained a new model that seems more accurate in testing. Instead of deploying
/// it to all users immediately, you start with 10% of traffic. If metrics look good after
/// a week, you increase to 50%, then 100%.
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
    /// Gets or sets the strategy for assigning users to versions (default: "random").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to assign requests to versions:
    /// - "random": Each request randomly assigned based on traffic split
    /// - "sticky": Users consistently get the same version (based on user ID hash)
    /// - "gradual": Gradually shift traffic from old to new version over time
    /// </para>
    /// </remarks>
    public string AssignmentStrategy { get; set; } = "random";

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

    /// <summary>
    /// Creates a simple 90/10 A/B test configuration.
    /// </summary>
    /// <param name="controlVersion">The stable control version.</param>
    /// <param name="experimentVersion">The new experimental version.</param>
    /// <returns>A 90/10 traffic split configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to test a new version on 10% of users.
    /// 90% continue using the stable version. Users get consistent experience (sticky).
    /// Good for initial rollouts.
    /// </para>
    /// </remarks>
    public static ABTestingConfig Simple9010(string controlVersion, string experimentVersion)
    {
        return new ABTestingConfig
        {
            Enabled = true,
            TrafficSplit = new Dictionary<string, double>
            {
                { controlVersion, 0.9 },
                { experimentVersion, 0.1 }
            },
            ControlVersion = controlVersion,
            AssignmentStrategy = "sticky"
        };
    }

    /// <summary>
    /// Creates a balanced 50/50 A/B test configuration.
    /// </summary>
    /// <param name="versionA">First version to test.</param>
    /// <param name="versionB">Second version to test.</param>
    /// <returns>A 50/50 traffic split configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for equal comparison between two versions.
    /// Each gets 50% of traffic. Good for comparing similar-quality models.
    /// </para>
    /// </remarks>
    public static ABTestingConfig Balanced5050(string versionA, string versionB)
    {
        return new ABTestingConfig
        {
            Enabled = true,
            TrafficSplit = new Dictionary<string, double>
            {
                { versionA, 0.5 },
                { versionB, 0.5 }
            },
            AssignmentStrategy = "sticky"
        };
    }

    /// <summary>
    /// Creates a disabled A/B testing configuration (single version).
    /// </summary>
    /// <returns>A configuration with A/B testing disabled.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to disable A/B testing.
    /// All traffic goes to the default version. Simplest deployment.
    /// </para>
    /// </remarks>
    public static ABTestingConfig Disabled()
    {
        return new ABTestingConfig
        {
            Enabled = false
        };
    }
}
