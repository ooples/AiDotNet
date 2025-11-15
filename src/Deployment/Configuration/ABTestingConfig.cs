namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for A/B testing - comparing multiple model versions by splitting traffic.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A/B testing lets you try out a new model version on a small percentage
/// of users before fully deploying it. This helps you:
/// - Test new models in production safely
/// - Compare performance between versions with real users
/// - Gradually roll out changes to minimize risk
/// - Make data-driven decisions about which model is better
///
/// **How it works:**
/// You specify how to split traffic between versions. For example:
/// - Version 1.0: 80% of traffic (current stable version)
/// - Version 2.0: 20% of traffic (new experimental version)
///
/// Then you monitor metrics like accuracy, latency, and user satisfaction to decide
/// which version is better.
///
/// **Example Use Case:**
/// You've trained a new model that seems more accurate in testing. Instead of deploying
/// it to all users immediately, you start with 10% of traffic. If metrics look good after
/// a week, you increase to 50%, then 100%.
/// </remarks>
public class ABTestingConfig
{
    /// <summary>
    /// Gets or sets whether A/B testing is enabled (default: false).
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the traffic split configuration.
    /// Dictionary maps version name to traffic percentage (0.0 to 1.0).
    /// Example: { "1.0": 0.8, "2.0": 0.2 } = 80% on v1.0, 20% on v2.0
    /// </summary>
    public Dictionary<string, double> TrafficSplit { get; set; } = new();

    /// <summary>
    /// Gets or sets the strategy for assigning users to versions (default: "random").
    /// - "random": Each request randomly assigned based on traffic split
    /// - "sticky": Users consistently get the same version (based on user ID hash)
    /// - "gradual": Gradually shift traffic from old to new version over time
    /// </summary>
    public string AssignmentStrategy { get; set; } = "random";

    /// <summary>
    /// Gets or sets the duration in days for the A/B test (default: 7).
    /// After this duration, you should analyze results and choose a winner.
    /// </summary>
    public int TestDurationDays { get; set; } = 7;

    /// <summary>
    /// Gets or sets whether to track experiment assignment for each request (default: true).
    /// Useful for analysis but adds slight overhead.
    /// </summary>
    public bool TrackAssignments { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum sample size per version before comparing results (default: 1000).
    /// Ensures statistical significance.
    /// </summary>
    public int MinSampleSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the control group version (baseline for comparison).
    /// Typically the current production version.
    /// </summary>
    public string? ControlVersion { get; set; }

    /// <summary>
    /// Creates a simple 90/10 A/B test configuration.
    /// </summary>
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
            AssignmentStrategy = "sticky" // Users get consistent experience
        };
    }

    /// <summary>
    /// Creates a balanced 50/50 A/B test configuration.
    /// </summary>
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
    public static ABTestingConfig Disabled()
    {
        return new ABTestingConfig
        {
            Enabled = false
        };
    }
}
