namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Represents a single A/B test configuration.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> An A/B test compares two or more model versions to determine
/// which performs better. This class defines a single test with its versions and traffic allocation.
/// </para>
/// </remarks>
public class ABTest
{
    /// <summary>
    /// Gets or sets the unique name for this test.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of this test.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the control version (baseline) for this test.
    /// </summary>
    public string ControlVersion { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the treatment version (new version being tested).
    /// </summary>
    public string TreatmentVersion { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the traffic percentage for the treatment version (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> What percentage of traffic goes to the new version.
    /// The remaining traffic goes to the control version.
    /// Example: 0.2 means 20% to treatment, 80% to control.
    /// </para>
    /// </remarks>
    public double TreatmentTrafficPercentage { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether this test is active (default: true).
    /// </summary>
    public bool IsActive { get; set; } = true;

    /// <summary>
    /// Gets or sets the start date of the test.
    /// </summary>
    public DateTime? StartDate { get; set; }

    /// <summary>
    /// Gets or sets the end date of the test.
    /// </summary>
    public DateTime? EndDate { get; set; }

    /// <summary>
    /// Gets or sets the primary metric to compare (e.g., "accuracy", "latency").
    /// </summary>
    public string? PrimaryMetric { get; set; }

    /// <summary>
    /// Gets or sets the minimum improvement threshold to consider the treatment a winner (default: 0.01 = 1%).
    /// </summary>
    public double MinimumImprovementThreshold { get; set; } = 0.01;
}
