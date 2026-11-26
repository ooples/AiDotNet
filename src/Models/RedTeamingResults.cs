namespace AiDotNet.Models;

using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Benchmarking;

/// <summary>
/// Contains results from red teaming a model.
/// </summary>
/// <remarks>
/// <para>
/// Every prompt is a case of one of the library's safety benchmarks (the categories of
/// <see cref="Options.AlignmentMethodOptions{T}.RedTeamingCategories"/>), sent to the model as text. Each response is
/// judged by the safety pipeline: an attack succeeds when the pipeline finds the response unsafe.
/// </para>
/// <para><b>For Beginners:</b> Each entry of the arrays below belongs to the prompt at the same index of
/// <see cref="Prompts"/>.</para>
/// </remarks>
public class RedTeamingResults
{
    /// <summary>
    /// Gets or sets the benchmark prompts that were sent to the model.
    /// </summary>
    public IReadOnlyList<SafetyBenchmarkCase> Prompts { get; set; } = Array.Empty<SafetyBenchmarkCase>();

    /// <summary>
    /// Gets or sets the model's responses to the prompts.
    /// </summary>
    public string[] ModelResponses { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the safety pipeline's report on each response.
    /// </summary>
    public SafetyReport[] ResponseReports { get; set; } = Array.Empty<SafetyReport>();

    /// <summary>
    /// Gets or sets which prompts produced an unsafe response.
    /// </summary>
    public bool[] SuccessfulAttacks { get; set; } = Array.Empty<bool>();

    /// <summary>
    /// Gets or sets the severity of each successful attack, 1 - the response's safety score (0 when it failed).
    /// </summary>
    public double[] SeverityScores { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the category of the most severe finding of each unsafe response (null when it was safe).
    /// </summary>
    public SafetyCategory?[] VulnerabilityCategories { get; set; } = Array.Empty<SafetyCategory?>();

    /// <summary>
    /// Gets or sets the share of prompts that produced an unsafe response.
    /// </summary>
    public double SuccessRate { get; set; }

    /// <summary>
    /// Gets or sets the average severity of successful attacks.
    /// </summary>
    public double AverageSeverity { get; set; }

    /// <summary>
    /// Gets or sets a report for each successful attack.
    /// </summary>
    public List<VulnerabilityReport> Vulnerabilities { get; set; } = new();
}

/// <summary>
/// Detailed report of a specific vulnerability found during red teaming.
/// </summary>
public class VulnerabilityReport
{
    /// <summary>
    /// Gets or sets the safety category of the vulnerability (null when the pipeline reported no finding).
    /// </summary>
    public SafetyCategory? Category { get; set; }

    /// <summary>
    /// Gets or sets the severity (0-1).
    /// </summary>
    public double Severity { get; set; }

    /// <summary>
    /// Gets or sets the description of the vulnerability.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the prompt that triggered it.
    /// </summary>
    public string ExamplePrompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model's unsafe response.
    /// </summary>
    public string ProblematicResponse { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets recommended mitigations.
    /// </summary>
    public string[] Recommendations { get; set; } = Array.Empty<string>();
}
