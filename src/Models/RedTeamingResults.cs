namespace AiDotNet.Models;

/// <summary>
/// Contains results from red teaming adversarial testing.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class RedTeamingResults<T>
{
    /// <summary>
    /// Gets or sets the adversarial prompts that were tested.
    /// </summary>
    public T[][] AdversarialPrompts { get; set; } = Array.Empty<T[]>();

    /// <summary>
    /// Gets or sets the model's responses to adversarial prompts.
    /// </summary>
    public T[][] ModelResponses { get; set; } = Array.Empty<T[]>();

    /// <summary>
    /// Gets or sets which prompts successfully caused misaligned behavior.
    /// </summary>
    public bool[] SuccessfulAttacks { get; set; } = Array.Empty<bool>();

    /// <summary>
    /// Gets or sets the severity scores for each vulnerability (0-1).
    /// </summary>
    public double[] SeverityScores { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the types of vulnerabilities found.
    /// </summary>
    public string[] VulnerabilityTypes { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the overall red teaming success rate.
    /// </summary>
    public double SuccessRate { get; set; }

    /// <summary>
    /// Gets or sets the average severity of successful attacks.
    /// </summary>
    public double AverageSeverity { get; set; }

    /// <summary>
    /// Gets or sets detailed descriptions of vulnerabilities found.
    /// </summary>
    public List<VulnerabilityReport> Vulnerabilities { get; set; } = new();
}

/// <summary>
/// Detailed report of a specific vulnerability found during red teaming.
/// </summary>
public class VulnerabilityReport
{
    /// <summary>
    /// Gets or sets the type of vulnerability.
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the severity score (0-1).
    /// </summary>
    public double Severity { get; set; }

    /// <summary>
    /// Gets or sets the description of the vulnerability.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets an example prompt that triggers the vulnerability.
    /// </summary>
    public string ExamplePrompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the problematic model response.
    /// </summary>
    public string ProblematicResponse { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets recommendations for fixing the vulnerability.
    /// </summary>
    public string[] Recommendations { get; set; } = Array.Empty<string>();
}
