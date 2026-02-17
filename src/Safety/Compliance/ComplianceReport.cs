namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Detailed report from regulatory compliance evaluation.
/// </summary>
public class ComplianceReport
{
    /// <summary>Whether the system is compliant with all checked regulations.</summary>
    public bool IsCompliant { get; init; }

    /// <summary>List of compliance violations found.</summary>
    public IReadOnlyList<ComplianceViolation> Violations { get; init; } = Array.Empty<ComplianceViolation>();

    /// <summary>List of compliance recommendations.</summary>
    public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();

    /// <summary>Regulations that were checked.</summary>
    public IReadOnlyList<string> RegulationsChecked { get; init; } = Array.Empty<string>();
}

/// <summary>
/// A specific compliance violation found during evaluation.
/// </summary>
public class ComplianceViolation
{
    /// <summary>The regulation that was violated.</summary>
    public string Regulation { get; init; } = string.Empty;

    /// <summary>The specific article or section violated.</summary>
    public string Article { get; init; } = string.Empty;

    /// <summary>Description of the violation.</summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>Severity of the violation.</summary>
    public string Severity { get; init; } = string.Empty;
}
