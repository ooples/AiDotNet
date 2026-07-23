namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a structured issue found in code.
/// </summary>
public sealed class CodeIssue
{
    public CodeIssueSeverity Severity { get; set; } = CodeIssueSeverity.Warning;

    public CodeIssueCategory Category { get; set; } = CodeIssueCategory.Other;

    public string Summary { get; set; } = string.Empty;

    public string? Details { get; set; }

    public string? Rationale { get; set; }

    public string? FixGuidance { get; set; }

    public string? TestGuidance { get; set; }

    public CodeLocation Location { get; set; } = new();
}
