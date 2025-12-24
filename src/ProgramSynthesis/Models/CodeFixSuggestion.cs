namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeFixSuggestion
{
    public string Summary { get; set; } = string.Empty;

    public string? Rationale { get; set; }

    public string? FixGuidance { get; set; }

    public string? TestGuidance { get; set; }

    public CodeTransformDiff? Diff { get; set; }
}
