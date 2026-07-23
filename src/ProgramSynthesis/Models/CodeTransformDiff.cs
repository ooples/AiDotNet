namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Structured diff for code transforms.
/// </summary>
public sealed class CodeTransformDiff
{
    /// <summary>
    /// Machine-applicable edit list.
    /// </summary>
    public List<CodeEditOperation> Edits { get; set; } = new();

    /// <summary>
    /// Optional unified diff representation (may be tier-redacted by Serving).
    /// </summary>
    public string? UnifiedDiff { get; set; }
}
