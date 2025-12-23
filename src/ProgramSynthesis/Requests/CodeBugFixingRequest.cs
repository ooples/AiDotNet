using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for repairing code issues.
/// </summary>
public sealed class CodeBugFixingRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.BugFixing;

    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Optional user-provided description of the observed bug (stack trace, failing test name, etc.).
    /// </summary>
    public string? BugDescription { get; set; }
}
