using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for refactoring code without changing behavior.
/// </summary>
public sealed class CodeRefactoringRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Refactoring;

    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Optional high-level refactoring goal (e.g., "extract method", "simplify loops").
    /// </summary>
    public string? Goal { get; set; }
}
