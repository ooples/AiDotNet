namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// A structured segment in an AST node path.
/// </summary>
public sealed class CodeAstPathSegment
{
    public string Kind { get; set; } = string.Empty;

    public string? Name { get; set; }

    public int? Index { get; set; }
}
