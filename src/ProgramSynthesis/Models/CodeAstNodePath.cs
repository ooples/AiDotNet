namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a stable, structured path to an AST node.
/// </summary>
public sealed class CodeAstNodePath
{
    public List<CodeAstPathSegment> Segments { get; set; } = new();

    public string? HumanReadablePath { get; set; }
}
