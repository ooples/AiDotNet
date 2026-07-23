namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Location information for an item within code.
/// </summary>
public sealed class CodeLocation
{
    public string? FilePath { get; set; }

    public CodeSpan Span { get; set; } = new();

    public CodeAstNodePath? NodePath { get; set; }
}
