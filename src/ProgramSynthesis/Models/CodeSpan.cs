namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a span in source text.
/// </summary>
public sealed class CodeSpan
{
    public CodePosition Start { get; set; } = new();

    public CodePosition End { get; set; } = new();
}
