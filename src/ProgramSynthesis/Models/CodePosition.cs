namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a position in source text.
/// </summary>
public sealed class CodePosition
{
    /// <summary>
    /// 1-based line number.
    /// </summary>
    public int Line { get; set; } = 1;

    /// <summary>
    /// 1-based column number.
    /// </summary>
    public int Column { get; set; } = 1;

    /// <summary>
    /// 0-based absolute offset from the start of the text.
    /// </summary>
    public int Offset { get; set; }
}
