namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeCallGraphEdge
{
    public string Caller { get; set; } = string.Empty;

    public string Callee { get; set; } = string.Empty;
}
