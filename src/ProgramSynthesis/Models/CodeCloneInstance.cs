namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeCloneInstance
{
    public CodeLocation Location { get; set; } = new();

    public string SnippetText { get; set; } = string.Empty;
}
