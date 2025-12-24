namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeSymbol
{
    public string Name { get; set; } = string.Empty;

    public CodeSymbolKind Kind { get; set; } = CodeSymbolKind.Other;

    public CodeLocation Location { get; set; } = new();
}
