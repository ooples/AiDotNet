namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeSearchHit
{
    public double Score { get; set; }

    public string SnippetText { get; set; } = string.Empty;

    public CodeLocation Location { get; set; } = new();

    public string? Symbol { get; set; }

    public CodeMatchType MatchType { get; set; } = CodeMatchType.Lexical;

    public CodeProvenance? Provenance { get; set; }

    public string? MatchExplanation { get; set; }
}
